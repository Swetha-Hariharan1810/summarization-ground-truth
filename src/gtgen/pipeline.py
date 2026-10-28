"""One call through the whole pipeline: label, check, render, judge, closed-set pass.

Outcome per call:
  accepted - passed every check; may carry the flag `incomplete_coverage`
  review   - needs a person: a review-level problem, a closed-set disagreement, or a
             failed label on a rare call type (those are too valuable to drop)
  rejected - still failing after the allowed attempts
  skipped  - not labelled (empty transcript, under the service's minimum length)
"""

import hashlib
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .checks import Problem, check_evidence_and_rules, check_schema, check_summary_text, failures
from .config import Settings
from .llm import LLMClient, LLMError, Usage
from .preprocess import PreparedCall
from .render import render, split_sentences
from .schema import CLOSED_SET_RESULT, JUDGE_RESULT, LABEL_RECORD

HINT_HEADER = (
    "An agent wrote the note below after the call. It may be wrong or incomplete, and it may "
    "describe things that happened after the call. Use it only to notice facts you might have "
    "missed; include a fact only if you can quote the transcript for it."
)


def _prompt(name: str) -> str:
    return resources.files("gtgen").joinpath("prompts", name).read_text()


def load_examples(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def labeller_system_prompt(examples: list[dict]) -> str:
    parts = [_prompt("labeller_system.md")]
    if examples:
        parts.append("## Worked examples")
        for ex in examples:
            parts.append(
                f"### {ex['name']}\n\nTranscript:\n{ex['transcript']}\n\n"
                f"Record:\n```json\n{json.dumps(ex['record'], indent=1)}\n```"
            )
    return "\n\n".join(parts)


def call_type(record: dict | None) -> str:
    if not record:
        return "unknown"
    if record["call_status"] != "complete":
        return record["call_status"]
    if any(i["outcome"] == "transferred_out" for i in record["issues"]):
        return "transfer_out_after_info"
    if record["transfer_in"].get("party"):
        return "transfer_in"
    return "multi_issue" if len(record["issues"]) > 1 else "single_issue"


RARE_TYPES = {"incomplete", "transfer_out_before_reason", "transfer_out_after_info", "transfer_in", "multi_issue"}


def closed_set_view(record: dict) -> dict:
    outcomes = sorted(i["outcome"] for i in record["issues"])
    actions = sorted({a["type"] for i in record["issues"] for a in i["actions"]})
    return {
        "call_status": record["call_status"],
        "caller_type": record["caller_type"],
        "outcomes": outcomes,
        "action_types": actions,
        "transferred_out": record["call_status"] == "transfer_out_before_reason" or "transferred_out" in outcomes,
        "transferred_in": bool(record["transfer_in"].get("party")),
    }


def closed_set_disagreements(record: dict, second: dict) -> list[Problem]:
    mine = closed_set_view(record)
    theirs = dict(second)
    theirs["outcomes"] = sorted(theirs["outcomes"])
    theirs["action_types"] = sorted(set(theirs["action_types"]))
    return [
        Problem("closed_set", "review", f"{key}: label says {mine[key]!r}, second pass says {theirs[key]!r}")
        for key in mine
        if mine[key] != theirs.get(key)
    ]


@dataclass
class Attempt:
    record: dict | None = None
    summary: str = ""
    judge: dict | None = None
    problems: list[Problem] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.record is not None and not failures(self.problems)


class Pipeline:
    def __init__(self, settings: Settings, llm: LLMClient, examples: list[dict]):
        self.settings = settings
        self.llm = llm
        self.labeller_system = labeller_system_prompt(examples)
        self.judge_system = _prompt("judge_system.md")
        self.closed_set_system = _prompt("closed_set_system.md")
        digest = hashlib.sha256(
            "\n".join([self.labeller_system, self.judge_system, self.closed_set_system]).encode()
        ).hexdigest()
        self.prompt_version = digest[:12]

    # --- model calls -----------------------------------------------------------

    def _label(self, call: PreparedCall, hint: str | None, feedback: list[str], previous: dict | None, usage: Usage) -> dict:
        user = f"Transcript:\n{call.numbered_transcript()}"
        if hint:
            user += f"\n\n{HINT_HEADER}\n\nAgent note:\n{hint}"
        if feedback:
            user += (
                "\n\nYour previous record for this call had these problems. Return a corrected record. "
                "Where a fact cannot be quoted from the transcript, leave it out.\n- "
                + "\n- ".join(feedback)
                + f"\n\nPrevious record:\n{json.dumps(previous)}"
            )
        result = self.llm.json_call(
            model=self.settings.labeller_model,
            system=self.labeller_system,
            user=user,
            schema=LABEL_RECORD,
            schema_name="label_record",
            reasoning_effort=self.settings.labeller_reasoning_effort,
        )
        usage.add(result)
        return result.data

    def _judge(self, call: PreparedCall, sentences: list[str], usage: Usage) -> dict:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
        result = self.llm.json_call(
            model=self.settings.judge_model,
            system=self.judge_system,
            user=f"Transcript:\n{call.numbered_transcript()}\n\nSummary sentences:\n{numbered}",
            schema=JUDGE_RESULT,
            schema_name="judge_result",
            reasoning_effort=self.settings.judge_reasoning_effort,
        )
        usage.add(result)
        return result.data

    def _closed_set(self, call: PreparedCall, usage: Usage) -> dict:
        result = self.llm.json_call(
            model=self.settings.closed_set_model,
            system=self.closed_set_system,
            user=f"Transcript:\n{call.numbered_transcript()}",
            schema=CLOSED_SET_RESULT,
            schema_name="closed_set_result",
            max_output_tokens=4000,
        )
        usage.add(result)
        return result.data

    # --- one attempt -----------------------------------------------------------

    def _attempt(self, call, hint, feedback, previous, usage) -> Attempt:
        att = Attempt()
        try:
            att.record = self._label(call, hint, feedback, previous, usage)
        except LLMError as e:
            att.problems.append(Problem("labeller", "fail", str(e)))
            return att

        att.problems += check_schema(att.record)
        if att.problems:
            return att
        att.problems += check_evidence_and_rules(att.record, call)
        if failures(att.problems):
            return att

        att.summary = render(att.record)
        att.problems += check_summary_text(att.summary)
        if failures(att.problems):
            return att

        sentences = split_sentences(att.summary)
        try:
            att.judge = self._judge(call, sentences, usage)
        except LLMError as e:
            att.problems.append(Problem("judge", "review", f"judge failed: {e}"))
            return att
        verdicts = {v["index"]: v for v in att.judge["sentences"]}
        for i, sentence in enumerate(sentences, start=1):
            v = verdicts.get(i)
            if v is None:
                att.problems.append(Problem("judge", "review", f"no verdict for sentence {i}"))
            elif v["verdict"] != "supported":
                att.problems.append(
                    Problem("judge", "fail", f"sentence {i} {v['verdict']} ({v['reason']}): {sentence!r}")
                )
        att.missing = att.judge["missing"]
        return att

    # --- the call ----------------------------------------------------------------

    def run(self, call: PreparedCall, hint: str | None = None) -> dict:
        usage = Usage()
        out = {
            "call_id": call.call_id,
            "status": None,
            "call_type": "unknown",
            "flags": [],
            "problems": [],
            "attempts": 0,
            "summary": "",
            "record": None,
            "judge": None,
            "closed_set": None,
            "prompt_version": self.prompt_version,
            "models": {
                "labeller": self.settings.labeller_model,
                "judge": self.settings.judge_model,
                "closed_set": self.settings.closed_set_model,
            },
        }
        if call.skip_reason:
            out.update(status="skipped", flags=[call.skip_reason])
            return out

        attempts: list[Attempt] = []
        feedback: list[str] = []
        for _ in range(max(1, self.settings.max_label_attempts)):
            previous = attempts[-1].record if attempts else None
            att = self._attempt(call, hint, feedback, previous, usage)
            attempts.append(att)
            if att.passed and not att.missing:
                break
            feedback = [p.message for p in failures(att.problems)] + [
                f"missing {m['field']}: {m['description']}" for m in att.missing
            ]

        # Prefer a passing attempt with nothing missing, then any passing attempt.
        passing = [a for a in attempts if a.passed]
        best = next((a for a in passing if not a.missing), passing[0] if passing else attempts[-1])

        out.update(
            attempts=len(attempts),
            record=best.record,
            summary=best.summary,
            judge=best.judge,
            call_type=call_type(best.record),
        )
        problems = list(best.problems)

        if not best.passed:
            rare = out["call_type"] in RARE_TYPES
            out["status"] = "review" if rare else "rejected"
        else:
            if best.missing:
                out["flags"].append("incomplete_coverage")
            try:
                out["closed_set"] = self._closed_set(call, usage)
                problems += closed_set_disagreements(best.record, out["closed_set"])
            except LLMError as e:
                problems.append(Problem("closed_set", "review", f"closed-set pass failed: {e}"))
            out["status"] = "review" if problems else "accepted"

        out["problems"] = [p.as_dict() for p in problems]
        out["usage"] = {
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "by_model": usage.by_model,
        }
        return out
