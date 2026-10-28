import copy
from dataclasses import replace

import pytest

from gtgen.config import load_settings
from gtgen.llm import LLMResult
from gtgen.pipeline import Pipeline
from gtgen.render import render, split_sentences


class FakeLLM:
    """Returns queued answers per schema name, in order."""

    def __init__(self, answers: dict[str, list]):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls = []

    def json_call(self, *, model, system, user, schema, schema_name, **kwargs):
        self.calls.append((schema_name, user))
        return LLMResult(data=copy.deepcopy(self.answers[schema_name].pop(0)), model=model, input_tokens=10, output_tokens=5)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.chdir("/")  # no .env picked up
    s = load_settings()
    return replace(s, labeller_model="labeller", judge_model="judge", closed_set_model="small", max_label_attempts=2)


def all_supported(record):
    n = len(split_sentences(render(record)))
    return {"sentences": [{"index": i, "verdict": "supported", "turns": [], "reason": ""} for i in range(1, n + 1)], "missing": []}


CLOSED_SET_REPLACEMENT = {
    "call_status": "complete", "caller_type": "patient", "outcomes": ["not_resolved"],
    "action_types": ["replacement"], "transferred_out": False, "transferred_in": False,
}


def test_clean_label_is_accepted(settings, replacement_example):
    call, record = replacement_example
    llm = FakeLLM({"label_record": [record], "judge_result": [all_supported(record)], "closed_set_result": [CLOSED_SET_REPLACEMENT]})
    out = Pipeline(settings, llm, []).run(call)
    assert out["status"] == "accepted", out["problems"]
    assert out["attempts"] == 1
    assert out["summary"] == render(record)


def test_bad_quote_is_retried_with_feedback(settings, replacement_example):
    call, record = replacement_example
    bad = copy.deepcopy(record)
    bad["issues"][0]["onset"]["quote"] = "it started after a fall"
    llm = FakeLLM({"label_record": [bad, record], "judge_result": [all_supported(record)], "closed_set_result": [CLOSED_SET_REPLACEMENT]})
    out = Pipeline(settings, llm, []).run(call)
    assert out["status"] == "accepted"
    assert out["attempts"] == 2
    retry_prompt = [u for name, u in llm.calls if name == "label_record"][1]
    assert "quote not found" in retry_prompt


def test_unsupported_sentence_twice_is_rejected(settings, replacement_example):
    call, record = replacement_example
    verdict = all_supported(record)
    verdict["sentences"][2]["verdict"] = "unsupported"
    llm = FakeLLM({"label_record": [record, record], "judge_result": [verdict, verdict]})
    out = Pipeline(settings, llm, []).run(call)
    assert out["status"] == "rejected"
    assert any(p["check"] == "judge" for p in out["problems"])


def test_failed_rare_call_type_goes_to_review(settings, transfer_example):
    call, record = transfer_example
    verdict = all_supported(record)
    verdict["sentences"][1]["verdict"] = "contradicted"
    llm = FakeLLM({"label_record": [record, record], "judge_result": [verdict, verdict]})
    out = Pipeline(settings, llm, []).run(call)
    assert out["call_type"] == "transfer_out_before_reason"
    assert out["status"] == "review"


def test_closed_set_disagreement_goes_to_review(settings, replacement_example):
    call, record = replacement_example
    second = dict(CLOSED_SET_REPLACEMENT, outcomes=["resolved"])
    llm = FakeLLM({"label_record": [record], "judge_result": [all_supported(record)], "closed_set_result": [second]})
    out = Pipeline(settings, llm, []).run(call)
    assert out["status"] == "review"
    assert any("outcomes" in p["message"] for p in out["problems"])


def test_missing_fact_retries_then_flags(settings, replacement_example):
    call, record = replacement_example
    verdict = all_supported(record)
    verdict["missing"] = [{"field": "device_id", "description": "recharger serial", "turns": []}]
    llm = FakeLLM({"label_record": [record, record], "judge_result": [verdict, verdict], "closed_set_result": [CLOSED_SET_REPLACEMENT]})
    out = Pipeline(settings, llm, []).run(call)
    assert out["status"] == "accepted"
    assert "incomplete_coverage" in out["flags"]
    assert out["attempts"] == 2
