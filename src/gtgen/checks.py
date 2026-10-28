"""Checks 1 to 3: schema, evidence quotes, and rules. All deterministic, no model calls.

A check returns problems with a severity:
  fail   - the label is wrong as it stands; regenerate, then reject
  review - the label may be right, but a person should look before it is used
"""

import difflib
import re
from dataclasses import dataclass

import jsonschema

from .preprocess import PreparedCall
from .schema import LABEL_RECORD

QUOTE_MATCH_THRESHOLD = 0.9

# Same phrases the service's get_summary() used to detect transfers, plus "transfer them".
# Here they only confirm that a transfer the labeller reports was actually said.
TRANSFER_RE = re.compile(
    "|".join(
        [
            r"let me get you to",
            r"let me get you over",
            r"connect you (?:with|over|to)",
            r"get you (?:right )?over to",
            r"get you to the right",
            r"transfer(?:ring)? (?:you|them|the call|your call)",
            r"not trained (?:with|on)",
            r"i'?m not trained",
            r"you'?re on the .{0,25}line",
            r"wrong (?:line|department|queue|number)",
            r"i need to get you there",
            r"right group of people",
            r"the right (?:team|group|department|spot|people)",
            r"i'?ll (?:get|put) you (?:through|over)",
            r"put you through to",
        ]
    )
)

# Words the agent's quote must contain for each action type; (pattern, severity).
ACTION_WORDS = {
    "replacement": (r"replac|new one|exchange|send|ship|overnight|fedex", "fail"),
    "redirected_hcp": (r"doctor|physician|provider|clinician|surgeon|\bdr\b|office", "fail"),
    "rep_informed": (r"\brep\b|representative|email", "review"),
    "call_back": (r"call (?:us )?back|callback|give us a call|call in", "review"),
    "record_update": (r"updat|chang|\badd|record|address|number|card|file", "review"),
}

BANNED_STRINGS = ["healthcare plan"]
PII_PATTERNS = {
    "phone number": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "date of birth": re.compile(r"\b(?:date of birth|born on|dob)\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I),
    "street address": re.compile(
        r"\b\d{2,6}\s+(?:\w+\s+){0,3}(?:street|avenue|road|drive|boulevard|lane|parkway|court)\b", re.I
    ),
}

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
IGNORED_ID_WORDS = {"point", "dash", "hyphen", "dot", "slash"}


@dataclass
class Problem:
    check: str
    severity: str
    message: str

    def as_dict(self) -> dict:
        return {"check": self.check, "severity": self.severity, "message": self.message}


def tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", (text or "").lower()).split()


def quote_match(quote: str, source: str) -> float:
    """Share of the quote's tokens found, in order, in the source text."""
    q, s = tokens(quote), tokens(source)
    if not q:
        return 0.0
    matcher = difflib.SequenceMatcher(None, q, s, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(q)


def spoken_id_stream(text: str, keep_articles: bool = True) -> str:
    """Read spelled-out letters and numbers as one string.

    "n l d one zero seven nine one seven n as in nancy" -> "NLD107917N".
    Words that are not part of an ID are skipped, so an ID read across two turns
    still joins up. Without `keep_articles`, a lone "a" or "i" is skipped, since it
    is usually the article or pronoun, not a letter.
    """
    words = tokens(text)
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w == "as" and i + 1 < len(words) and words[i + 1] == "in":
            i += 3  # "n as in nancy": the letter was already taken
            continue
        if w == "double" and i + 1 < len(words) and words[i + 1] in DIGIT_WORDS:
            out.append(DIGIT_WORDS[words[i + 1]] * 2)
            i += 2
            continue
        if w in DIGIT_WORDS:
            out.append(DIGIT_WORDS[w])
        elif len(w) == 1 and w.isalpha():
            spelled = i + 1 < len(words) and words[i + 1] == "as"
            if keep_articles or w not in ("a", "i") or spelled:
                out.append(w.upper())
        elif w.isalnum() and any(c.isdigit() for c in w):
            out.append(w.upper())
        elif w in IGNORED_ID_WORDS:
            pass
        i += 1
    return "".join(out)


def id_in_turns(value: str, source: str) -> bool:
    value = re.sub(r"[^A-Za-z0-9]", "", value or "").upper()
    if not value:
        return False
    return value in spoken_id_stream(source) or value in spoken_id_stream(source, keep_articles=False)


def check_schema(record: dict) -> list[Problem]:
    validator = jsonschema.Draft202012Validator(LABEL_RECORD)
    return [
        Problem("schema", "fail", f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}")
        for e in validator.iter_errors(record)
    ]


class _Evidence:
    """Collects quote problems for one record against one prepared call."""

    def __init__(self, call: PreparedCall):
        self.turns = call.turn_map()
        self.problems: list[Problem] = []

    def source(self, turn_ids: list[str]) -> str:
        return " ".join(self.turns[t].text for t in turn_ids if t in self.turns)

    def speakers(self, turn_ids: list[str]) -> set[str]:
        return {self.turns[t].speaker.lower() for t in turn_ids if t in self.turns}

    def cited(self, where: str, turn_ids: list[str], quote: str | None) -> bool:
        """The quote must exist, cite real turns, and match those turns."""
        if not quote:
            self.problems.append(Problem("quotes", "fail", f"{where}: fact has no quote"))
            return False
        unknown = [t for t in turn_ids if t not in self.turns]
        if not turn_ids or unknown:
            self.problems.append(
                Problem("quotes", "fail", f"{where}: cites unknown or no turns {unknown or turn_ids}")
            )
            return False
        score = quote_match(quote, self.source(turn_ids))
        if score < QUOTE_MATCH_THRESHOLD:
            self.problems.append(
                Problem("quotes", "fail", f"{where}: quote not found in {turn_ids} (match {score:.2f}): {quote!r}")
            )
            return False
        return True

    def agent_said(self, where: str, turn_ids: list[str], quote: str | None) -> bool:
        if not self.cited(where, turn_ids, quote):
            return False
        if self.speakers(turn_ids) - {"agent"}:
            self.problems.append(Problem("rules", "fail", f"{where}: must quote the agent's own turns"))
            return False
        return True


def check_evidence_and_rules(record: dict, call: PreparedCall) -> list[Problem]:
    ev = _Evidence(call)
    problems = ev.problems
    status = record["call_status"]

    if status == "transfer_out_before_reason":
        se = record["status_evidence"]
        if ev.agent_said("status_evidence", se["turns"], se["quote"]) and not TRANSFER_RE.search(se["quote"]):
            problems.append(Problem("rules", "fail", "transfer before reason: quote has no transfer phrase"))
    if status == "complete" and not record["issues"]:
        problems.append(Problem("rules", "fail", "complete call with no issues"))
    if status != "complete" and record["issues"]:
        problems.append(Problem("rules", "fail", f"{status} call must not list issues"))

    ti = record["transfer_in"]
    if ti.get("party"):
        ev.cited("transfer_in", ti["turns"], ti["quote"])

    reported_transfer = status == "transfer_out_before_reason"
    for n, issue in enumerate(record["issues"], start=1):
        at = f"issue {n}"
        for name in ("reason", "onset"):
            fact = issue[name]
            if fact.get("text"):
                ev.cited(f"{at} {name}", fact["turns"], fact["quote"])
        if not issue["reason"].get("text"):
            problems.append(Problem("rules", "review", f"{at}: no reason for call"))

        circ = issue["circumstances"]
        if circ["status"] == "asked" and circ.get("text"):
            ev.cited(f"{at} circumstances", circ["turns"], circ["quote"])

        for k, step in enumerate(issue["troubleshooting"], start=1):
            ev.cited(f"{at} troubleshooting {k}", step["turns"], step["quote"])

        for k, device in enumerate(issue["device_ids"], start=1):
            where = f"{at} device {k}"
            if not ev.cited(where, device["turns"], device["quote"]):
                continue
            source = ev.source(device["turns"])
            if not id_in_turns(device["serial"], source):
                problems.append(Problem("quotes", "fail", f"{where}: serial {device['serial']!r} not read in {device['turns']}"))
            if device.get("model") and not id_in_turns(device["model"], source):
                problems.append(Problem("quotes", "fail", f"{where}: model {device['model']!r} not read in {device['turns']}"))

        if issue["outcome"] == "transferred_out":
            reported_transfer = True
            to = issue["transfer_out"]
            if ev.agent_said(f"{at} transfer", to["turns"], to["quote"]) and not TRANSFER_RE.search(to["quote"]):
                problems.append(Problem("rules", "fail", f"{at}: transfer quote has no transfer phrase"))
        if issue["outcome"] == "not_possible" and issue["outcome_reason"].get("text"):
            orr = issue["outcome_reason"]
            ev.cited(f"{at} outcome reason", orr["turns"], orr["quote"])

        types = [a["type"] for a in issue["actions"]]
        for k, action in enumerate(issue["actions"], start=1):
            where = f"{at} action {k} ({action['type']})"
            if ev.agent_said(where, action["turns"], action["quote"]):
                pattern, severity = ACTION_WORDS[action["type"]]
                if not re.search(pattern, action["quote"].lower()):
                    problems.append(Problem("rules", severity, f"{where}: quote does not state this action"))
        if issue["outcome"] == "resolved" and "replacement" in types:
            problems.append(Problem("rules", "review", f"{at}: resolved but a replacement was sent"))

    for k, s in enumerate(record["other_statements"], start=1):
        ev.cited(f"other statement {k}", s["turns"], s["quote"])

    agent_transfer = any(
        t.speaker.lower() == "agent" and TRANSFER_RE.search(t.text) for t in call.turns
    )
    if agent_transfer and not reported_transfer:
        problems.append(Problem("rules", "review", "agent used transfer language but no transfer is recorded"))
    if record["unreadable_share"] == "high":
        problems.append(Problem("rules", "review", "most of the transcript is unreadable"))
    return problems


def check_summary_text(summary: str) -> list[Problem]:
    problems = []
    lower = summary.lower()
    for banned in BANNED_STRINGS:
        if banned in lower:
            problems.append(Problem("rules", "fail", f"summary contains banned string {banned!r}"))
    for kind, pattern in PII_PATTERNS.items():
        if pattern.search(summary):
            problems.append(Problem("rules", "fail", f"summary may contain a {kind}"))
    # Repeated sentences are removed by the renderer, so they are not checked here.
    return problems


def failures(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == "fail"]
