"""Write the summary from a label record.

Every closed-set value maps to one fixed sentence, written here and nowhere else.
The labeller's own words enter only through cited free-text fields, so rendering
cannot add a fact the record does not hold.
"""

import re

INCOMPLETE_SENTENCE = "This call was not completed to get all the information as needed."
TRANSFER_BEFORE_REASON_SENTENCE = "The caller was transferred to another team for assistance."
CIRCUMSTANCES_NOT_ASKED_SENTENCE = (
    "Agent did not ask about the circumstances that led to the reported issue."
)

CALLER_PHRASE = {
    "patient": "a patient",
    "patients_representative": "a patient's representative",
    "healthcare_provider": "a healthcare provider",
    "manufacturers_representative": "a manufacturer's representative",
    "third_party": "a third party",
}
DEVICE_PHRASE = {
    "external": "an external device",
    "implant": "an implantable neurostimulator",
    "implant_and_external": "an implantable neurostimulator and an external device",
}
OUTCOME_SENTENCE = {
    "resolved": "The troubleshooting steps that were taken on the call resolved the issue.",
    "not_resolved": "The issue was not resolved through troubleshooting.",
    "none_needed": "No troubleshooting was required.",
}


def caller_noun(caller_type: str) -> str:
    return "the patient" if caller_type == "patient" else "the caller"


def _sentence(text: str) -> str:
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def _clause(text: str) -> str:
    """A fragment that follows fixed words, e.g. after 'The reason for call was'."""
    text = " ".join((text or "").split()).strip().rstrip(".")
    return text[0].lower() + text[1:] if text and not text[:2].isupper() else text


def _device_id_sentence(device: dict) -> str:
    serial = re.sub(r"[^A-Za-z0-9]", "", device["serial"]).upper()
    model = (device.get("model") or "").strip()
    name = _sentence(device["name"]).rstrip(".")
    return f"{name} ({model}) s/n {serial}." if model else f"{name} s/n {serial}."


def _action_sentence(action: dict, caller: str) -> str:
    detail = _clause(action.get("detail") or "")
    kind = action["type"]
    if kind == "replacement":
        return f"An email was sent to the repair department to replace the {detail or 'device'}."
    if kind == "redirected_hcp":
        return "The patient was redirected to their healthcare provider to further address the issue."
    if kind == "rep_informed":
        return "Agent sent an email to the Medtronic representative for visibility."
    if kind == "call_back":
        return _sentence(f"{caller} will call back {detail}".strip())
    if kind == "record_update":
        return f"Agent updated the patient's {detail or 'record'}."
    raise ValueError(f"unknown action type {kind!r}")


def _issue_sentences(issue: dict, record: dict, is_first: bool) -> list[str]:
    caller = caller_noun(record["caller_type"])
    out = [
        f"Information was received from {CALLER_PHRASE[record['caller_type']]} "
        f"regarding {DEVICE_PHRASE[issue['device_type']]}."
    ]
    party = (record["transfer_in"].get("party") or "").strip()
    if is_first and party:
        out.append(f"The call was transferred in from {party}.")
    out += [_device_id_sentence(d) for d in issue["device_ids"]]

    if issue["reason"].get("text"):
        out.append(f"The reason for call was {_clause(issue['reason']['text'])}.")
    if issue["onset"].get("text"):
        out.append(_sentence(issue["onset"]["text"]))

    circumstances = issue["circumstances"]
    if circumstances["status"] == "asked" and circumstances.get("text"):
        out.append(_sentence(circumstances["text"]))
    elif circumstances["status"] == "not_asked":
        out.append(CIRCUMSTANCES_NOT_ASKED_SENTENCE)

    out += [_sentence(step["sentence"]) for step in issue["troubleshooting"]]

    outcome = issue["outcome"]
    if outcome == "transferred_out":
        team = (issue["transfer_out"].get("team") or "").strip() or "another team"
        out.append(f"The caller was transferred to {team} for further assistance.")
    elif outcome == "not_possible":
        reason = _clause(issue["outcome_reason"].get("text") or "")
        out.append(
            f"Troubleshooting was unable to be performed because {reason}."
            if reason
            else "Troubleshooting was unable to be performed."
        )
    else:
        out.append(OUTCOME_SENTENCE[outcome])

    out += [_action_sentence(a, caller) for a in issue["actions"]]
    return out


def split_sentences(summary: str) -> list[str]:
    # Split on sentence ends, but not inside numbers ("3.6") or "s/n".
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", summary.strip())
    return [p.strip() for p in parts if p.strip()]


def render(record: dict) -> str:
    status = record["call_status"]
    if status == "incomplete":
        return INCOMPLETE_SENTENCE
    if status == "transfer_out_before_reason":
        opening = f"Information was received from {CALLER_PHRASE[record['caller_type']]}."
        return f"{opening} {TRANSFER_BEFORE_REASON_SENTENCE}"

    # One block per issue, then the other statements. Repeats are removed within a
    # block only: each issue legitimately repeats the opening and fixed sentences.
    blocks = [_issue_sentences(issue, record, is_first=i == 0) for i, issue in enumerate(record["issues"])]
    blocks.append([_sentence(s["sentence"]) for s in record["other_statements"]])

    out = []
    for block in blocks:
        seen = set()
        for s in block:
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    return " ".join(out)
