"""JSON schemas for the three model outputs: label record, judge verdicts, closed-set pass.

The schemas are written for OpenAI structured outputs in strict mode: every property
is listed in `required`, optional values are nullable, and no extra keys are allowed.
They are also used by check 1 to validate whatever comes back.
"""

CALLER_TYPES = [
    "patient",
    "patients_representative",
    "healthcare_provider",
    "manufacturers_representative",
    "third_party",
]
DEVICE_TYPES = ["external", "implant", "implant_and_external"]
CALL_STATUSES = ["complete", "incomplete", "transfer_out_before_reason"]
OUTCOMES = ["resolved", "not_resolved", "not_possible", "none_needed", "transferred_out"]
ACTION_TYPES = ["replacement", "redirected_hcp", "rep_informed", "call_back", "record_update"]
CIRCUMSTANCES = ["asked", "not_asked", "not_applicable"]
MISSING_FIELDS = ["reason", "outcome", "action", "transfer", "device_id"]

NULL_STR = {"type": ["string", "null"]}
TURNS = {"type": "array", "items": {"type": "string"}}


def _obj(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


# A free-text fact. `text` is null when the transcript does not state it.
FACT = _obj({"text": NULL_STR, "turns": TURNS, "quote": NULL_STR})

# A sentence written by the labeller (troubleshooting, other statements).
SENTENCE = _obj({"sentence": {"type": "string"}, "turns": TURNS, "quote": {"type": "string"}})

EVIDENCE = _obj({"turns": TURNS, "quote": NULL_STR})

DEVICE_ID = _obj(
    {
        "name": {"type": "string"},
        "model": NULL_STR,
        "serial": {"type": "string"},
        "turns": TURNS,
        "quote": {"type": "string"},
    }
)

ACTION = _obj(
    {
        "type": {"type": "string", "enum": ACTION_TYPES},
        "detail": NULL_STR,
        "turns": TURNS,
        "quote": {"type": "string"},
    }
)

ISSUE = _obj(
    {
        "device_type": {"type": "string", "enum": DEVICE_TYPES},
        "device_ids": {"type": "array", "items": DEVICE_ID},
        "reason": FACT,
        "onset": FACT,
        "circumstances": _obj(
            {
                "status": {"type": "string", "enum": CIRCUMSTANCES},
                "text": NULL_STR,
                "turns": TURNS,
                "quote": NULL_STR,
            }
        ),
        "troubleshooting": {"type": "array", "items": SENTENCE},
        "outcome": {"type": "string", "enum": OUTCOMES},
        "outcome_reason": FACT,
        "transfer_out": _obj({"team": NULL_STR, "turns": TURNS, "quote": NULL_STR}),
        "actions": {"type": "array", "items": ACTION},
    }
)

LABEL_RECORD = _obj(
    {
        "call_status": {"type": "string", "enum": CALL_STATUSES},
        "status_evidence": EVIDENCE,
        "caller_type": {"type": "string", "enum": CALLER_TYPES},
        "transfer_in": _obj({"party": NULL_STR, "turns": TURNS, "quote": NULL_STR}),
        "issues": {"type": "array", "items": ISSUE},
        "other_statements": {"type": "array", "items": SENTENCE},
        "unreadable_share": {"type": "string", "enum": ["low", "medium", "high"]},
        "labeller_notes": {"type": "string"},
    }
)

JUDGE_RESULT = _obj(
    {
        "sentences": {
            "type": "array",
            "items": _obj(
                {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
                    "turns": TURNS,
                    "reason": {"type": "string"},
                }
            ),
        },
        "missing": {
            "type": "array",
            "items": _obj(
                {
                    "field": {"type": "string", "enum": MISSING_FIELDS},
                    "description": {"type": "string"},
                    "turns": TURNS,
                }
            ),
        },
    }
)

CLOSED_SET_RESULT = _obj(
    {
        "call_status": {"type": "string", "enum": CALL_STATUSES},
        "caller_type": {"type": "string", "enum": CALLER_TYPES},
        "outcomes": {"type": "array", "items": {"type": "string", "enum": OUTCOMES}},
        "action_types": {"type": "array", "items": {"type": "string", "enum": ACTION_TYPES}},
        "transferred_out": {"type": "boolean"},
        "transferred_in": {"type": "boolean"},
    }
)
