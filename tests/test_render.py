import copy

from gtgen.render import INCOMPLETE_SENTENCE, render, split_sentences


def test_replacement_example(replacement_example):
    _, record = replacement_example
    summary = render(record)
    assert summary.startswith("Information was received from a patient regarding an external device.")
    assert "Controller s/n NLD107917N." in summary
    assert "The issue was not resolved through troubleshooting." in summary
    assert summary.endswith("An email was sent to the repair department to replace the recharger cord.")


def test_transfer_before_reason(transfer_example):
    _, record = transfer_example
    assert render(record) == (
        "Information was received from a patient. The caller was transferred to another team for assistance."
    )


def test_transfer_after_information(replacement_example):
    _, record = replacement_example
    issue = record["issues"][0]
    issue["outcome"] = "transferred_out"
    issue["transfer_out"] = {"team": "technical services", "turns": ["T9"], "quote": "x"}
    issue["actions"] = []
    assert render(record).endswith("The caller was transferred to technical services for further assistance.")


def test_incomplete():
    record = {"call_status": "incomplete", "caller_type": "patient", "issues": []}
    assert render(record) == INCOMPLETE_SENTENCE


def test_two_issues_repeat_the_opening(replacement_example):
    _, record = replacement_example
    record["issues"].append(copy.deepcopy(record["issues"][0]))
    summary = render(record)
    assert summary.count("Information was received from a patient regarding an external device.") == 2


def test_transfer_in_follows_the_opening(replacement_example):
    _, record = replacement_example
    record["transfer_in"] = {"party": "SunMed", "turns": ["T1"], "quote": "x"}
    sentences = split_sentences(render(record))
    assert sentences[1] == "The call was transferred in from SunMed."


def test_split_sentences_keeps_numbers_together():
    assert split_sentences("Software Problem (1-Intellis, 2-3.6) was shown. The patient called.") == [
        "Software Problem (1-Intellis, 2-3.6) was shown.",
        "The patient called.",
    ]
