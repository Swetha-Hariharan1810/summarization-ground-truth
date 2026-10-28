from gtgen.checks import (
    check_evidence_and_rules,
    check_schema,
    check_summary_text,
    failures,
    id_in_turns,
    quote_match,
    spoken_id_stream,
)


def fail_messages(problems):
    return [p.message for p in failures(problems)]


def test_spoken_id_stream_reads_spelled_serials():
    assert spoken_id_stream("n l d one zero seven nine one seven n as in nancy") == "NLD107917N"
    assert spoken_id_stream("nine seven seven four five") == "97745"
    assert spoken_id_stream("m nine nine five four zero two a as in apple") == "M995402A"
    assert spoken_id_stream("double eight four") == "884"


def test_id_in_turns_ignores_articles_between_parts():
    assert id_in_turns("NLF137598N", "n l f one three and a seven five nine eight n as in nancy")
    assert not id_in_turns("NLD107917N", "n l d one zero seven nine")  # partly read: never completed
    # A misheard letter ("and" for "n") is not guessed: the serial must be left out or reviewed.
    assert not id_in_turns("NLF137598N", "and l f one three seven five nine eight n")


def test_quote_match():
    assert quote_match("battery pack", "let's take out the battery pack") == 1.0
    assert quote_match("the cord was cut by the dog", "let's take out the battery pack") < 0.9


def test_examples_pass(replacement_example, transfer_example):
    for call, record in (replacement_example, transfer_example):
        assert check_schema(record) == []
        assert fail_messages(check_evidence_and_rules(record, call)) == []


def test_invented_quote_fails(replacement_example):
    call, record = replacement_example
    record["issues"][0]["onset"]["quote"] = "it started after the patient dropped it"
    assert any("quote not found" in m for m in fail_messages(check_evidence_and_rules(record, call)))


def test_fact_without_quote_fails(replacement_example):
    call, record = replacement_example
    record["issues"][0]["onset"]["quote"] = None
    assert any("no quote" in m for m in fail_messages(check_evidence_and_rules(record, call)))


def test_action_must_be_the_agents_words(replacement_example):
    call, record = replacement_example
    record["issues"][0]["actions"][0].update(turns=["T8"], quote="yes it is flashing")
    assert any("agent's own turns" in m for m in fail_messages(check_evidence_and_rules(record, call)))


def test_redirect_needs_doctor_words(replacement_example):
    call, record = replacement_example
    record["issues"][0]["actions"].append(
        {"type": "redirected_hcp", "detail": None, "turns": ["T9"], "quote": "i'll submit a replacement request"}
    )
    assert any("does not state this action" in m for m in fail_messages(check_evidence_and_rules(record, call)))


def test_guessed_serial_fails(replacement_example):
    call, record = replacement_example
    record["issues"][0]["device_ids"][0]["serial"] = "NLD107918N"
    assert any("not read" in m for m in fail_messages(check_evidence_and_rules(record, call)))


def test_transfer_needs_transfer_words(transfer_example):
    call, record = transfer_example
    record["status_evidence"] = {"turns": ["T3"], "quote": "this is the neuromodulation line"}
    assert any("no transfer phrase" in m for m in fail_messages(check_evidence_and_rules(record, call)))


def test_missed_transfer_is_flagged_for_review(transfer_example, replacement_example):
    call, _ = transfer_example
    _, record = replacement_example  # a complete call record on a transcript with a transfer
    record["issues"][0].update(device_ids=[], troubleshooting=[], actions=[])
    for k in ("reason", "onset"):
        record["issues"][0][k] = {"text": None, "turns": [], "quote": None}
    problems = check_evidence_and_rules(record, call)
    assert any(p.severity == "review" and "transfer" in p.message for p in problems)


def test_summary_text_checks():
    assert fail_messages(check_summary_text("Patient said their managing healthcare plan was Dr. Cooper."))
    assert fail_messages(check_summary_text("The patient gave their number 555 123 4567."))
    assert fail_messages(check_summary_text("Patient lives at 1702 East Parkway Drive."))
    assert not fail_messages(check_summary_text("The controller was at 80% and the implant at 40%."))


def test_schema_rejects_unknown_values(replacement_example):
    _, record = replacement_example
    record["issues"][0]["outcome"] = "probably_fixed"
    assert check_schema(record)
