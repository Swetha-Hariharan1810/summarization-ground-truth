import os
from pathlib import Path

import pytest

from gtgen.preprocess import ServicePreprocessing

SERVICE_SRC = os.getenv("GTGEN_TEST_SERVICE_SRC", "")

pytestmark = pytest.mark.skipif(
    not (SERVICE_SRC and (Path(SERVICE_SRC) / "preprocess" / "process_transcripts.py").exists()),
    reason="set GTGEN_TEST_SERVICE_SRC to the summarization service's src/ to run",
)

RAW = """Agent:  neuromodulation this is sam can i get your first and last name and a phone number please
Caller:  sure it's [NAME] [PHONE]
Agent:  thank you and what is the reason for your call today
Caller:  my controller says no device found when i try to charge
Caller:  it started two days ago
Agent:  okay let's take out the battery pack and read me the serial number that starts with n l d
Caller:  n l d one zero seven nine one seven n as in nancy
Agent:  great now plug the controller into the wall do you see a green light flashing
Caller:  yes it is flashing
Agent:  okay it sounds like the recharger cord needs replacing i'll submit a replacement request and send it overnight"""


def test_turns_match_serving_text():
    call = ServicePreprocessing(SERVICE_SRC).prepare("x", RAW)
    assert call.skip_reason is None
    # Consecutive turns by the same speaker are merged, as the service does.
    assert [t.speaker for t in call.turns[:4]] == ["Agent", "Caller", "Agent", "Caller"]
    assert call.turns[3].text.endswith("it started two days ago")
    joined = " ".join(f"{t.speaker}: {t.text}" for t in call.turns)
    assert joined == call.serving_text


def test_short_call_is_skipped():
    call = ServicePreprocessing(SERVICE_SRC).prepare("x", "Agent: hello\nCaller: bye")
    assert call.skip_reason == "under_min_words"
