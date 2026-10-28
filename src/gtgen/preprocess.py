"""Turn a raw transcript into exactly what the summarization service feeds its model.

The cleaning functions are imported from the service (SUMM_SERVICE_SRC), not copied,
so a label is always made from the same text the model will be trained and served on.
The steps below mirror `get_summary()` in the service's `src/summ_main.py`.
"""

import importlib
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Same threshold as MIN_WORDS in the service: shorter calls are answered with
# "Empty Call" by the service and never reach the model.
MIN_WORDS = 26


@dataclass
class Turn:
    id: str
    speaker: str
    text: str


@dataclass
class PreparedCall:
    call_id: str
    serving_text: str
    turns: list[Turn] = field(default_factory=list)
    skip_reason: str | None = None

    def numbered_transcript(self) -> str:
        return "\n".join(f"[{t.id}] {t.speaker}: {t.text}" for t in self.turns)

    def turn_map(self) -> dict[str, Turn]:
        return {t.id: t for t in self.turns}

    @classmethod
    def from_numbered(cls, call_id: str, numbered: str) -> "PreparedCall":
        """Rebuild a call from `numbered_transcript()` output (examples, tests)."""
        turns = []
        for line in numbered.strip().split("\n"):
            tid, _, rest = line.partition("] ")
            speaker, _, text = rest.partition(": ")
            turns.append(Turn(id=tid.lstrip("["), speaker=speaker, text=text))
        return cls(call_id, " ".join(f"{t.speaker}: {t.text}" for t in turns), turns)


class ServicePreprocessing:
    """Thin wrapper around the service's preprocess.process_transcripts module."""

    def __init__(self, service_src: str):
        if not service_src:
            raise SystemExit("SUMM_SERVICE_SRC is not set; point it at the service's src/ directory")
        src = Path(service_src).expanduser().resolve()
        if not (src / "preprocess" / "process_transcripts.py").exists():
            raise SystemExit(f"SUMM_SERVICE_SRC={src} does not contain preprocess/process_transcripts.py")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        with warnings.catch_warnings():
            # The service module has an invalid escape in a string literal; harmless.
            warnings.simplefilter("ignore", SyntaxWarning)
            module = importlib.import_module("preprocess.process_transcripts")
        self.aggregate_by_channel = module.aggregate_by_channel
        self.clean_transcript = module.clean_transcript

    def _second_pass(self, text: str) -> str:
        # get_summary() runs clean_transcript a second time with these flags
        # (word_to_digit is off in the service config).
        return self.clean_transcript(
            [text], phrases_removal=False, words_removal=False, after_number=True
        )[0]

    def prepare(self, call_id: str, raw_transcript: str) -> PreparedCall:
        raw_transcript = raw_transcript or ""
        aggregated = self.aggregate_by_channel(raw_transcript)
        lines = [line for line in aggregated.split("\n") if line]
        serving_text = self._second_pass(" ".join(lines))

        if not raw_transcript.strip():
            return PreparedCall(call_id, serving_text, skip_reason="empty_transcript")
        if len(serving_text.split()) < MIN_WORDS:
            return PreparedCall(call_id, serving_text, skip_reason="under_min_words")

        turns = []
        for line in lines:
            speaker, sep, body = line.partition(": ")
            if not sep:
                speaker, body = "Unknown", line
            body = self._second_pass(body)
            if body:
                turns.append(Turn(id=f"T{len(turns) + 1}", speaker=speaker, text=body))
        return PreparedCall(call_id, serving_text, turns)
