import copy
import json
from pathlib import Path

import pytest

from gtgen.preprocess import PreparedCall

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def examples():
    lines = (ROOT / "examples" / "examples.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture
def replacement_example(examples):
    ex = examples[0]
    return PreparedCall.from_numbered("ex1", ex["transcript"]), copy.deepcopy(ex["record"])


@pytest.fixture
def transfer_example(examples):
    ex = examples[1]
    return PreparedCall.from_numbered("ex2", ex["transcript"]), copy.deepcopy(ex["record"])
