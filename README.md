# summarization-ground-truth

Builds training labels for the call summarization model with as little hallucination as possible. A GPT model never writes the summary directly. It fills a record in which every fact carries the transcript turn and a verbatim quote it came from. Code checks every quote, a program writes the summary from the record using fixed sentences, and a second model checks each sentence against the transcript. Labels that fail are regenerated once, then rejected or sent to a person.

Design docs:

- [Ground truth generation design](docs/ground-truth-generation-design.md) ([live doc](https://claude.ai/artifact/5768iTMWsF25Z7DFVxkFFe))
- [Model selection and training plan](docs/model-selection-and-training-plan.md) ([live doc](https://claude.ai/artifact/VrSjCNc1Whhonu5ggsAAYm)): the model that will train on these labels

```
prepare ──> label ──> report
              │
              ├─> export         train / validation / test CSVs + manifest
              └─> review-sample  stratified CSV for SME review
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then fill in the key and model names
```

`.env` holds the GPT key and model names; it is git-ignored. The main settings:

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER` | `openai` or `azure` |
| `OPENAI_API_KEY` | Key for `openai` (with optional `OPENAI_BASE_URL`) |
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | For `azure`; model names are then deployment names |
| `LABELLER_MODEL` | Writes the record. Use the strongest model you have |
| `JUDGE_MODEL` | Checks the summary. Must differ from the labeller |
| `CLOSED_SET_MODEL` | Second, cheaper pass over caller, outcome, actions and transfer |
| `LABELLER_REASONING_EFFORT`, `JUDGE_REASONING_EFFORT` | Optional, for reasoning models |
| `SUMM_SERVICE_SRC` | The summarization service's `src/` directory |
| `MAX_CONCURRENCY`, `MAX_LABEL_ATTEMPTS` | Parallel calls (default 4); label attempts per call (default 2) |

The models must support structured outputs (`response_format` with a strict JSON schema).

**Patient data.** Use a provider and account covered by your BAA (for example Azure OpenAI under your Microsoft agreement) and get compliance sign-off before sending transcripts. Call data, prepared files and labels are git-ignored; keep them out of the repo.

## Run

```bash
# 1. Clean transcripts exactly as the service does, keep the date window, assign splits.
gtgen prepare --input calls.csv --out runs/v1/prepared.jsonl \
  --start 2024-07-01 --end 2026-09-30 --test-start 2026-07-01 \
  --hint-column updated_summary          # optional: keep the agent note for --use-hint

# 2. Pilot: a random sample first, then read the report and review the sample.
gtgen label --prepared runs/v1/prepared.jsonl --out runs/v1/pilot.jsonl --split train --sample 200
gtgen report --labels runs/v1/pilot.jsonl
gtgen review-sample --prepared runs/v1/prepared.jsonl --labels runs/v1/pilot.jsonl \
  --out runs/v1/pilot_review.csv --per-type 30

# 3. Full run. Resumable: rerun the same command after an interruption.
gtgen label --prepared runs/v1/prepared.jsonl --out runs/v1/labels.jsonl

# 4. Training files (accepted labels only, unless --include-review).
gtgen export --labels runs/v1/labels.jsonl --out-dir runs/v1/dataset
```

The input CSV needs `call_id`, `date` and `transcript` columns. Other names: `--id-column`, `--date-column`, `--transcript-column`. `--group-column` keeps rows that share a value (for example a patient id) in the same split.

## What happens to each call

1. **Prepare.** `aggregate_by_channel` and `clean_transcript` are imported from the service, so the label is made from the same text the model is trained and served on. Turns are numbered `[T1]`, `[T2]` … for citation. Calls under 26 words (the service answers "Empty Call") and blank transcripts are skipped.
2. **Label.** The labeller fills the record (`src/gtgen/schema.py`) using the spec in `src/gtgen/prompts/labeller_system.md` and the worked examples in `examples/examples.jsonl`.
3. **Checks 1 to 3, in code** (`src/gtgen/checks.py`):
   - the record matches the schema;
   - every quote matches its cited turns (at least 90% of tokens, in order);
   - every serial and model number can be rebuilt from the spelled-out letters and digits in its turns;
   - actions and transfers quote the agent's own turns, using words that state the action;
   - the summary holds no phone number, date of birth, street address or "healthcare plan".
4. **Render** (`src/gtgen/render.py`). Fixed sentences for caller, device, outcome, actions, transfers and incomplete calls; the cited free text fills the rest.
5. **Judge.** A different model marks each sentence supported, unsupported or contradicted, and lists missing key facts.
6. **Closed-set pass.** A second model classifies caller, outcomes, actions and transfer from the transcript alone. Any disagreement with the label goes to review.

Any failure triggers one regeneration, with the problems quoted back to the labeller.

| Status | Meaning | Exported |
| --- | --- | --- |
| `accepted` | Passed every check (may be flagged `incomplete_coverage`) | Yes |
| `review` | A review-level problem, a closed-set disagreement, or a failed label on a rare call type (transfer, incomplete, multi-issue) | With `--include-review` |
| `rejected` | Still failing after the allowed attempts | No |
| `skipped` | Blank or under the minimum length | No |

`report` shows the reject rate per call type. Above 15% for a type, the training set is losing exactly those calls: fix the prompt or review that type by hand rather than dropping it.

Each output row also records the prompt version (a hash of the prompts and examples), the models used and the token usage.

## Before the pilot

- **Replace the worked examples.** `examples/examples.jsonl` holds two synthetic calls that show the format. Add about 8 SME-approved examples from real calls: normal, two issues, transfer out before the reason, transfer out after partial information, transfer in, incomplete, mostly garbled, and a call where the agent did *not* redirect to the healthcare provider. Every example must pass the checks (`pytest` loads them).
- **Calibrate the judge** against SME sentence marks on a frozen set (design, section 6) before trusting its verdicts.

## Known limits

- **Speech-recognition errors in serials are not repaired.** If ASR wrote "and l f" for "n l f", the serial check fails rather than guess. The labeller is told to leave such IDs out.
- **No batch API.** Calls are made synchronously with `MAX_CONCURRENCY` in parallel. A provider batch API would cut the cost roughly in half.

## Tests

```bash
pytest
# the preprocessing tests need the service source:
GTGEN_TEST_SERVICE_SRC=../summarization-service/src pytest
```

The tests use a fake model client and need no key.
