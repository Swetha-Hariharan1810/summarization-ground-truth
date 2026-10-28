"""Command line: prepare -> label -> report -> export / review-sample."""

import argparse
import hashlib
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .config import load_settings
from .preprocess import PreparedCall, ServicePreprocessing, Turn


def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _split_for(date: pd.Timestamp, group_key: str, test_start: pd.Timestamp, val_share: float) -> str:
    if date >= test_start:
        return "test"
    bucket = int(hashlib.sha256(group_key.encode()).hexdigest(), 16) % 10_000
    return "validation" if bucket < val_share * 10_000 else "train"


def cmd_prepare(args) -> None:
    settings = load_settings(args.env)
    pre = ServicePreprocessing(settings.summ_service_src)
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    for col in (args.id_column, args.date_column, args.transcript_column):
        if col not in df.columns:
            raise SystemExit(f"column {col!r} not in {args.input}; found {list(df.columns)}")

    dates = pd.to_datetime(df[args.date_column], utc=True, errors="coerce")
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    test_start = pd.Timestamp(args.test_start, tz="UTC")
    in_window = dates.notna() & (dates >= start) & (dates < end)
    df, dates = df[in_window], dates[in_window]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen = set()
    with out.open("w") as f:
        for (_, row), date in zip(df.iterrows(), dates):
            call_id = str(row[args.id_column])
            if call_id in seen:
                counts["duplicate_id"] += 1
                continue
            seen.add(call_id)
            call = pre.prepare(call_id, row[args.transcript_column])
            group = str(row[args.group_column]) if args.group_column else call_id
            split = _split_for(date, group, test_start, args.val_share)
            counts[call.skip_reason or split] += 1
            f.write(
                json.dumps(
                    {
                        "call_id": call_id,
                        "date": date.isoformat(),
                        "split": split,
                        "skip_reason": call.skip_reason,
                        "serving_text": call.serving_text,
                        "turns": [t.__dict__ for t in call.turns],
                        "hint": row[args.hint_column] if args.hint_column else None,
                    }
                )
                + "\n"
            )
    print(f"wrote {out}: {dict(counts)}")


def _to_call(row: dict) -> PreparedCall:
    return PreparedCall(
        row["call_id"], row["serving_text"], [Turn(**t) for t in row["turns"]], row["skip_reason"]
    )


def cmd_label(args) -> None:
    from .llm import LLMClient
    from .pipeline import Pipeline, load_examples

    settings = load_settings(args.env)
    settings.validate_for_labelling()
    rows = _read_jsonl(args.prepared)
    if args.split:
        rows = [r for r in rows if r["split"] in args.split]
    if args.sample:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["call_id"] for r in _read_jsonl(out)}
    todo = [r for r in rows if r["call_id"] not in done]
    print(f"{len(rows)} calls selected, {len(done)} already labelled, {len(todo)} to do")
    if not todo:
        return

    pipeline = Pipeline(settings, LLMClient(settings), load_examples(args.examples))
    print(f"prompt version {pipeline.prompt_version}")
    lock = threading.Lock()
    status = Counter()

    def work(row: dict) -> dict:
        result = pipeline.run(_to_call(row), hint=row.get("hint") if args.use_hint else None)
        result.update(date=row["date"], split=row["split"], serving_text=row["serving_text"])
        return result

    with out.open("a") as f, ThreadPoolExecutor(max_workers=settings.max_concurrency) as pool:
        futures = {pool.submit(work, row): row["call_id"] for row in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            call_id = futures[future]
            try:
                result = future.result()
            except Exception as e:  # keep the run going; the call is retried on the next run
                print(f"[{n}/{len(todo)}] {call_id}: error {type(e).__name__}: {e}", file=sys.stderr)
                continue
            with lock:
                f.write(json.dumps(result) + "\n")
                f.flush()
            status[result["status"]] += 1
            print(f"[{n}/{len(todo)}] {call_id}: {result['status']} ({result['call_type']})")
    print(dict(status))


def cmd_report(args) -> None:
    rows = _read_jsonl(args.labels)
    if not rows:
        raise SystemExit(f"no rows in {args.labels}")
    print(f"{len(rows)} calls\n")

    print("Status:", dict(Counter(r["status"] for r in rows)))
    by_type = defaultdict(Counter)
    for r in rows:
        if r["status"] != "skipped":
            by_type[r["call_type"]][r["status"]] += 1
    print("\nBy call type (reject rate over 15% means the training set is losing that type):")
    print(f"  {'call type':28} {'total':>6} {'accepted':>9} {'review':>7} {'rejected':>9} {'reject %':>9}")
    for t, c in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(c.values())
        warn = "  <-- over 15%" if c["rejected"] / total > 0.15 else ""
        print(
            f"  {t:28} {total:6} {c['accepted']:9} {c['review']:7} {c['rejected']:9} "
            f"{100 * c['rejected'] / total:8.1f}%{warn}"
        )

    print("\nProblems by check and severity:")
    problems = Counter((p["check"], p["severity"]) for r in rows for p in r["problems"])
    for (check, severity), n in problems.most_common():
        print(f"  {check:12} {severity:7} {n}")

    print("\nMost common problem messages:")
    messages = Counter(
        p["message"].split(":")[-1].strip()[:90] for r in rows for p in r["problems"] if p["severity"] == "fail"
    )
    for message, n in messages.most_common(args.top):
        print(f"  {n:5}  {message}")

    flags = Counter(f for r in rows for f in r["flags"])
    print("\nFlags:", dict(flags))

    usage = defaultdict(Counter)
    for r in rows:
        for model, u in r.get("usage", {}).get("by_model", {}).items():
            usage[model].update(u)
    print("\nToken usage by model:")
    for model, u in usage.items():
        print(f"  {model}: {u['calls']} calls, {u['input_tokens']:,} in, {u['output_tokens']:,} out")


def cmd_export(args) -> None:
    rows = _read_jsonl(args.labels)
    keep = {"accepted"} | ({"review"} if args.include_review else set())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        chosen = [r for r in rows if r["split"] == split and r["status"] in keep]
        df = pd.DataFrame(
            {
                "call_id": [r["call_id"] for r in chosen],
                "date": [r["date"] for r in chosen],
                "call_type": [r["call_type"] for r in chosen],
                "transcript": [r["serving_text"] for r in chosen],
                "summary": [r["summary"] for r in chosen],
            }
        )
        path = out / f"{split}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path}: {len(df)} rows")
    manifest = {
        "labels": str(args.labels),
        "statuses_included": sorted(keep),
        "prompt_versions": sorted({r["prompt_version"] for r in rows}),
        "models": sorted({json.dumps(r["models"], sort_keys=True) for r in rows}),
        "counts": {
            split: dict(Counter(r["status"] for r in rows if r["split"] == split))
            for split in ("train", "validation", "test")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out / 'manifest.json'}")


REVIEW_COLUMNS = [
    "unsupported_sentences",
    "contradicted_sentences",
    "missing_key_fact",
    "caller_correct",
    "outcome_correct",
    "actions_correct",
    "transfer_correct",
    "device_ids_correct",
    "overall",
    "corrected_summary",
    "reviewer",
]


def cmd_review_sample(args) -> None:
    prepared = {r["call_id"]: r for r in _read_jsonl(args.prepared)}
    rows = [r for r in _read_jsonl(args.labels) if r["status"] != "skipped"]
    if args.split:
        rows = [r for r in rows if r["split"] in args.split]
    rng = random.Random(args.seed)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["call_type"]].append(r)
    sample = []
    for t, items in sorted(by_type.items()):
        rng.shuffle(items)
        sample += items[: args.per_type]
        print(f"  {t}: {min(len(items), args.per_type)} of {len(items)}")

    def numbered(call_id: str) -> str:
        p = prepared.get(call_id)
        return _to_call(p).numbered_transcript() if p else ""

    df = pd.DataFrame(
        {
            "call_id": [r["call_id"] for r in sample],
            "call_type": [r["call_type"] for r in sample],
            "pipeline_status": [r["status"] for r in sample],
            # Transcript first: reviewers read it before the summary.
            "transcript": [numbered(r["call_id"]) for r in sample],
            "summary": [r["summary"] for r in sample],
            "pipeline_problems": ["\n".join(p["message"] for p in r["problems"]) for r in sample],
            **{c: [""] * len(sample) for c in REVIEW_COLUMNS},
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} calls")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gtgen", description=__doc__)
    parser.add_argument("--env", help="path to a .env file (default: ./.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="clean transcripts with the service preprocessing and assign splits")
    p.add_argument("--input", required=True, help="CSV export of calls")
    p.add_argument("--out", required=True, help="prepared JSONL")
    p.add_argument("--start", default="2024-07-01")
    p.add_argument("--end", default="2026-09-30", help="last day included")
    p.add_argument("--test-start", default="2026-07-01")
    p.add_argument("--val-share", type=float, default=0.1)
    p.add_argument("--id-column", default="call_id")
    p.add_argument("--date-column", default="date")
    p.add_argument("--transcript-column", default="transcript")
    p.add_argument("--group-column", help="keep rows sharing this value in one split (e.g. a patient id)")
    p.add_argument("--hint-column", help="agent note column, stored for --use-hint")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("label", help="run the labelling pipeline (resumable)")
    p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True, help="labels JSONL; appended to, already-labelled calls skipped")
    p.add_argument("--examples", default="examples/examples.jsonl")
    p.add_argument("--split", nargs="*", choices=["train", "validation", "test"])
    p.add_argument("--sample", type=int, help="label a random sample of this many calls")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--use-hint", action="store_true", help="show the agent note to the labeller as a hint")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("report", help="status, reject rate by call type, problems, token usage")
    p.add_argument("--labels", required=True)
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", help="write train/validation/test CSVs for training")
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--include-review", action="store_true", help="also export calls marked for review")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("review-sample", help="stratified sample CSV for SME review")
    p.add_argument("--prepared", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-type", type=int, default=30)
    p.add_argument("--split", nargs="*", choices=["train", "validation", "test"])
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=cmd_review_sample)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
