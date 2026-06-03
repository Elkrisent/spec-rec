"""T8.3 / T8.5 — Evaluation harness.

Runs the full pipeline on the synthetic dataset and holdout set,
computes precision/recall per field, prints the evaluation report.

Usage (requires Ollama running):
    python -m claim_verifier.eval.run_eval
    python -m claim_verifier.eval.run_eval --out eval/report.md
    python -m claim_verifier.eval.run_eval --dataset data_gen/output --holdout claim_verifier/eval/holdout

Subsequent runs use the LLM response cache — no Ollama needed once cached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from claim_verifier.eval.metrics import (
    EvalSummary,
    compute_eval_summary,
    format_report,
)

_FIELDS = [
    "hospital_name",
    "admission_date",
    "discharge_date",
    "diagnosis",
    "billed_amount",
    "length_of_stay",
]

_DEFAULT_DATASET = Path("data_gen/output")
_DEFAULT_HOLDOUT = Path("claim_verifier/eval/holdout")


def _run_case(
    case_id: str,
    transcript_path: Path,
    bill_path: Path,
    backend,
    judge,
) -> dict[str, str]:
    """Run one case through the pipeline. Returns {field: status} dict."""
    from claim_verifier.pipeline import run as pipeline_run

    result = pipeline_run(case_id, transcript_path, bill_path, backend, judge)
    if result.verification_result is None:
        return {f: "MISSING" for f in _FIELDS}
    fv = result.verification_result.field_verdicts
    return {f: fv[f].status for f in _FIELDS if f in fv}


def eval_dataset(
    dataset_dir: Path,
    backend,
    judge,
    verbose: bool = True,
) -> tuple[list[dict], list[dict[str, str]]]:
    """Run the pipeline on every case in a dataset directory.

    Returns (records, verdicts). records come from ground_truth.jsonl;
    verdicts are the per-field statuses the pipeline produced.
    """
    gt_path = dataset_dir / "ground_truth.jsonl"
    records: list[dict] = []
    with gt_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    verdicts: list[dict[str, str]] = []
    for i, record in enumerate(records, start=1):
        case_id = record["case_id"]
        if verbose:
            print(
                f"  [{i}/{len(records)}] {case_id} ({record['error_type']})...",
                end="",
                flush=True,
            )
        try:
            v = _run_case(
                case_id,
                dataset_dir / record["transcript_txt"],
                dataset_dir / record["bill_pdf"],
                backend,
                judge,
            )
            verdicts.append(v)
            if verbose:
                print(" done")
        except Exception as exc:
            verdicts.append({f: "MISSING" for f in _FIELDS})
            if verbose:
                print(f" ERROR: {exc}")

    return records, verdicts


def eval_holdout(
    holdout_dir: Path,
    backend,
    judge,
    verbose: bool = True,
) -> tuple[list[dict], list[dict[str, str]]]:
    """Run the pipeline on every case in the holdout directory.

    Each case is a subdirectory containing transcript.txt, bill.pdf,
    and ground_truth.json.
    """
    records: list[dict] = []
    verdicts: list[dict[str, str]] = []

    case_dirs = sorted(d for d in holdout_dir.iterdir() if d.is_dir())
    for case_dir in case_dirs:
        gt_path = case_dir / "ground_truth.json"
        if not gt_path.exists():
            continue
        record = json.loads(gt_path.read_text(encoding="utf-8"))
        records.append(record)

        if verbose:
            print(
                f"  {record['case_id']} ({record['error_type']})...",
                end="",
                flush=True,
            )
        try:
            v = _run_case(
                record["case_id"],
                case_dir / "transcript.txt",
                case_dir / "bill.pdf",
                backend,
                judge,
            )
            verdicts.append(v)
            if verbose:
                print(" done")
        except Exception as exc:
            verdicts.append({f: "MISSING" for f in _FIELDS})
            if verbose:
                print(f" ERROR: {exc}")

    return records, verdicts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation harness on synthetic dataset and holdout."
    )
    parser.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        help=f"Path to dataset directory (default: {_DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--holdout",
        default=str(_DEFAULT_HOLDOUT),
        help=f"Path to holdout directory (default: {_DEFAULT_HOLDOUT})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to save the report (optional; prints to stdout if omitted)",
    )
    args = parser.parse_args()

    from claim_verifier.backends.ollama import OllamaBackend
    from claim_verifier.judge import LLMJudge

    backend = OllamaBackend()
    judge = LLMJudge(backend)

    dataset_dir = Path(args.dataset)
    holdout_dir = Path(args.holdout)

    print(f"Evaluating synthetic dataset ({dataset_dir})...")
    syn_records, syn_verdicts = eval_dataset(dataset_dir, backend, judge)
    syn_summary = compute_eval_summary(syn_records, syn_verdicts)

    holdout_summary: EvalSummary | None = None
    if holdout_dir.exists():
        print(f"\nEvaluating holdout ({holdout_dir})...")
        hold_records, hold_verdicts = eval_holdout(holdout_dir, backend, judge)
        if hold_records:
            holdout_summary = compute_eval_summary(hold_records, hold_verdicts)

    report = format_report(syn_summary, holdout_summary)
    print("\n" + report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
