"""Score two or more model checkpoints on GSM8K + BBH and diff the results.

The point of this script is the before/after comparison: run the same prompts,
the same decoding settings and the same scoring code against a pre-finetune and
a post-finetune checkpoint, then report where the two differ.

An 8B model generating ~512 tokens per example takes several seconds per
example, and a run is thousands of examples, so the work is spread over every
available GPU. Each worker is a separate process pinned to one GPU that loads
one model and then pulls jobs (a BBH subtask, or a slice of GSM8K) from a
shared pool. Jobs are claimed by atomically creating a .claim file, so workers
sharing a model self-balance -- BBH subtasks differ by several times in how
long their answers run, which a static split would handle badly. Finished jobs
leave a .json behind and are skipped on a re-run, so an interrupted comparison
resumes instead of starting over.

    python compare_models.py --tag stage1 --bbh-limit 50

    # then, any time later, just the table again:
    python compare_models.py --tag stage1 --bbh-limit 50 --report-only
"""

import argparse
import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from common import DEFAULT_EVAL_DIR, RESULTS_DIR, load_model, write_jsonl
from run_bbh import discover_tasks, evaluate_bbh
from run_gsm8k import evaluate_gsm8k

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODELS = [f"before={MODELS_DIR / 'CPM8B'}", f"after={MODELS_DIR / 'stage1_dense'}"]
# GSM8K is one flat 1319-example file; splitting it into this many jobs gives
# the scheduler pieces small enough to balance against the BBH subtasks.
GSM8K_CHUNKS = 8


def parse_models(specs):
    models = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--model expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        models[name] = path
    return models


def build_jobs(model_name, tasks, bbh_limit, gsm8k_limit):
    """One job per BBH subtask plus GSM8K_CHUNKS strided slices of GSM8K."""
    jobs = [
        {"id": f"{model_name}__bbh__{task}", "model": model_name, "type": "bbh",
         "task": task, "limit": bbh_limit}
        for task in tasks
    ]
    jobs += [
        {"id": f"{model_name}__gsm8k__{i}of{GSM8K_CHUNKS}", "model": model_name, "type": "gsm8k",
         "shard": [i, GSM8K_CHUNKS], "limit": gsm8k_limit}
        for i in range(GSM8K_CHUNKS)
    ]
    return jobs


def claim(path: Path) -> bool:
    """Atomically claim a job; False if another worker got there first."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return False
        raise
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True


def run_worker(args):
    """Worker mode: load one model, then drain the shared job pool."""
    work_dir = Path(args.work_dir)
    jobs = [j for j in json.loads(Path(args.jobs_file).read_text()) if j["model"] == args.model_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[{args.worker_id}] loading {args.model_dir} on {device}", flush=True)
    tokenizer, model = load_model(args.model_dir, device, dtype)

    for job in jobs:
        out_path = work_dir / f"{job['id']}.json"
        if out_path.exists():
            continue
        if not claim(work_dir / f"{job['id']}.claim"):
            continue

        start = time.time()
        print(f"[{args.worker_id}] start {job['id']}", flush=True)
        if job["type"] == "bbh":
            predictions, summary = evaluate_bbh(
                tokenizer, model, device, eval_dir=args.eval_dir,
                tasks=[job["task"]], limit=job["limit"],
                max_new_tokens=args.max_new_tokens,
            )
        else:
            predictions, summary = evaluate_gsm8k(
                tokenizer, model, device, eval_dir=args.eval_dir,
                limit=job["limit"], max_new_tokens=args.max_new_tokens,
                shard_index=job["shard"][0], num_shards=job["shard"][1],
            )

        write_jsonl(work_dir / f"{job['id']}_predictions.jsonl", predictions)
        # Written last: the .json is what marks the job done, so it must not
        # appear before the predictions it summarizes.
        out_path.write_text(json.dumps({"job": job, "summary": summary}, ensure_ascii=False, indent=2))
        print(f"[{args.worker_id}] done {job['id']} in {time.time() - start:.0f}s", flush=True)

    print(f"[{args.worker_id}] no jobs left", flush=True)


def response_length(path: Path):
    """(mean chars, count) over a predictions file, for the chain-of-thought
    length check -- a fine-tune that stops reasoning step by step shows up here
    as a collapse in response length, which is easy to mistake for the answer
    extractor breaking."""
    if not path.exists():
        return 0, 0
    lengths = [len(json.loads(line)["response"]) for line in path.open(encoding="utf-8") if line.strip()]
    return sum(lengths), len(lengths)


def collect(work_dir: Path, models, tasks):
    """Merge finished job files into one summary per model."""
    results = {}
    for name in models:
        gsm_correct = gsm_total = 0
        per_task = {}
        missing = []
        chars = chars_n = 0
        for task in tasks:
            path = work_dir / f"{name}__bbh__{task}.json"
            if not path.exists():
                missing.append(f"bbh:{task}")
                continue
            summary = json.loads(path.read_text())["summary"]
            per_task[task] = {
                "accuracy": summary["per_task_accuracy"][task],
                "num_examples": summary["num_examples"],
            }
            c, n = response_length(work_dir / f"{name}__bbh__{task}_predictions.jsonl")
            per_task[task]["mean_response_chars"] = c / n if n else 0.0
            chars += c
            chars_n += n
        for i in range(GSM8K_CHUNKS):
            path = work_dir / f"{name}__gsm8k__{i}of{GSM8K_CHUNKS}.json"
            if not path.exists():
                missing.append(f"gsm8k:{i}")
                continue
            summary = json.loads(path.read_text())["summary"]
            gsm_correct += summary["correct"]
            gsm_total += summary["num_examples"]
            c, n = response_length(work_dir / f"{name}__gsm8k__{i}of{GSM8K_CHUNKS}_predictions.jsonl")
            chars += c
            chars_n += n

        macro = sum(t["accuracy"] for t in per_task.values()) / len(per_task) if per_task else 0.0
        results[name] = {
            "gsm8k": {"correct": gsm_correct, "num_examples": gsm_total,
                      "accuracy": gsm_correct / gsm_total if gsm_total else 0.0},
            "bbh": {"per_task": per_task,
                    "num_examples": sum(t["num_examples"] for t in per_task.values()),
                    "macro_avg_accuracy": macro},
            "mean_response_chars": chars / chars_n if chars_n else 0.0,
            "missing_jobs": missing,
        }
    return results


def format_report(results, models, tasks):
    names = list(models)
    lines = []
    width = max(len(t) for t in tasks) + 2
    # While a run is still in flight the models have finished different subsets
    # of BBH, and a macro-average over different task sets is not a comparison
    # at all -- one model can look better purely by having finished the easy
    # subtasks. Every cross-model number below is restricted to the subtasks
    # that all models have finished.
    common = [t for t in tasks if all(t in results[n]["bbh"]["per_task"] for n in names)]

    def row(label, values, delta=None):
        cells = "".join(f"{v:>12}" for v in values)
        return f"  {label:<{width}}{cells}" + (f"{delta:>12}" if delta is not None else "")

    def pct(x):
        return f"{100 * x:.2f}%"

    lines.append("=" * (width + 12 * (len(names) + 1) + 2))
    lines.append(row("", names, "delta" if len(names) == 2 else None))
    lines.append("-" * (width + 12 * (len(names) + 1) + 2))

    gsm = [results[n]["gsm8k"] for n in names]
    # A model with no finished GSM8K job has accuracy 0.0, which is not a score
    # -- showing it as one would invent a delta against nothing.
    gsm_ready = all(g["num_examples"] for g in gsm)
    lines.append(row(
        "GSM8K (n=" + ("/".join(str(g["num_examples"]) for g in gsm)) + ")",
        [pct(g["accuracy"]) if g["num_examples"] else "--" for g in gsm],
        (f"{100 * (gsm[1]['accuracy'] - gsm[0]['accuracy']):+.2f}" if gsm_ready else "n/a")
        if len(names) == 2 else None,
    ))
    macros = [
        sum(results[n]["bbh"]["per_task"][t]["accuracy"] for t in common) / len(common) if common else 0.0
        for n in names
    ]
    bbh_n = sum(results[names[0]]["bbh"]["per_task"][t]["num_examples"] for t in common)
    lines.append(row(
        f"BBH macro ({len(common)} tasks, n={bbh_n})",
        [pct(m) for m in macros],
        f"{100 * (macros[1] - macros[0]):+.2f}" if len(names) == 2 else None,
    ))
    lengths = [
        sum(results[n]["bbh"]["per_task"][t]["mean_response_chars"] for t in common) / len(common)
        if common else 0.0
        for n in names
    ]
    lines.append(row(
        "BBH mean response chars",
        [f"{x:.0f}" for x in lengths],
        f"{lengths[1] - lengths[0]:+.0f}" if len(names) == 2 else None,
    ))
    lines.append("-" * (width + 12 * (len(names) + 1) + 2))
    lines.append("  BBH per subtask:")
    for task in common:
        accs = [results[n]["bbh"]["per_task"][task]["accuracy"] for n in names]
        delta = f"{100 * (accs[1] - accs[0]):+.2f}" if len(names) == 2 else None
        lines.append(row(task, [pct(a) for a in accs], delta))
    lines.append("=" * (width + 12 * (len(names) + 1) + 2))

    for name in names:
        missing = results[name]["missing_jobs"]
        if missing:
            lines.append(f"  WARNING: {name} is missing {len(missing)} unfinished job(s): {', '.join(missing[:6])}"
                         + (" ..." if len(missing) > 6 else ""))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", dest="models", default=None,
                        help="name=path, repeatable; default: before=models/CPM8B after=models/stage1_dense")
    parser.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--tag", type=str, default="compare", help="Names the results/<tag>/ output directory")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids, default: all visible")
    parser.add_argument("--bbh-limit", type=int, default=50, help="Examples per BBH subtask (0 = all 250)")
    parser.add_argument("--gsm8k-limit", type=int, default=None, help="Total GSM8K examples (default: all 1319)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--report-only", action="store_true", help="Only re-print the table from finished jobs")
    # Worker-mode plumbing; set by the parent process, not meant to be typed.
    parser.add_argument("--worker-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--jobs-file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_id:
        return run_worker(args)

    models = parse_models(args.models or DEFAULT_MODELS)
    for name, path in models.items():
        if not Path(path).is_dir():
            raise SystemExit(f"model {name!r}: {path} is not a directory")

    tasks = discover_tasks(Path(args.eval_dir) / "bbh_jsonl")
    work_dir = RESULTS_DIR / args.tag
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        gpus = args.gpus.split(",") if args.gpus else [str(i) for i in range(torch.cuda.device_count())]
        if not gpus:
            raise SystemExit("no GPUs available")

        jobs = []
        for name in models:
            jobs += build_jobs(name, tasks, args.bbh_limit or None, args.gsm8k_limit)
        jobs_file = work_dir / "jobs.json"
        jobs_file.write_text(json.dumps(jobs, indent=2))

        # A worker is bound to the model it loaded, so GPUs are dealt out to
        # models round-robin; workers on the same model then share its jobs.
        names = list(models)
        assignments = [(gpu, names[i % len(names)]) for i, gpu in enumerate(gpus)]
        print(f"Launching {len(assignments)} workers over {len(jobs)} jobs:")
        for gpu, name in assignments:
            print(f"  gpu{gpu} -> {name} ({models[name]})")

        procs = []
        for gpu, name in assignments:
            worker_id = f"gpu{gpu}:{name}"
            log_path = work_dir / f"worker_gpu{gpu}.log"
            cmd = [
                sys.executable, __file__,
                "--worker-id", worker_id, "--model-name", name, "--model-dir", models[name],
                "--jobs-file", str(jobs_file), "--work-dir", str(work_dir),
                "--eval-dir", args.eval_dir, "--max-new-tokens", str(args.max_new_tokens),
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log = open(log_path, "a", buffering=1)
            procs.append((worker_id, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"  logging to {log_path}")

        failed = []
        for worker_id, proc, log in procs:
            if proc.wait() != 0:
                failed.append(f"{worker_id} (exit {proc.returncode})")
            log.close()
        if failed:
            print(f"\nWARNING: worker(s) exited non-zero: {', '.join(failed)}", file=sys.stderr)

    results = collect(work_dir, models, tasks)
    report = format_report(results, models, tasks)
    print("\n" + report)

    summary_path = work_dir / "comparison.json"
    summary_path.write_text(json.dumps(
        {"tag": args.tag, "models": models, "bbh_limit": args.bbh_limit,
         "gsm8k_limit": args.gsm8k_limit, "max_new_tokens": args.max_new_tokens,
         "results": results},
        ensure_ascii=False, indent=2))
    (work_dir / "comparison.txt").write_text(report + "\n")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
