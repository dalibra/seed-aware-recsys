#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(os.environ.get("NOTE_SIG_ROOT", Path(__file__).resolve().parents[1]))
RUNS_DIR = ROOT / "splitlight" / "runs"
DATA_PATH = Path(os.environ.get("SEQ_SPLITS_DATA_PATH", ROOT / "data"))
OUTPUT_ROOT = ROOT / "outputs" / "seed_matrix"
LOG_DIR = ROOT / "logs" / "gru4rec_null"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))

SPLIT = "leave-one-out-no_cold_items"
SEEDS = [11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]
DATASETS = ["Movielens-20m"]


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int


def result_path(job: Job) -> Path:
    return (
        OUTPUT_ROOT
        / job.dataset
        / SPLIT
        / "RNN"
        / "C0"
        / f"test_per_user_seed_{job.seed}.csv"
    )


def log_path(job: Job) -> Path:
    return LOG_DIR / f"{job.dataset}_GRU4Rec_C0_seed{job.seed}.log"


def build_cmd(job: Job, gpu: int) -> list[str]:
    return [
        str(PYTHON_BIN if PYTHON_BIN.exists() else Path(sys.executable)),
        "train_rs.py",
        "model=RNN",
        f"dataset={job.dataset}",
        f"split_name={SPLIT}",
        "run_tag=C0",
        f"metrics_save_dir={OUTPUT_ROOT}",
        f"random_state={job.seed}",
        f"cuda_visible_devices={gpu}",
        "max_length=50",
        "model.model_params.input_size=64",
        "model.model_params.hidden_size=64",
        "model.model_params.num_layers=1",
        "model.model_params.dropout=0.1",
        "seqrec_module.lr=0.001",
        "dataloader.batch_size=1024",
        "dataloader.test_batch_size=2048",
        "dataloader.validation_size=5000",
        "dataloader.validation_sample_seed=20260216",
        "dataloader.num_workers=4",
        "trainer_params.max_epochs=50",
        "patience=5",
        "calc_val_metrics=false",
        "evaluator.metrics=[NDCG,HitRate,MRR]",
        "evaluator.top_k=[10,20]",
    ]


def make_jobs(args: argparse.Namespace) -> list[Job]:
    datasets = args.datasets or DATASETS
    seeds = args.seeds or SEEDS
    jobs = [Job(dataset, seed) for dataset in datasets for seed in seeds]
    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


def main() -> int:
    global DATA_PATH, OUTPUT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    DATA_PATH = args.data_path
    OUTPUT_ROOT = args.output_root

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(args)
    pending = [job for job in jobs if args.force or not result_path(job).exists()]
    skipped = len(jobs) - len(pending)
    print(f"scheduled={len(pending)} skipped_completed={skipped} total={len(jobs)}", flush=True)

    if args.dry_run:
        for job in pending:
            print(job, "->", result_path(job))
            print(" ".join(build_cmd(job, args.gpus[0])))
        return 0

    env = os.environ.copy()
    env["SEQ_SPLITS_DATA_PATH"] = str(DATA_PATH)
    env.setdefault("PYTHONUNBUFFERED", "1")

    running: dict[int, tuple[subprocess.Popen, Job, object]] = {}
    free_gpus = list(args.gpus)
    completed = skipped

    while pending or running:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            job = pending.pop(0)
            lp = log_path(job)
            lp.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(lp, "w", buffering=1)
            cmd = build_cmd(job, gpu)
            print(f"start gpu={gpu} dataset={job.dataset} seed={job.seed} log={lp}", flush=True)
            log_fh.write(" ".join(cmd) + "\n\n")
            proc = subprocess.Popen(cmd, cwd=RUNS_DIR, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            running[gpu] = (proc, job, log_fh)

        time.sleep(10)
        for gpu, (proc, job, log_fh) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            log_fh.close()
            del running[gpu]
            free_gpus.append(gpu)
            free_gpus.sort()
            if ret != 0:
                print(f"failed dataset={job.dataset} seed={job.seed} exit={ret} log={log_path(job)}", flush=True)
                for other_proc, _, other_log_fh in running.values():
                    other_proc.terminate()
                    other_log_fh.close()
                return ret
            completed += 1
            print(f"done dataset={job.dataset} seed={job.seed} completed={completed}/{len(jobs)}", flush=True)

    print("all jobs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
