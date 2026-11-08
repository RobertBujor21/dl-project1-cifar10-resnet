"""Run a list of training runs one after another on this machine's GPU.

A plan file has one run per line: a run name followed by train.py arguments.
Blank lines and lines starting with "#" are ignored. Runs that already
finished (runs/<name>/metrics.json exists) are skipped, so the queue can be
restarted after a crash, a reboot or the laptop going to sleep.

    python run_queue.py plans/ablations.txt --workers 6

The output of each run is shown on screen and saved to logs/<name>.out.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time


def read_plan(path):
    runs = []

    with open(path) as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            name, _, arguments = line.partition(" ")
            runs.append((name, shlex.split(arguments)))

    return runs


def run_and_log(command, log_path):
    # Every line of output is printed and written to the log file as soon
    # as it arrives, so a long run can be followed live.
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        process.wait()

    return process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", help="plan file, e.g. plans/ablations.txt")
    parser.add_argument(
        "--workers", type=int, default=None, help="data-loading workers per run"
    )
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    runs = read_plan(args.plan)
    print(f"{len(runs)} runs in {args.plan}: {[name for name, _ in runs]}")

    for name, run_args in runs:
        if os.path.exists(os.path.join(args.out_dir, name, "metrics.json")):
            print(f"\n=== {name}: already finished, skipping")
            continue

        command = [
            sys.executable,
            "-u",
            "train.py",
            "--name",
            name,
            "--out-dir",
            args.out_dir,
            *run_args,
        ]

        if args.workers is not None:
            command += ["--workers", str(args.workers)]

        print(f"\n=== {name}: {' '.join(command[2:])}", flush=True)
        start = time.time()
        return_code = run_and_log(command, os.path.join("logs", f"{name}.out"))

        if return_code == 0:
            status = "done"
        else:
            status = f"FAILED (exit code {return_code})"

        print(
            f"=== {name}: {status} after {(time.time() - start) / 60:.1f} min",
            flush=True,
        )

    summary_path = os.path.join(args.out_dir, "summary.csv")

    if os.path.exists(summary_path):
        print(f"\nAll runs so far ({summary_path}):")

        with open(summary_path) as file:
            print(file.read())


if __name__ == "__main__":
    main()
