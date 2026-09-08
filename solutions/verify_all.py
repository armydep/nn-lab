"""Run every solution and report which succeed.

    python3 solutions/verify_all.py
    python3 solutions/verify_all.py --quick    # skip the slow training runs

Solutions assert their own definitions of done, so a zero exit code here means
every chapter actually met its goal -- not merely that it ran without crashing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SOLUTIONS = Path(__file__).parent
REPO_ROOT = SOLUTIONS.parent
CORPUS = REPO_ROOT / "sample-data" / "shakespeare.txt"

# (label, path, args, needs_corpus, slow)
CHECKS = [
    ("01 single neuron", "01-single-neuron/train.py", [], False, False),
    ("01 optional t2 (two inputs)", "01-single-neuron/optional/t2.py", [], False, False),
    ("01 optional t3 (learning rate)", "01-single-neuron/optional/t3.py", [], False, False),
    ("02 binary classifier", "02-binary-classifier/train.py", [], False, False),
    ("02 optional t2 (non-separable)", "02-binary-classifier/optional/t2.py", [], False, False),
    ("02 optional t3 (log(0) trap)", "02-binary-classifier/optional/t3.py", [], False, False),
    ("03 backpropagation", "03-backpropagation/gradients.py", [], False, False),
    ("03 optional t2 (two paths)", "03-backpropagation/optional/t2.py", [], False, False),
    ("03 optional t3 (planted bug, eps)", "03-backpropagation/optional/t3.py", [], False, False),
    ("04 MLP from scratch", "04-mlp-from-scratch/train.py", [], False, False),
    ("04 optional t2 (symmetry)", "04-mlp-from-scratch/optional/t2.py", [], False, False),
    ("04 optional t3 (no nonlinearity)", "04-mlp-from-scratch/optional/t3.py", [], False, False),
    ("05 micrograd engine", "05-pytorch-mlp/micrograd.py", [], False, False),
    ("05 pytorch MLP", "05-pytorch-mlp/train.py", [], False, False),
    ("05 optional t2 (extend engine)", "05-pytorch-mlp/optional/t2.py", [], False, False),
    ("05 optional t3 (zero_grad)", "05-pytorch-mlp/optional/t3.py", [], False, False),
    ("06 bigram LM", "06-bigram-language-model/train.py", [], False, False),
    ("07 embedding LM", "07-embedding-language-model/train.py", [], False, False),
    ("08 attention", "08-attention/attention.py", [], False, False),
    ("09 tiny transformer", "09-tiny-transformer/model.py", [], False, False),
    ("10 tiny GPT (train)", "10-tiny-gpt/train.py", ["--steps", "60"], True, True),
    ("10 tiny GPT (generate)", "10-tiny-gpt/generate.py", ["--tokens", "40"], True, True),
    ("11 experiments", "11-training-experiments/run_experiments.py",
     ["--steps", "25", "--sweep", "lr"], True, True),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the slow training runs")
    args = parser.parse_args()

    have_corpus = CORPUS.exists()
    if not have_corpus:
        print("NOTE: sample-data/shakespeare.txt missing -- Chapter 10/11 will be skipped.")
        print("      Run: python3 sample-data/download_corpus.py\n")

    results: list[tuple[str, str, float]] = []
    for label, relative, extra, needs_corpus, slow in CHECKS:
        if slow and args.quick:
            results.append((label, "SKIP (quick)", 0.0))
            continue
        if needs_corpus and not have_corpus:
            results.append((label, "SKIP (no corpus)", 0.0))
            continue

        script = SOLUTIONS / relative
        print(f"running {label} ...", end=" ", flush=True)
        start = time.time()
        proc = subprocess.run(
            [sys.executable, str(script), *extra],
            capture_output=True,
            text=True,
            cwd=script.parent,      # Chapter 10 imports its sibling modules
        )
        elapsed = time.time() - start

        if proc.returncode == 0:
            print(f"OK ({elapsed:.1f}s)")
            results.append((label, "OK", elapsed))
        else:
            print(f"FAILED ({elapsed:.1f}s)")
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            for line in tail:
                print(f"    {line}")
            results.append((label, "FAILED", elapsed))

    print("\n" + "=" * 60)
    print(f"{'chapter':<34} {'result':<16} {'time':>6}")
    print("-" * 60)
    for label, status, elapsed in results:
        timing = f"{elapsed:.1f}s" if elapsed else "-"
        print(f"{label:<34} {status:<16} {timing:>6}")

    failed = [label for label, status, _ in results if status == "FAILED"]
    print("=" * 60)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All solutions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
