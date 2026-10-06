---
pretty_name: Dimension42 NANO execution trace sample
size_categories:
- n<1K
tags:
- program-synthesis
- execution-traces
- nano
- self-modifying-code
---

# Dimension42 NANO execution trace sample

A small reproducible dataset from [Dimension42](https://github.com/enderPeer/Dimension42):
**160 instruction traces from five programs**, covering addition modulo 256 and XOR with 7.
Each underlying program was checked on all **65,536 input pairs** with the independent existing Python reference.
The trace-producing interpreter is cross-checked against that reference for the exported inputs.

## Files and schema

- `traces.jsonl`: one row per program/input pair. Fields: `program_hex`, `task`, `input_a`, `input_b`, `output`, `trace`.
- `trace`: instruction steps with PC, executed byte, decoded instruction, accumulator before/after, changed cells, and output.
- `verification.json`: exhaustive-check counts, output-table hashes, and the reference interpreter's SHA-256.

Execution uses five-byte code in M0–M4, a in M5, b in M6, zero-initialized A and other data cells,
and the first OUT within 64 instructions. A timeout or missing answer is a failure.

Regenerate from the source repository:

```bash
python tools/export_trace_sample.py
```

Load locally with the Hugging Face `datasets` package:

```python
from datasets import load_dataset
sample = load_dataset("json", data_files="datasets/nano-adders-sample/traces.jsonl", split="train")
```

The loader's `train` label does not define a research split. **This is a demonstration sample, not a
balanced or held-out generalization benchmark.** Four programs share the addition function;
the fifth is the self-modifying XOR example. These programs and tasks are publicly known.
Do not report performance on this sample as unseen-program or unseen-function generalization.

## Provenance and publication

Four programs are the first four entries in `explorer/results/add2/adders_L5.txt`;
the fifth is `17C5915071`. Sample pairs use seed 42 plus fixed edge cases.
Verification applies to the specified finite domain and reference implementation, not arbitrary runtimes.
No personal data or cluster credentials are included.

This folder is ready for a Hugging Face dataset repository but has **not been uploaded there**.
The repository owner must choose an explicit redistribution license before a licensed dataset release;
this card intentionally does not assign a license on the owner's behalf.
