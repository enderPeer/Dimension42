# Experimental NANO-256 / 512 steps

`nano256-bank-v1` is a separate CPU-only experiment. The original NANO interpreters,
five/six-byte maps, trained models, and handwritten OS keep their original semantics.
Nothing here restarts, replaces, or attaches to an active cluster worker.

## Machine definition

| Property | Rule |
|---|---|
| Memory | 256 byte cells, absolute addresses 0–255 |
| Program | 1–16 bytes, loaded at absolute address 0; normally five bytes for existing experiments |
| State | 8-bit accumulator A, bank register B in 0–15, PC |
| Instruction size | One byte, high nibble operation / low nibble argument n |
| Default budget | 512 executed instructions; configurable 1–65,535 |
| Initialization | A=B=PC=0; non-code memory zero unless initial data is supplied |
| Code fetch | Always absolute M[PC]; PC wraps at program length, unaffected by bank selection |
| Data address | Memory operations use `16*B+n` |
| Display | Still absolute M8–MF, an 8×8 monochrome display; extra RAM does not itself enlarge the display |
| Stop | HLT or the step limit; optional first-output evaluation |
| Output capture | First eight output bytes plus an accurate total count; trace mode records every output event |

### Instruction table

| Opcode | Operation |
|---|---|
| `0n` | **BANK n:** B=n (this replaces original NOP encodings) |
| `1n`, `2n`, `3n` | A=n, A+=n, A-=n; byte arithmetic wraps modulo 256 |
| `4n`, `5n`, `6n` | Load, store, add using M[16*B+n] |
| `7n`, `8n` | Jump to n modulo program length; conditional jump requires A=0 |
| `9n`, `An` | Increment/decrement M[16*B+n] |
| `Bn` | Rotate A left by n modulo 8 |
| `Cn`, `Dn`, `En` | XOR A with, output, or invert M[16*B+n] |
| `Fn` | Halt; argument ignored |

Bank zero retains access to the program, allowing self-modification. Other banks provide
additional data storage. Selecting bank 15 and address F reaches absolute address 255.
Instruction fetch and immediate arguments never use the bank. Programs longer than eight
bytes overlap the fixed display region because code and data still share memory.

**This is not binary-compatible with arbitrary old NANO programs:** nonzero `0n` bytes
used to do nothing. Existing model weights and search results cannot be treated as verified
for this ISA. Even bank-zero programs can have different final states when run longer.

## Run locally with Python

From the repository root; Python 3.10+ and no additional packages are needed:

```bash
# Five bytes: select bank F, load 1, write address FF, output it, halt.
python explorer/nano256_ref.py "0F 11 5F DF F0" --trace

# Output its own byte 512 times: steps=512, output_count=512 (not truncated to a byte).
python explorer/nano256_ref.py D0

# Increment cell FF 256 times over 512 instructions; final byte wraps to 0.
python explorer/nano256_ref.py "0F 9F 71"

# Supply data outside the code and stop on the first output.
python explorer/nano256_ref.py "01 4F DF F0" --set 0x1f=213 --first-output

python -m unittest discover -s tests -p test_nano256.py -v
```

Each invocation starts a fresh VM. Memory assignments that overwrite initial code are rejected;
the program itself can still modify its code through bank zero. JSON includes the ISA identifier,
budget, registers, all memory, first outputs, output count, stop reason, and state checksum.

## Native CPU interpreter

On a machine with GCC:

```bash
gcc -O3 -std=c11 -Wall -Wextra explorer/nano256_cpu.c -o explorer/nano256_cpu
./explorer/nano256_cpu 0F115FDFF0
./explorer/nano256_cpu D0 512
./explorer/nano256_cpu 014FDFF0 512 1 0x1f=213
```

The positional arguments after code are step budget, first-output flag (0/1), and optional
`address=value` initial-memory assignments. `--batch` reads one whitespace-separated request per
line from standard input and emits one JSON result per line. Use uninterrupted hex in batch mode.
The C runner exposes final state, not instruction traces; use Python's `--trace` for dissection.

## Verification and limits

Validation: seven focused Python tests passed. The native C build passed GCC's
`-Wall -Wextra -Werror` checks and matched the Python interpreter on **3,331 complete
state comparisons**, including all one-byte programs at five step budgets, randomized
banked inputs, self-modification, and a 65,535-output counter case. Nine invalid native
requests were also rejected. These are test results, not a formal proof.

Step and output counters support 512 and beyond. State hashes use FNV-1a-32 with a new
`NANO256v1` prefix, then byte fields (length, A, bank, PC, halted, trace-A, trace-bank, class,
ever-self-modified), little-endian 32-bit step/output counts, memory[256], seen-memory-bits[256],
and the first eight outputs padded with zeros. Old NANO hashes are not comparable.

Class priority is DRAW > OUTPUT > SELF-MOD > ACTIVE > HALT > IDLE. DRAW uses bits seen in
absolute M8–MF. SELF-MOD uses changed final code bytes, while `self_modified` separately records
any changed code write. ACTIVE includes nonzero accumulator/bank history and non-display data
bits seen during execution, including initially supplied data. These broad labels do not prove
a useful input/output function. Hashes can collide; interpreter checks compare the full JSON state.

The new memory is 16 times larger, and the default budget allows up to 8 times as many instructions.
Those ratios are not measured runtime multipliers. CUDA/Vulkan, the handwritten kernel,
distributed enumeration, and AI training have **not** been ported to NANO-256.
