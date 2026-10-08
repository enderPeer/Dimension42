# Artificial-life toolkit: all 5-byte NANO programs

Source: exhaustive scan of all 1,099,511,627,776 five-byte programs (`nano_alife.cu`, 482 s on 5 GPUs). Every
stored program (199,310) was re-checked with an independent Python implementation, and all are identical.
Rules: no inputs, 64 steps, 16-byte memory, the program sits in M0–M4.

| building block | definition | programs | list |
|---|---|---:|---|
| Replicator | exact copy of its own 5 bytes appears elsewhere in memory (M5–M15) | **316** | `replicators.txt` (all) |
| Mutating replicator | a copy with exactly one written byte different (reproduction + mutation) | **52,261** | `mutating_replicators.txt` (all) |
| Repairer, 2 positions | 2 of its own bytes get damaged before the start (3 wrong values each) and are exactly restored every time | **3,666** | `repairers_multi.txt` (all) |
| Repairer, 1 position | the same for one byte | 126,319,172 | `repairers_sample.txt` (sample) |
| Self-healer | changes its own code while running, ends exactly original | 12,796,352,074 | `healers_sample.txt` (sample) |
| Walker | changes the operand of its own instructions at least 4 times (pointer engine) | 114,222,087,156 | `walkers_sample.txt` (sample) |
| Stable organism | never halts, does work, code intact after 64 steps | 481,976,596,898 | `stable_sample.txt` (sample) |

## Examples

**Replicator** `55 41 90 91 55` = ST 5 · LD 1 · INC 0 · INC 1 · ST 5
Uses its own first two bytes as write and read pointers, counts them up, and copies itself byte by byte
to M5–M9 within 21 steps. The original's pointer bytes are worn afterwards, so no 5-byte replicator keeps its
original intact.

**Mutating replicator** `41 55 12 91 90` = LD 1 · ST 5 · LDI 2 · INC 1 · INC 0
Child at M8: `90 55 12 91 90`, with the first instruction changed from `LD 1` to `INC 0`.
In 45,138 of 52,261 cases only an operand changes in the child (for example which cell is read), and in 7,123
a whole instruction changes. Every mutating replicator is also a walker.

**Repairer** `11 52 01 54 01` = LDI 1 · ST 2 · NOP · ST 4 · NOP
Writes the value 01 into its byte 2 and byte 4 in every round, and that's exactly what belongs there. If someone
sets these bytes to 00, 42 or F0 (= HLT), they're repaired before they're ever executed. It's a protected
program with two "immune" positions.

## What 5 bytes cannot do

- No program is both a replicator and a repairer (0 of 316), and none is a replicator and a stable organism.
- Nothing protects more than 2 of its 5 bytes.

Combinations need more room. That's the case for a larger NANO memory, where programs are put together
from these building blocks and evolution does the combining.
