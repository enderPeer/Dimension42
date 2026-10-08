# NANO explorer on the cluster

## Program explorer (all 1,099,511,627,776 five-byte programs)

```bash
python explorer/app/server.py
```

Then open http://127.0.0.1:8742 (local only). It needs `numpy` and `zstandard`.

- **Map**: all programs as 1024 chunks, then a chunk as 1024×1024 pixels (each pixel = 1024 programs, coloured by
  class mix), then a block of 1024 single programs
- **Program**: its 5 instructions, step-by-step playback (A, PC, memory, 8×8 display now and over all steps, output)
- **Search** any program by hex (`A0 9F 00 D8 F0`), or jump to a **random program of a class**
- Every program you open is re-simulated live in the browser and compared with the stored result
  (checked on 20,480 programs: 0 mismatches)
- Data: `results/L5/chunk_NNNN.zst` (1024 files, 3.4 GB; byte i of chunk N = class of program N·2^30+i) and
  `results/L5/manifest.json`. A chunk picture is cached in `results/L5/cache/` the first time it is opened (~4 s).

5-byte run: 159.6 s on 9 GPUs, 6.9 G programs/s. The bottleneck was the AMD GPUs writing results to uncached memory.
With cached memory (fixed) a repeat run took 91.6 s at 12.0 G programs/s and gave identical results for all 1024 chunks.

## Which programs answer 2 + 2 = 4?

`python explorer/search_run.py` (`nano_search.cu`, `nano_search.comp` + `nano_search_vk.c`, check: `nano_search_cpu.c`).
Input: a in M5, b in M6. Answer: the first value printed with OUT. All 1,103,823,438,080 programs of 1–5 bytes
were searched in 45 s on 9 GPUs. Every adder candidate was proven on all 65,536 input pairs (1.4 s).

| bytes | answer 4 | real adders | clean (1 answer + HLT) | always 4 | 2·a | 2·b | a·b | coincidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 3 | 1,488 | 0 | 0 | 1,480 | 0 | 0 | 0 | 8 |
| 4 | 709,366 | 100 | 0 | 700,564 | 75 | 77 | 0 | 8,550 |
| 5 | 280,936,873 | 74,527 | 1,407 | 275,329,221 | 49,856 | 51,387 | 0 | 5,431,882 |

- Shortest real adder: 4 bytes, `45 66 50 D0` = LD M5 · ADD M6 · ST M0 · OUT M0. It needs every one of those
  4 steps, so programs of 1–3 bytes can never add.
- Of the 5-byte programs that say 4 for 2+2, only 0.027 % really add. 98 % just print a constant 4.
- No program up to 5 bytes multiplies.
- Checks: CPU, CUDA and Vulkan give identical counts and adder lists on 8 test ranges (incl. all 1–3-byte programs).
  The Python reference agrees on 3,100 proven adders × 303 input pairs.
- Results: `results/add2/summary.json` and `results/add2/adders_L4.txt` / `adders_L5.txt`. In the explorer, click
  **2 + 2 = 4 ?** to browse them, and enter any inputs a, b on a program to run it.

## An AI that writes adders (`ai/train_adder_ai.py`)

A small transformer (803,874 parameters) learns to write NANO programs from the proven adders. The prompt is
`ADD <cell of a> <cell of b>` and the answer is 10 hex digits (5 bytes). Data: the exhaustive search was rerun for
11 input layouts (about 45 s each). Training used all proven adders of 8 layouts (1.2 M examples, 35 s on the RTX 4090).
Every program it writes is checked exactly on all 65,536 input pairs.

| test (2,000 programs written each) | proven adders | new (not in training) | random guessing |
|---|---:|---:|---:|
| trained layout M5+M6 | 96.1 % | – | 0.000007 % |
| **new combination M6+M8** (cells known, pair never seen) | **91.1 %** | 1,665 of 1,774 | 0.000007 % |
| **new combination M7+M9** | **87.1 %** | 1,611 of 1,716 | 0.000007 % |
| unknown cell M5+M10 (M10 never in training) | 4.0 % | 13 of 66 | 0.000007 % |
| unknown cell M9+M10 | 6.6 % | 58 of 122 | 0.000007 % |

- New combinations: it learned the pattern (load a, add b, store, output) and binds the right cells. Almost
  all of its correct programs are new.
- Unknown cell: it fails, as expected, since it never learned what "M10" means. The few hits are programs that
  add three cells, e.g. `65 6A 69 59 D9` = M5 + M10 + M9, where the extra cell happens to be 0.
- Every correct program it wrote is also in the exhaustive list, which is a consistency check between the two methods.
- Model: `ai/adder_ai.pt`, numbers: `ai/results.json`.

## Test: can the AI add THREE numbers? (`ai/test_three.py`)

Ground truth: `search3_run.py` (`nano_search3.cu` / `.comp` / `_vk.c`, check `nano_search3_cpu.c`). For 10 layouts
(three cells out of M5–M9), all 1.1 trillion programs were searched, then every candidate was proven on all
16,777,216 triples. About 275 million 5-byte programs answer 6 for 2+2+2, but only 423–2,504 per layout really add
three numbers. Some 4-byte programs manage it by rewriting their own code: in `67 A0 5C E2`, `DEC M0` turns
`ADD M7` into `ADD M6` and then `ADD M5`.

Every program the AI writes (2,000 per layout) climbs a staircase:
step 1 = answers 2+2+2 = 6, step 2 = correct on 32 triples, step 3 = correct on all 16,777,216 triples.

| layout (never seen as a triple) | A: no practice, steps 1 / 2 / 3 | B: + examples of 6 other layouts, steps 1 / 2 / 3 | B: share of all adders found |
|---|---|---|---|
| M5 M6 M8 | 367 / 336 / 336 | 1,773 / 1,735 / **1,735** | 443 of 626 (71 %) |
| M5 M7 M9 | 83 / 66 / 66 | 1,865 / 1,863 / **1,863** | 411 of 440 (93 %) |
| M6 M7 M8 | 65 / 53 / 53 | 1,723 / 1,715 / **1,715** | 656 of 1,703 (39 %) |
| M6 M8 M9 | 61 / 45 / 45 | 1,784 / 1,781 / **1,781** | 576 of 847 (68 %) |

- A (no practice): 2–17 % pass. It mostly writes two-number adders (2+2+2 → "4" in 37–68 % of cases).
  The hits are programs from its training list that happen to add three cells.
- B (6 example layouts): 86–93 % pass all three steps, on cell combinations it never saw as a triple.
- Every program that passed step 2 also passed the full proof, so 32 test triples were already enough to catch
  every fake.
- No program is "new": every three-number adder is also a two-number adder (with the third cell at 0), so all of
  them were already in the training data under a different task. The test therefore measures whether the AI
  picks the right programs for an unseen task, not whether it invents new ones.

## Map of all 6-byte programs (`cluster6.py`)

All 281,474,976,710,656 programs (2^48), the class of every one stored zstd-compressed on the cluster:
65,536 files `~/dimension42-explorer/L6/xx/yy.zst` (chunk = programs `xx yy 00 00 00 00` … `xx yy FF FF FF FF`),
794 GB in total. Index of which file lives on which node, with counts and hashes: `results/L6/chunks.jsonl`.

| node | chunks | size |
|---|---:|---:|
| adler40 | 34,003 | 428 GB |
| falke64 | 13,860 | 177 GB |
| specht32 | 10,942 | 120 GB |
| knecht24 | 6,731 | 69 GB |

Classes: DRAW 40.3 %, ACTIVE 16.5 %, OUTPUT 13.8 %, SELF-MOD 12.3 %, HALT 10.0 %, IDLE 7.1 %.

- Run: 2026-10-06 19:13 to 2026-10-07 10:47, in two parts. falke64 dropped off the network at 21:09:34
  (its Intel I226-V network chip lost its PCIe link: `igc … PCIe link lost, device now detached`). Falke kept
  running without network until it was restarted the next morning; the rest was computed by 7 GPUs.
- Checks: all 13,860 chunks on falke64 decompressed and recounted (0 errors). 100 random chunks each on the
  other nodes recounted (300/300). 12 random chunks recomputed on the other GPU family (NVIDIA ↔ AMD): all
  identical. File count = log on every node. 7 half-written files from the failure were removed (their chunks
  were recomputed elsewhere).
- Peak temperatures: GPU hotspot 97 °C (RX 9060 XT, limit 110 °C), NVIDIA up to 74 °C, no thermal throttling.

## Tools

Runs **every** NANO program of L bytes (the CPU from Dimension42 v0.2, PC wrapping at L) for 64 steps
and classifies it. Nothing is skipped, deduplicated or estimated: every program is simulated in full.

| file | runs on |
|------|---------|
| `nano_cpu.c`  | all CPUs (C + OpenMP, `gcc -O3 -march=native -fopenmp`). Also defines the exact rules |
| `nano_cuda.cu`| RTX 4090, 4080, 3× 3060 (`nvcc -O3 -arch=native`) |
| `nano.comp` + `nano_vk.c` | RX 9070 XT, RX 9060 XT, 2× R9700 (Vulkan, `glslc` + `gcc -lvulkan`) |
| `nano_ref.py` | slow Python reference, used for checking |
| `cluster_run.py` | splits the space by measured speed, runs all 13 devices at once, collects and checks results |

On the nodes, sources and binaries live in `~/dimension42-explorer/` (user `ender`).

## How correctness was checked

- C vs. the Dimension42 OS itself (QEMU memory dump), 1-byte programs: 256/256 identical
- C vs. Python reference: 256/256 1-byte and 6,000/6,000 3-byte programs from 6 regions, identical hash of the full final state
- **All 16,777,216 three-byte programs** computed independently on 13 devices (4 CPUs, 5 CUDA, 4 Vulkan GPUs):
  identical class files (md5 `5e5432ca36481f136c34ee2e3891ddd2`) and identical hash over all final states (`36026779016276479`)
- **All 4,294,967,296 four-byte programs** computed independently on all 9 GPUs: identical hash (`9223496200659325639`)
- This caught a real bug: on the RX 9060 XT, one long Vulkan submission ran into amdgpu's ~2 s graphics-ring watchdog.
  The driver reset the ring (twice, 15:12:41 and 15:13:06 on specht32, recovered) and **silently lost ~92 million programs**.
  Fixed by using the compute queue and short submissions. Every worker now also refuses to report unless class counts add up to the program count.

## Results: 3-byte programs (16,777,216)

| class | programs |
|-------|---------:|
| IDLE | 2,765,336 |
| HALT | 1,644,086 |
| ACTIVE | 4,832,745 |
| SELF-MOD | 1,098,060 |
| OUTPUT | 2,076,629 |
| DRAW | 4,360,360 |

Class of every program: `results/classes_L3.bin` (16 MiB, byte i = class of program i).

| how | compute | end-to-end from this PC |
|-----|--------:|------------------------:|
| whole cluster, 13 devices | 9.6 ms (slowest device) | 2.0 s |
| RTX 4090 alone | 4.2 ms | 2.0 s |
| best CPU alone (i9-12900K, 24 threads) | 219 ms | |

For 3 bytes the computing takes milliseconds. The 2 seconds are SSH, GPU driver start-up and copying 16 MiB to this
PC over Wi-Fi. A single 4090 is just as fast end-to-end as the whole cluster.

## How long bigger spaces take

Measured sustained speed (each GPU ran the full 4-byte space on its own):
4090 4.56, 4080 2.82, R9700 2.16 + 2.03, 9070 XT 2.02, 9060 XT 1.12, 3060 3× 0.74, CPUs together 0.17
→ **cluster ≈ 17.1 billion programs/second**.
Real full-cluster run of all 4-byte programs: 369 ms compute (11.6 G/s, because short runs don't reach full clock), 1.9 s end-to-end.

| program size | programs | full cluster |
|---|---:|---:|
| 1 byte | 256 | instant |
| 2 bytes | 65,536 | 4 µs |
| 3 bytes | 16.8 million | 1 ms (measured: 10 ms compute, 2 s end-to-end) |
| 4 bytes | 4.3 billion | 0.25 s (measured: 0.37 s compute, 1.9 s end-to-end) |
| 5 bytes | 1.1 trillion | ~1 minute |
| 6 bytes | 281 trillion | ~4.6 hours |
| 7 bytes | 72 quadrillion | ~49 days |
| 8 bytes | 18.4 quintillion | ~34 years |
| 9 bytes | | ~8,800 years |
| 10 bytes | | ~2.2 million years |

Each extra byte multiplies the search space by 256. This timing table is historical: it predates the
completed five-byte run. Current GPU explorers accept lengths 1–8; the five-byte map's measured wall time
is in `results/L5/manifest.json`. Six-byte mapping finished on 2026-10-07 (65,536 chunks, 794 GB on the cluster, about 54 GPU-hours; classes DRAW 40.3 %, ACTIVE 16.5 %, OUTPUT 13.8 %, SELF-MOD 12.3 %, HALT 10.0 %, IDLE 7.1 %); only sampled phenotype and rarity probes (8.4 M and 20 M programs) have used it, an exhaustive six-byte I/O catalog is open. See the root README for the current
quickstart and the distinction between behavior classification and verified input/output functions.
