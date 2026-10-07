# Universe 7

**A new experimental line of [Dimension42](https://github.com/enderPeer/Dimension42). Machine code only.**

The experiment starts again at one bit and two possibilities. Programs are bit strings. They grow one bit
at a time toward a ceiling of **6 bytes = 48 bits**, and the instruction table was chosen by exhaustive
search so that this universe produces the **largest number of different programs** for its size.

| Generation | Programs | Different screens | New screens never seen before |
|---:|---:|---:|---:|
| 1 bit | 2 | 2 | 2 |
| 2 bits | 4 | 4 | 2 |
| 3 bits | 8 | 8 | 6 |
| 4 bits | 16 | 14 | 10 |
| 8 bits | 256 | 242 | 228 |
| 16 bits | 65,536 | 59,987 | |
| 22 bits | 4,194,304 | 4,138,003 | |
| **1..22 bits** | **8,388,606** | **8,318,667 = 99.17%** | |

Full table: [results/variety-to-22-bits.txt](results/variety-to-22-bits.txt). Every program of 1 to 8 bits
with its screen: [results/generations-1-to-8.txt](results/generations-1-to-8.txt).

## The law of this universe

- **The world** is an 8×8 screen of 64 pixels, all dark at the start, and a head that starts at pixel 0 (top left).
- **A program** is a string of 1 to 48 bits. It is read cyclically: after the last bit comes the first.
- **An instruction** is two bits. There are exactly four, so every bit pattern is a valid program and no
  two codes mean the same thing. There is no NOP and no HALT.
- **A run** is 127 instructions. Then the screen is the result.

| Code | Name | Effect |
|---|---|---|
| `00` | **F4** | flip the pixel under the head, then head += 4 |
| `01` | **F8** | flip the pixel under the head, then head += 8 |
| `10` | **S3** | light the pixel under the head, then head += 3 |
| `11` | **C3** | darken the pixel under the head, then head += 3 |

The head wraps modulo 64. The table lives at file offset `0134` of [`src/u7.hex`](src/u7.hex): eight
bytes, pen then stride for each code.

### Why this table

The score is the number of different final screens over *all* programs up to a length. A perfect universe
scores one new screen per program. The search ([results/table-search.txt](results/table-search.txt)) went in
five rounds:

1. **Hand-picked tables.** The Dimension42 BIT CPU (`0` flips, `1` moves right) scores **17%** up to 14 bits.
   Tables with jumps, skips, and separate move/draw instructions all stay below 21%. Separate "draw" and
   "move" steps waste bits: moving without drawing changes nothing visible, so many programs collapse onto
   the same screen.
2. **Every 3-instruction table** over 93 candidate operations (125,580 tables). Pen operations that draw
   *and* move in one step win by a wide margin: 99.5% up to 11 bits.
3. **Every 4-instruction table** over the 19 best operations (3,876 tables), scored up to 13 bits. Conditional
   skips, loops, and plain moves never enter the top 20. Four pens with strides 4, 8, 3, 3 lead.
4. **The step count.** An even number of steps is a trap: a flip instruction run alone revisits every pixel
   an even number of times and leaves the screen dark, so the 1-bit programs `0` and `1` would be identical.
   127 is odd, close to the original 128, and scores among the best of all odd counts from 65 to 255.
5. **Confirmation up to 22 bits**, all 8,388,606 programs: **99.17%** distinct.

What stops it reaching 100%: a program repeated is the same program. `0000` runs exactly like `00` and `0`,
so at 4 bits at most 14 of 16 screens can be new within the generation. Lengths that divide the run evenly
(8, 16) also collapse a little more. Up to 22 bits, 1,043 programs lose to a cycle on the 48-bit horizon.

## The runner: 316 bytes of x86-64

[`src/u7.hex`](src/u7.hex) is a complete, static Linux executable written byte by byte. There is no
assembler and no compiler. It is a 64-byte ELF header, a 56-byte program header, 180 bytes of code, and the
8-byte instruction table.

~~~bash
python tools/build.py          # hex text -> build/u7 (316 bytes)
./build/u7 0110                # run a program and print the screen
~~~

~~~
$ ./build/u7 0110
#.#.#.#.
...#.#.#
.#....#.
#.#.#...
.#.#.#.#
....#.#.
#.#....#
.#.#.#..
~~~

`tools/build.py` only converts hex to bytes and checks the `=XXXX` offset markers, exactly like the
Dimension42 OS build.

## Host-side observers

These are not part of the universe. They only measure it.

| Tool | What |
|---|---|
| `tools/ref.py` | Reference model. `python tools/ref.py --check 12` runs the built binary on all 8,190 programs up to 12 bits and compares every screen. Result: 0 mismatches. |
| `tools/variety.c` | Counts distinct screens per generation. `gcc -O2 -fopenmp tools/variety.c -o build/variety && ./build/variety 22` reproduces the table above in a few minutes. |
| `tools/table_search.c` | The exhaustive table search that chose the law. |

## Horizon

6 bytes is 2^48 = 281,474,976,710,656 programs. The 22-bit map is 1/67,108,864 of that. The line continues
generation by generation; every result committed here is a snapshot of how far the map reached.
