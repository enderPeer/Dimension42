# Universe 7

**A new experimental line of [Dimension42](https://github.com/enderPeer/Dimension42). Machine code only.**

The experiment starts again at one bit and two possibilities. Programs are bit strings. They grow one bit
at a time toward a ceiling of **6 bytes = 48 bits**, and the instruction table was chosen by exhaustive
search so that this universe produces the **largest number of different programs** for its size.

| Generation | Programs | Different screens |
|---:|---:|---:|
| 1 bit | 2 | 2 |
| 2 bits | 4 | 4 |
| 3 bits | 8 | 8 |
| 4 bits | 16 | 16 |
| 8 bits | 256 | 256 |
| 16 bits | 65,536 | 65,536 |
| 22 bits | 4,194,304 | 4,194,304 |
| **1..22 bits** | **8,388,606** | **8,388,606 = 100%** |

Every program up to 22 bits draws a picture no other program of its length draws. Full table:
[results/variety-to-22-bits.txt](results/variety-to-22-bits.txt). Every program of 1 to 6 bits with its
screen: [results/generations-1-to-6.txt](results/generations-1-to-6.txt). Across generations the only
repeats are the inevitable ones: `0000` is the same program as `00` and `0`.

The first version of this universe had an 8×8 screen and reached 99.17% on the same programs; its numbers
are kept in [results/table-search.txt](results/table-search.txt). The law did not change when the screen
grew, only the wrap of the head.

## The law of this universe

- **The world** is a 64×64 screen of 4,096 pixels, all dark at the start, and a head that starts at pixel 0 (top left).
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

The head wraps modulo 4,096, so stride 64 would be one row down. The table lives at file offset `0158` of
[`src/u7.hex`](src/u7.hex): eight bytes, pen then stride for each code.

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
5. **Confirmation up to 22 bits** on the original 8×8 screen: **99.17%** distinct. The missing 0.83% were
   programs whose head orbit closed within 127 steps on 64 pixels and erased part of its own drawing.
6. **The screen grows to 64×64.** With 4,096 pixels the head never wraps inside a run, so paths stop
   erasing each other. 894 of 5,985 candidate tables now tie at 100% on short programs. The original
   law was kept because it stays at **100.00% on all 8,388,606 programs up to 22 bits**, while most rivals
   fall apart when the step count grows (see round 6 in the search record).

The screen is now larger than the program space: 2^4096 pictures against at most 2^48 programs. Variety is
no longer limited by the canvas, only by the law and the 127-step run.

## The runner: 352 bytes of x86-64

[`src/u7.hex`](src/u7.hex) is a complete, static Linux executable written byte by byte. There is no
assembler and no compiler. It is a 64-byte ELF header, a 56-byte program header, 224 bytes of code, and the
8-byte instruction table. The screen is 512 bytes on the stack; one `btc`, `bts`, or `btr` with a register bit
offset draws a pixel.

~~~bash
python tools/build.py          # hex text -> build/u7 (352 bytes)
./build/u7 0110                # run a program and print the 64x64 screen
~~~

~~~
$ ./build/u7 0110 | head -4
#.......#..#.......#..#.......#..#.......#..#.......#..#.......#
..#.......#..#.......#..#.......#..#.......#..#.......#..#......
.#..#.......#..#.......#..#.......#..#.......#..#.......#..#....
...#..#.......#..#.......#..#.......#..#.......#..#.......#..#..
~~~

(The first four of 64 rows.)

`tools/build.py` only converts hex to bytes and checks the `=XXXX` offset markers, exactly like the
Dimension42 OS build.

## Host-side observers

These are not part of the universe. They only measure it.

| Tool | What |
|---|---|
| `tools/ref.py` | Reference model. `python tools/ref.py --check 12` runs the built binary on all 2,046 programs up to 10 bits and compares every screen. Result: 0 mismatches. |
| `tools/variety.c` | Counts distinct screens per generation. `gcc -O2 -fopenmp tools/variety.c -o build/variety && ./build/variety 22` reproduces the table above in about ten minutes. |
| `tools/table_search.c` | The exhaustive table search that chose the law. |

## Horizon

6 bytes is 2^48 = 281,474,976,710,656 programs. The 22-bit map is 1/67,108,864 of that. The line continues
generation by generation; every result committed here is a snapshot of how far the map reached.
