# [P] Dimension42: exhaustive tiny-program data and small neural program generators

Project: https://github.com/enderPeer/Dimension42

Disclosure: this is our project; this draft was prepared with AI assistance.

Dimension42 explores every one of the 2^40 five-byte programs of NANO, a 16-byte-memory virtual CPU
with self-modifying code. A separate input/output search groups programs into 219,846 signatures
on 16 fixed probe pairs, including 84,933 singleton signatures. Those signatures are not full
truth-table equivalence classes.

There are two different learning experiments:

1. Layout-conditioned addition: after practice on six three-input layouts, sampled programs achieve
85.75–93.15% exhaustive correctness on four held-out layouts. Each successful candidate is checked
on all 16,777,216 triples. However, all successful program identities were already seen under other
training tasks, so this is layout transfer, not unseen-program synthesis.
2. Programming by Example: a small Transformer maps eight input/output examples to five program bytes.
The original version generates random programs live; later versions balance sampling over catalog
signatures while still generating fresh execution examples. Ten function signatures are withheld,
and solutions are checked on all 65,536 pairs. These held-out results are much weaker than the
layout-transfer numbers.

The repository includes source, saved results, a CPU-only self-modification demo, and a single-GPU
reproduction path. We are interested in feedback on separating function-level generalization from
program overlap, and comparing supervised byte prediction with execution-feedback training.
