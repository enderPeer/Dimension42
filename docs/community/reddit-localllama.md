# Training small local Transformers to write five-byte programs from examples

Project: https://github.com/enderPeer/Dimension42

Disclosure: this is our project; this draft was prepared with AI assistance.

Our Programming by Example setup trains a small Transformer to map eight input/output pairs to
a five-byte NANO program. The initial model has about 14.4M parameters; a later experiment uses
38.1M. These are purpose-trained program models, not general chat models.

We first explored the full five-byte program space. Catalog-based training samples each of
219,836 retained probe signatures equally, then generates fresh execution examples. The original
PBE version instead samples random programs live. Training uses supervised next-byte prediction;
execution checks evaluate generated candidates but do not currently supply training rewards.

Ten target-function signatures are excluded from training. Generated candidates must pass all
65,536 input pairs for correctness. Generalization remains limited even while training loss falls.
A separate addition experiment achieves approximately 86–93% on unseen three-input layouts,
but its successful program identities overlap training; that number is not the PBE success rate.

The code, results, a CPU-only demo, and a single-NVIDIA-GPU reproduction path are public.
We are interested in execution-guided repair, better sampling, and verification-based self-training
for tiny models where the entire input domain can be checked.
