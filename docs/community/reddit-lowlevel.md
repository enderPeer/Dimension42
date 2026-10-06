# Five bytes that rewrite an XOR instruction into an output instruction

Project: https://github.com/enderPeer/Dimension42

Disclosure: this is our project; this draft was prepared with AI assistance.

One example from our NANO experiments is `17 C5 91 50 71`. With a in M5 and b in M6,
it computes a XOR b XOR 7. It increments its own XOR instruction from C5 through CF;
the next increment changes it to D0, meaning OUT M0. The initial instruction's memory cell
has meanwhile become storage for the answer. All 65,536 input pairs pass under the 64-step limit.

Another example, `67 A0 5C E2`, adds three input bytes in four bytes of code by changing
ADD M7 into ADD M6 and ADD M5, then rewriting other instructions to store and output the result.

We mapped all 1,099,511,627,776 five-byte programs into broad behavior classes. A separate 16-probe
catalog has 84,933 singleton signatures; 98.67% of their programs change code before answering on
at least one probe. This does not establish 84,933 distinct fully verified functions.

The README has instruction-by-instruction explanations and a standard-library Python demo.
The repository also contains the handwritten x86 OS and CPU/CUDA/Vulkan interpreters.
Feedback on tiny instruction-set design and independent interpreter verification would be useful.
