# Wide-circuit benchmark datasets

The wide-circuit optimization example reads the two source datasets here and
writes optimized OpenQASM 2 circuits with identical filenames into the result
directories selected by `max_partition_size`:

- size 3 writes to `IBMEagle_results_3qbit` and
  `QASMBenchmarks_results_3qbit`, with metadata in `results_3qbit.json`;
- size 4 writes to `IBMEagle_results_4qbit` and
  `QASMBenchmarks_results_4qbit`, with metadata in `results_4qbit.json`.

- `IBMEagle` contains 51 unitary circuits from the IBM Eagle/QMill benchmark
  corpus. They were converted from the `njross/optimizer` Quipper sources with
  [`onestruggler/qasm-quipper`](https://github.com/onestruggler/qasm-quipper).
  The source `_before` marker was removed from the archived filenames. Nine
  generated adder circuits containing non-unitary resets are deliberately
  excluded.
- `QASMBenchmarks` contains the 26 original circuits from the curated
  `processed_qasm` benchmark set used in the BQSKit comparison experiments.
  That set is primarily derived from [PNNL
  QASMBench](https://github.com/pnnl/QASMBench), with additional
  [VeriQBench](https://github.com/Veri-Q/Benchmark) and
  [BQSKit](https://github.com/BQSKit/bqskit) example circuits. Generated
  `_u3cx`, `_squander`, and `_qsearch` results are deliberately excluded.

The two input directories contain 77 circuits in total. The four result
directories are created and populated by
`examples/decomposition/wide_circuit_optimization.py`; generated results are
not stored in the repository.
