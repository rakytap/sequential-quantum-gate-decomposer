This snapshot refreshes all 118 previously completed OSR cases (3q and 4q), reusing their audited A2A circuits and saved permutation libraries. The 36 incomplete cases retain their recorded statuses. Counts never increase relative to the archived OSR results. The original 28 October snapshot remains unchanged in Git history and `results_archive_20261028/`.

Paired CNOT comparisons with BQSKit:

- A2A (unchanged): 61 wins, 23 ties, 1 loss.
- Routing: 83 wins, 2 ties, 0 losses.
- Final: 84 wins, 11 ties, 2 losses.

See [comparison.tex](comparison.tex), [routing-replay.csv](routing-replay.csv), and [routing-replay-summary.json](routing-replay-summary.json). The routed and final columns retain their respective best verified candidates. Candidate audits bind improved raw routes to their own final outputs when an older complete pipeline wins.

Times are cumulative optimization costs: the original complete experiment plus measured routing/final extensions and conservative charges for interrupted final passes. They include host stalls and recovery delay. They are not fresh-run timings or a controlled speed comparison. Original wall/audit measurements remain in `routing_replay.original_timing`; new wall/audit totals are not claimed.

Verify the small result files from the Git revision containing this snapshot:

```bash
python benchmarks/partitioning/archive_results.py verify \
  --manifest benchmarks/partitioning/results_archive_20261111/manifest.json
```

The gzip proofs and saved synthesis libraries are distributed separately from Git. Their asset hashes and file paths are recorded in [manifest.json](manifest.json) and [SHA256SUMS](SHA256SUMS). The prepared assets are kept outside the Git checkout and have not been uploaded automatically. After pushing this branch, create or use release `benchmark-results-20261111` in `rakytap/sequential-quantum-gate-decomposer` and attach every `.tar` plus `SHA256SUMS` from that directory. Each tar is below GitHub's 2 GiB asset limit. Git push alone does not upload this evidence.

After the release assets have been uploaded, download and verify them from a checkout containing this snapshot:

```bash
gh release download benchmark-results-20261111 --repo rakytap/sequential-quantum-gate-decomposer \
  --dir /tmp/benchmark-results-20261111 --pattern '*.tar' --pattern SHA256SUMS
(cd /tmp/benchmark-results-20261111 && sha256sum --check SHA256SUMS)
for asset in /tmp/benchmark-results-20261111/*.tar; do
  tar -xf "$asset" -C .
done
python benchmarks/partitioning/archive_results.py verify --include-evidence \
  --manifest benchmarks/partitioning/results_archive_20261111/manifest.json
```

Checksum checks establish artifact integrity. The result records also retain the independent rewrite-audit verification reports. No remote branch, pull request, or release is created by packaging.
