# Slice 4 Step 4a handback — R-strict and apply timing
> **From:** Step 4b · **To:** planning · **Date:** 2026-10-07 ·
> **Slice:** M-F5a task-4 — four attribution routes (REQ-004, REQ-005, CAP-004) ·
> **Status:** NOT code-ready. Two gaps. Q2 is disposed. Q1 waits on the Research Manager ·
> **Binder:** `/tmp/rev-mf5a-t4-step4b/REVIEW.md` sha256 `d67c233d561aabd8a10578cc72947c1c75ab9199e19d9c6cfe78473ff44548f0` ·
> **Tip:** `226ffc294b82266d20f551d8346da11674699723` · Developer code stays uncommitted ·
> **Stage:** Step 4b is in flight under REQUIRED CHANGES. C0 text that says the Developer has not started is stale. This file does not flip `step-4b-authorized` and does not open a counted (c)

## Context

The descriptor builds. R-base, R-fused, and R-hybrid execute. R-strict raises and does not fall back. The published apply time is `result.runtime_ms`, the whole partition loop. No counted (c), no shipped closeout, and no four-route bundle until Q1 is picked. G-13 stays open.

## Question 1 — R-strict on the frozen anchor

**Contract:** ADR-F5A-001 names R-strict as `execute_partitioned_density_channel_native`, apply label numpy Kraus. REQ-004 wants that row at the same widths and frozen workload. ADR-F5A-006 forbids a second workload. REQ-005 and CAP-004 forbid a channel-native runtime edit. Mini-spec §2 hands back on a raise and invents no specs.

**What Step 4b found:** The code raises `ValueError("attribution route handback for R-strict: …")`, chained to `NoisyRuntimeValidationError` (`unsupported_runtime_operation` / `channel_native_noise_presence` / `runtime_preflight`). No timing, no fallback, no substitute route. The schema still requires all four ids, so a bundle without R-strict is rejected. `DENSITY_NOISE` puts every channel on qubits 0 and 1. Strict runtime refuses a partition with no noise and a motif spanning more than two qubits. Probe: width 4 at cap 2 fails `channel_native_noise_presence`; cap 3 and cap 4 fail `channel_native_qubit_span`; widths 6 and 8 at cap 2 fail `pure_unitary_partition`. R-strict cannot run this anchor at any partition cap or at 4, 6, or 8.

**Why the probe missed it:** The code-ready check called `build_phase3_continuity_partition_descriptor_set` on the unpacked width-4 VQE and stopped at the surface counts 18 / 12 / 9 / 3. It never called a route. The descriptor succeeds. The runtime refuses.

**Question:** Which of A, B, or C does the Research Manager pick? The Research Manager owns the requirement and the workload. **HOLD any R-strict “fix” until that pick.** The Developer must not invent a quiet fallback, a stub timing row, or a dropped R-strict id.

- **A — recommended for this re-fix only.** Keep the raise. Document the refuse reason. R-strict rows stay out of the re-fix’s timing work. Continue R-base, R-fused, and R-hybrid only. Leave the four-id schema in place, so a three-row bundle still fails. REQ-004 stays open. No new workload and no runtime edit.
- **B — workload change.** Put a noise operation on every partition of this anchor, including qubits that are unitary-only today (width 4: qubits 2 and 3). Do that as a route-only schedule. Do not edit the counted E-VQE `DENSITY_NOISE`. Raising `max_partition_qubits` does not fix it (caps 3 and 4 still refuse). This is a second workload under ADR-F5A-006. CAP-004 stays hold-the-line unless the change is a channel-native runtime edit, which this option does not authorize.
- **C — escalate only.** Loosen ADR-F5A-006 (allow a second workload), REQ-004 (drop R-strict or the same-workload rule), or REQ-005 / CAP-004 (allow a channel-native runtime edit). Not a Developer choice.

**Authority:** Research Manager. **This handback does not re-issue code-ready for a four-route timed bundle.**

## Question 2 — apply timing (disposed)

**Contract:** REQ-004 and ADR-F5A-001: R-base’s apply component is C++ `NoisyCircuit.apply_to`. Task-1 §7: numerator is the apply sub-time, divisor `operation_count * 4^4` = 3072. \(T_\mathrm{public}\) and \(T_\mathrm{lower}\) are not the numerator. Task-1 F-4: split orchestration from apply on the harness side; a runtime edit hands back.

**What Step 4b found:** `attribution_route_lane.py:96–99` sets apply from `result.runtime_ms` (clock around the whole partition loop, `noisy_runtime_core.py:842–963`). Reviewer probe, 300 calls: R-base lane apply 288,271 ns versus 61,195 ns inside `apply_to` (4.87×, 8 calls, 79 % non-apply). Published throughput would be about 94–105 ns/entry/op. Committed E-VQE `apply_to` on the same anchor and divisor is 13.0 ns/entry/op (`212f7038…`).

**Planner pick, firm:** harness-side wrap of the apply primitives only. No `squander/**` edit. No `runtime_ms` as apply. No RM hold on this pick: it implements REQ-004, it does not redefine it.

For each sample, wrap and then restore, in a `finally`:

- R-base: `NoisyCircuit.apply_to` only.
- R-fused: `NoisyCircuit.apply_to` and `DensityMatrix.apply_local_unitary`.
- R-hybrid: `noisy_runtime_channel_native._apply_kraus_bundle` and `DensityMatrix.apply_local_unitary`.
- R-strict: no wrap. The call still raises.

`_member_to_kraus_bundle` and `_compose_kraus_bundles` are not apply. Apply nanoseconds are the sum of the wrapped calls. Orchestration nanoseconds are the public-entry wall minus that sum. Throughput is apply nanoseconds / 3072. The one-sided bound stays on orchestration and on apply only. S-g is not retuned. A test must show apply strictly below the runtime body and a primitive-call count above zero on every executed route (mutants L1 and L4).

## B2 and B3 (Developer, no new design)

**B2.** `attribution_route_lane.py:58` stores the constant `"the executed class"`. Derive R-hybrid’s label from `result.partitions[*].partition_runtime_class`. On this anchor that is numpy Kraus on partitions 0 and 2 (`phase31_channel_native`) and C++ `apply_local_unitary` on partitions 1, 3, and 4 (`phase3_unitary_island_fused`). The validator requires that record.

**B3.** Add negatives the mutation run left alive (8 of 26 killed): an R-oracle row in the bundle validator, with and without the E1 sentence; `t_lower_ns` and `upper_bound_95_O`; a divisor other than 3072; `milestone_counted` true; a three-row bundle with R-strict removed. Do not make the three-row bundle lawful.

## Developer re-fix

Authorized now, inside the existing allowlist, under Option A: B1 as picked above, B2, and B3. Leave the R-strict raise unchanged. Do not add a fallback, a stub row, or a route that drops R-strict. Do not edit `squander/**` or the three counted bundles. Do not run a counted (c). If the Research Manager later picks B or C, stop and wait. That pick is not authorized here.

## Stop-conditions

No ADR was amended. No stage line was flipped. The C0 stamp remains `226ffc29`. Step 4b code stays the Developer’s uncommitted tree.
