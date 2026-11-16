# Local validation and versioning

The rewrite starts at **0.1.0**. Root `VERSION` drives the CUDA package's CMake
version. Contract schema v1 is a separate compatibility identifier. Frozen
upstream package/ROS metadata keeps its historical version; it is not the
rewrite release identifier. All validation runs in the existing Compose dev
container. No GitHub workflows, hosted runners, installed hooks, or new services.

## Commands

Start the container with `docker compose up -d --build` when needed. From the
repository root in PowerShell:

```powershell
./scripts/check.ps1 fast
./scripts/check.ps1 verify
```

Equivalent from another shell:

```sh
docker compose exec -T dev python3 scripts/local_ci.py verify
```

| Profile | Coverage | Use |
| --- | --- | --- |
| fast | Version/oracle guard, harness failure tests, host and GPU contracts | Routine CUDA/header edits |
| verify | fast plus CPU parity and Python binding reset/step | Before development checkpoints/commits |
| release | verify plus clean committed tree and unused tag requirement | Before release tagging |
| legacy | Original headless upstream suite | Upstream/dependency changes |
| doctor | Compiler/GPU information and CUDA installation smoke | Container/driver/toolkit changes |
| sanitize | fast plus Compute Sanitizer memcheck | Device-memory changes when host tooling supports it |

Every profile checks oracle integrity and records source/toolchain/device identity.
Only verify/release are full normal gates. The WSL/WDDM host currently cannot
initialize the sanitizer debugger interface; sanitize fails rather than skips.
Add applicable racecheck/synccheck workloads as kernels arrive and host support
becomes available. Do not make unavailable tooling a silently passing gate.

`docker/check.sh` delegates to verify. Existing focused check scripts and CTest
filters remain debugging tools, not full validation evidence. Do not run them
concurrently with the harness against shared build trees. Builds are incremental
in /build; verify builds flightgym against the same CPU oracle library, avoiding
a duplicate CPU build. No automatic changed-file selection is needed yet: the
useful suites are small, and missing dependencies would weaken validation.

## Reports and failure handling

Each run saves a unique ignored `.local-ci/` directory with `report.json`,
per-stage logs, and CTest/pytest JUnit XML. Reports record profile, version,
commit, dirty status, source fingerprint, toolchain/GPU identity, commands,
durations, and results. A report validates its recorded contents, not later edits.
No latest-pass marker can hide a newer failure. Reports are not committed.

Fingerprints include tracked and non-ignored untracked files, normalizing CRLF
to LF for text. A second fingerprint check fails if sources changed during
validation. Dirty development trees are allowed and recorded; release needs a
clean commit. An advisory /build lock prevents concurrent harness runs from
reconfiguring shared caches and releases automatically on exit.

The first failure stops validation with nonzero exit. Failed builds never run
stale binaries. Commands/tests have timeouts; CTest discovering zero tests is an
error. Missing GPU/toolchain and sanitizer initialization errors are failures.

## Test selection policy

Every added test must identify an observable failure and an assertion that
detects it. Prefer a small boundary fixture over declarations repeated as tests.

| Existing coverage | Decision |
| --- | --- |
| CPU parity suite | verify: rejection, clamps, motors, integration, bounds, rewards, reset routing |
| Host contract test | fast: multi-slot indexing, 64-bit offsets, overflow rejection, const reads |
| GPU contract test | fast: real device pointers/indexing, guarded tail lanes |
| Python smoke | verify: native loading, binding layouts, actual state advancement |
| Upstream logging/Eigen demonstrations | legacy only; printing logs/versions is not a behavioral assertion |
| CUDA installation smoke | doctor only; routine device contracts already prove kernel launch |
| Harness tests | fast: process failures/timeouts, oracle tampering, untracked source fingerprints |

The CPU suite validates the **CPU reference**, not CPU/GPU dynamics parity.
As kernels arrive, run the same oracle inputs through CUDA and compare complete
state, motors, time, observations, reward, and terminal/reset behavior in verify.
Independently passing CPU and GPU suites does not establish parity.

Add regression tests for reproduced bugs, lane/reset isolation, remainder
substeps, batch tails, and stream ownership as those implementations arrive.
Geometry needs independent fixtures; headless tests cannot certify Unity mesh
collisions. Avoid constructor-only tests, output-printing tests, repeated static
assertions, expected results computed by the implementation under test, and
noisy one-shot performance assertions. Docs-only edits need review/link checks,
not new simulator tests. Keep frozen upstream tests intact for investigation.

No performance thresholds exist before dynamics kernels. At the first kernel,
add a separate benchmark command with warmup, repeated samples, fixed E/A/query
workloads, synchronization at measurement boundaries, and machine-specific
baselines. Save wall time, VRAM, transfer, and synchronization evidence. Review
threshold changes. Do not add a speculative benchmark matrix to routine checks.

## Oracle integrity

`scripts/oracle-baseline.json` records normalized hashes of Flightlib source,
headers, configuration, CMake definition, and parity tests. Modified, added, or
deleted files in those areas fail validation. The manifest records its reference
commit and is never regenerated by the harness.

Intentional oracle/test expansion requires reviewing the diff, explaining it in
the changelog, and updating only affected manifest entries. Never blindly
regenerate it to make a check green, or change expected behavior to accommodate
unexplained GPU drift. Differential tests need justified field-specific
tolerances. This is a review guard, not tamper-proof security: both manifest and
source can be edited. Harness failure tests use isolated temporary repositories.

## Version checkpoints

Keep VERSION unchanged during ordinary commits. For a checkpoint:

1. Update VERSION and CHANGELOG together. Patches cover compatible fixes; minor
   releases cover capabilities or pre-1.0 breaking changes. Breaking tensor
   semantics also require a new contract version/namespace and migration notes.
2. Run verify, inspect changes, and commit the intended checkpoint locally.
3. Run `./scripts/check.ps1 release` on the clean commit.
4. After success, create an annotated local tag at the exact reported commit:
   `git tag -a v0.1.0 <reported-commit> -m "Flightlib CUDA 0.1.0"`.

Never move an existing tag; use a new version for a new fix. Historical tags
remain untouched. The harness never commits, tags, or pushes automatically.
The current uncommitted baseline is VERSION=0.1.0; tag it after the intended
files are committed and release validation passes. No binary ABI promise is
implied. When editing the harness itself, run fast and verify.
