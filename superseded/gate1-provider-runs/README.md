# Superseded Gate 1 provider runs

These artifacts are retained as historical evidence and are not current
benchmark results.

They were superseded on 2026-07-23 by fresh four-repeat runs over the locked
90-clip numeric corpus recorded in
[`benchmark/numeric-corpus-lock.json`](../../benchmark/numeric-corpus-lock.json).

## Why each run was superseded

- `deepgram-5x/` used five repeats (450 observations), lacked raw transcript
  strings, and was not directly comparable with the four-repeat provider runs.
- `cartesia-parser-bug-4x/` reported 36 failures because the old scorer parsed
  Cartesia's raw `two 550` transcript as `570`. The corrected scorer parses it
  as `2550`; a fresh run produced 0/360 failures.
- `elevenlabs-historical-4x/` is a valid historical run, but a fresh locked
  four-repeat run produced 7/360 failures rather than its earlier 5/360.
- `reports/` contains manually written or stale comparisons based on those
  superseded artifacts.
- `fixtures/cartesia-false-positives/` contains the nine WAVs copied as
  high-error fixtures solely because of the old Cartesia parser bug.

The current comparison is generated from raw fresh-run JSONL evidence by
`scripts/generate_provider_comparison.py`.
