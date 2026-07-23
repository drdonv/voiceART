# VoiceART

VoiceART is a reproducible speech-to-text robustness benchmark for voice
providers. It submits byte-identical audio to multiple STT systems, retains
every raw transcript, normalizes the target amount, and generates an auditable
provider comparison.

This repository evaluates STT providers. It does not call voice agents or
evaluate downstream agent behavior.

## Canonical English results

The canonical run uses one locked corpus of 90 Kokoro WAVs and four independent
transcriptions per clip.

| Provider | Incorrect observations | Failure rate | Affected clips |
|---|---:|---:|---:|
| Deepgram Nova-3 | 0/360 | 0.00% | 0/90 |
| Cartesia Ink | 0/360 | 0.00% | 0/90 |
| ElevenLabs Scribe v2 | 7/360 | 1.94% | 3/90 |

All seven verified failures collapsed `2550` to `25` at fast speech rates.
The raw transcripts, exact repeats, result hashes, and provider settings are in
[`benchmark/provider-comparison.md`](benchmark/provider-comparison.md).

## Evidence and reproducibility

- `corpus/numeric/` contains all 90 WAVs and the source manifest.
- `benchmark/numeric-corpus-lock.json` records every WAV SHA-256, the manifest
  SHA-256, and an aggregate corpus digest.
- `benchmark/runs/` contains the canonical raw JSONL and summaries.
- `scripts/generate_provider_comparison.py` verifies the corpus and result
  evidence before regenerating the comparison.
- `superseded/` retains older incompatible or incorrectly scored runs.

Verify the evidence and regenerate the comparison:

```console
python scripts/generate_provider_comparison.py
```

The locked corpus aggregate is:

```text
1391fc5c6090e310598d92248c74990054a819ab26f7ead3534aaf5426ec9cd1
```

## Installation

Python 3.11 or newer is required.

```console
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Provider credentials are read from environment variables:

```console
export DEEPGRAM_API_KEY=...
export CARTESIA_API_KEY=...
export ELEVENLABS_API_KEY=...
export OPENAI_API_KEY=...
```

Keys are never serialized into result artifacts.

## Running the English benchmark

The included locked corpus is reused unless its manifest or files no longer
match the configured 10-phrase × 3-voice × 3-rate grid.

```console
voiceart numeric --backend deepgram --repeats 4
voiceart numeric --backend cartesia --repeats 4
voiceart numeric --backend elevenlabs --repeats 4
voiceart numeric --backend openai --repeats 4
```

Additional adapters are implemented for local faster-whisper, AssemblyAI, Groq,
OpenAI GPT-4o Transcribe, and Deepgram Flux.

## Additional suites

The package also contains:

- A 40-clip Spanish numeric suite using Kokoro Spanish voices.
- An experimental ten-language suite using native macOS voices and FFmpeg.

These suites are separate from the canonical English provider comparison.

## Scoring

Each raw transcript is normalized to an amount. Both word and digit forms are
supported, including:

- `two thousand five hundred fifty`
- `twenty five fifty`
- `two five five zero`
- `2 550`
- `two 550`

An observation fails only when the normalized amount differs from that clip's
declared `spoken_amount`.

## License

Apache-2.0.
