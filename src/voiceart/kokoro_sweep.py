"""Shared Kokoro synthesis support for adversarial ASR corpora."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

KOKORO_SAMPLE_RATE = 24_000
DEFAULT_VOICES: tuple[str, ...] = ("af_heart", "af_bella", "am_adam")
DEFAULT_RATES: dict[str, float] = {"slow": 0.80, "normal": 1.00, "fast": 1.35}


class SynthesisError(Exception):
    """Raised when the local Kokoro sweep cannot be synthesized."""


def synthesize_kokoro_sweep(
    output_dir: Path,
    *,
    attack: str,
    phrases: Sequence[Mapping[str, str]],
    voices: Sequence[str] = DEFAULT_VOICES,
    rates: Mapping[str, float] = DEFAULT_RATES,
    language_code: str = "a",
) -> Path:
    """Synthesize every phrase, voice, and speaking-rate combination."""
    try:
        import numpy as np
        import soundfile as sf  # type: ignore[import-not-found]
        from kokoro import KPipeline  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SynthesisError(
            "Kokoro synthesis dependencies are missing. Install with: "
            "pip install 'voiceart-benchmark[kokoro]'"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code=language_code)
    samples: list[dict[str, object]] = []

    for phrase in phrases:
        phrase_id = phrase["id"]
        text = phrase["text"]
        expected = {
            key: value for key, value in phrase.items() if key.startswith("expected_")
        }
        for voice in voices:
            for rate_name, speed in rates.items():
                parts = [
                    audio
                    for _graphemes, _phonemes, audio in pipeline(
                        text,
                        voice=voice,
                        speed=speed,
                    )
                ]
                if not parts:
                    raise SynthesisError(
                        f"Kokoro produced no audio for {phrase_id}/{voice}/{rate_name}"
                    )

                filename = _sample_filename(phrase_id, voice, rate_name, speed)
                sf.write(
                    output_dir / filename,
                    np.concatenate(parts),
                    KOKORO_SAMPLE_RATE,
                    subtype="PCM_16",
                )
                samples.append(
                    {
                        "file": filename,
                        "phrase_id": phrase_id,
                        "text": text,
                        **expected,
                        "voice": voice,
                        "rate": rate_name,
                        "speed": speed,
                    }
                )

    manifest = {
        "attack": attack,
        "sample_rate_hz": KOKORO_SAMPLE_RATE,
        "phrases": [dict(phrase) for phrase in phrases],
        "voices": list(voices),
        "rates": dict(rates),
        "language_code": language_code,
        "sample_count": len(samples),
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def sweep_sample_count(
    phrases: Iterable[object],
    voices: Iterable[str],
    rates: Mapping[str, float],
) -> int:
    """Return the Cartesian-product size for a configured sweep."""
    return sum(1 for _ in phrases) * sum(1 for _ in voices) * len(rates)


def _sample_filename(
    phrase_id: str,
    voice: str,
    rate_name: str,
    speed: float,
) -> str:
    """Build a stable, sortable WAV filename for one sweep sample."""
    speed_slug = f"{speed:.2f}".replace(".", "p")
    return f"{phrase_id}__{voice}__{rate_name}-{speed_slug}.wav"
