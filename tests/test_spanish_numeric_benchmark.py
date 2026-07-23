"""Tests for the Spanish Deepgram Nova-3 numeric benchmark."""

import json
from pathlib import Path

import pytest

from voiceart.spanish_numeric_benchmark import (
    SPANISH_RATES,
    SPANISH_UTTERANCES,
    SPANISH_VOICES,
    extract_spanish_amount,
    run_spanish_numeric_benchmark,
)


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("ciento cincuenta dólares", "150"),
        ("mil doscientos dólares", "1200"),
        ("dos mil quinientos cincuenta", "2550"),
        ("veinticinco cincuenta", "2550"),
        ("dos cinco cinco cero", "2550"),
        ("cuarenta y dos", "42"),
        ("setenta y cinco", "75"),
        ("quinientos veinte", "520"),
        ("nueve mil novecientos noventa y nueve", "9999"),
        ("La cantidad es $2,550.", "2550"),
        ("Envía 42 dólares al beneficiario de prueba.", "42"),
        ("Envía 75 dólares al beneficiario de prueba.", "75"),
    ],
)
def test_extract_spanish_amount(transcript: str, expected: str) -> None:
    """Spanish words, compact forms, accents, and digits should normalize."""
    assert extract_spanish_amount(transcript) == expected


def test_spanish_suite_has_exactly_40_clips_and_four_repeats() -> None:
    """Ten utterances × two voices × two rates yields the requested 40 cases."""
    assert len(SPANISH_UTTERANCES) * len(SPANISH_VOICES) * len(SPANISH_RATES) == 40


def test_spanish_runner_uses_language_and_extractor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner should synthesize Spanish and score with four repeats."""
    benchmark_calls: list[dict[str, object]] = []
    measure_calls: list[dict[str, object]] = []

    class DummyTranscriber:
        def transcribe(self, wav_path: Path) -> str:
            return wav_path.name

    def fake_benchmark(output_dir: Path, **kwargs: object) -> Path:
        benchmark_calls.append(kwargs)
        output_dir.mkdir(parents=True)
        manifest = output_dir / "manifest.json"
        manifest.write_text(json.dumps({"samples": []}))
        return manifest

    def fake_measure(sweep_dir: Path, **kwargs: object) -> Path:
        measure_calls.append(kwargs)
        result = tmp_path / "results" / sweep_dir.name
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        "voiceart.spanish_numeric_benchmark.benchmark_numeric_utterances", fake_benchmark
    )
    monkeypatch.setattr(
        "voiceart.spanish_numeric_benchmark.measure_numeric_robustness", fake_measure
    )

    result = run_spanish_numeric_benchmark(
        sweep_dir=tmp_path / "sweep", transcriber=DummyTranscriber()
    )

    assert result.name == "sweep"
    assert benchmark_calls[0]["language_code"] == "e"
    assert measure_calls[0]["repeats"] == 4
    assert measure_calls[0]["amount_extractor"] is extract_spanish_amount
