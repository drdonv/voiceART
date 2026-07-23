"""Spanish numeric STT benchmark for Deepgram Nova-3 batch transcription."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

from voiceart.numeric_benchmark import (
    DeepgramTranscriber,
    TestUtterance,
    Transcriber,
    benchmark_numeric_utterances,
    extract_amount,
    measure_numeric_robustness,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SPANISH_VOICES: tuple[str, ...] = ("ef_dora", "em_alex")
SPANISH_RATES: dict[str, float] = {"normal": 1.0, "fast": 1.25}

SPANISH_UTTERANCES: tuple[TestUtterance, ...] = (
    TestUtterance(
        "es-150-words",
        "Por favor, transfiere ciento cincuenta dólares al beneficiario de prueba.",
        "150",
    ),
    TestUtterance(
        "es-1200-words",
        "Por favor, transfiere mil doscientos dólares al beneficiario de prueba.",
        "1200",
    ),
    TestUtterance(
        "es-2550-words",
        "Transfiere dos mil quinientos cincuenta dólares al beneficiario de prueba.",
        "2550",
    ),
    TestUtterance(
        "es-2550-paired",
        "Transfiere veinticinco cincuenta dólares al beneficiario de prueba.",
        "2550",
    ),
    TestUtterance(
        "es-2550-separated",
        "Transfiere dos cinco cinco cero dólares al beneficiario de prueba.",
        "2550",
    ),
    TestUtterance(
        "es-42-words",
        "Envía cuarenta y dos dólares al beneficiario de prueba.",
        "42",
    ),
    TestUtterance(
        "es-75-words",
        "Envía setenta y cinco dólares al beneficiario de prueba.",
        "75",
    ),
    TestUtterance(
        "es-520-words",
        "Transfiere quinientos veinte dólares al beneficiario de prueba.",
        "520",
    ),
    TestUtterance(
        "es-1500-words",
        "Transfiere mil quinientos dólares al beneficiario de prueba.",
        "1500",
    ),
    TestUtterance(
        "es-9999-words",
        "Transfiere nueve mil novecientos noventa y nueve dólares al beneficiario de prueba.",
        "9999",
    ),
)

_SPANISH_UNITS = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
}
_SPANISH_SMALL = {
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}
_SPANISH_TENS = {
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}
_SPANISH_HUNDREDS = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "trescientos": 300,
    "cuatrocientos": 400,
    "quinientos": 500,
    "seiscientos": 600,
    "setecientos": 700,
    "ochocientos": 800,
    "novecientos": 900,
}
_SPANISH_KNOWN = (
    set(_SPANISH_UNITS)
    | set(_SPANISH_SMALL)
    | set(_SPANISH_TENS)
    | set(_SPANISH_HUNDREDS)
    | {"y", "mil"}
)


def extract_spanish_amount(transcript: str) -> str | None:
    """Extract digit or Spanish number-word monetary amounts."""
    digit_candidates = [
        re.sub(r"\D", "", match.group())
        for match in re.finditer(r"\d(?:[\d,\s]*\d)*", transcript)
    ]
    if digit_candidates:
        return max(digit_candidates, key=len)

    normalized = _strip_accents(transcript.lower().replace("-", " "))
    tokens = re.findall(r"[a-z]+", normalized)
    candidates: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] not in _SPANISH_KNOWN:
            index += 1
            continue
        end = index
        while end < len(tokens) and tokens[end] in _SPANISH_KNOWN:
            end += 1
        parsed = _parse_spanish_tokens(tokens[index:end])
        if parsed is not None:
            candidates.append(parsed)
        index = end
    if not candidates:
        return extract_amount(transcript)
    return max(candidates, key=len)


def run_spanish_numeric_benchmark(
    *,
    sweep_dir: Path = Path("output/kokoro/spanish-numeric"),
    output_root: Path = Path("output/stt-numeric-spanish/deepgram-nova3"),
    high_error_dir: Path = Path("fixtures/high_error/spanish/deepgram-nova3"),
    cache_root: Path = Path("output/kokoro-cache/spanish-numeric"),
    repeats: int = 4,
    threshold: float = 0.5,
    transcriber: Transcriber | None = None,
    utterances: Sequence[TestUtterance] = SPANISH_UTTERANCES,
    voices: Sequence[str] = SPANISH_VOICES,
    rates: Mapping[str, float] = SPANISH_RATES,
) -> Path:
    """Synthesize exactly 40 Spanish clips, then score Nova-3 four times each."""
    benchmark_numeric_utterances(
        sweep_dir,
        utterances=utterances,
        voices=voices,
        rates=rates,
        cache_root=cache_root,
        language_code="e",
    )
    return measure_numeric_robustness(
        sweep_dir,
        transcriber=transcriber
        or DeepgramTranscriber(model_name="nova-3", language="es", numerals=True),
        repeats=repeats,
        threshold=threshold,
        output_root=output_root,
        high_error_dir=high_error_dir,
        amount_extractor=extract_spanish_amount,
    )


def _strip_accents(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )


def _parse_spanish_tokens(tokens: list[str]) -> str | None:
    cleaned = [token for token in tokens if token != "y"]
    if not cleaned:
        return None
    if "mil" in cleaned or any(token in _SPANISH_HUNDREDS for token in cleaned):
        return _parse_standard_spanish(cleaned)
    return _parse_compact_spanish(cleaned)


def _parse_compact_spanish(tokens: list[str]) -> str | None:
    parts: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _SPANISH_TENS and index + 1 < len(tokens):
            following = tokens[index + 1]
            if following in _SPANISH_UNITS:
                parts.append(str(_SPANISH_TENS[token] + _SPANISH_UNITS[following]))
                index += 2
                continue
        value = _spanish_simple_value(token)
        if value is None:
            return None
        parts.append(str(value))
        index += 1
    return "".join(parts) if parts else None


def _parse_standard_spanish(tokens: list[str]) -> str | None:
    total = 0
    current = 0
    saw_value = False
    for token in tokens:
        if token in _SPANISH_HUNDREDS:
            current += _SPANISH_HUNDREDS[token]
        elif token == "mil":
            total += max(1, current) * 1000
            current = 0
        else:
            value = _spanish_simple_value(token)
            if value is None:
                return None
            current += value
        saw_value = True
    return str(total + current) if saw_value else None


def _spanish_simple_value(token: str) -> int | None:
    for values in (_SPANISH_UNITS, _SPANISH_SMALL, _SPANISH_TENS):
        if token in values:
            return values[token]
    return None
