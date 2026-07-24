"""Deepgram Nova-3 multilingual numeric benchmark."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from voiceart.numeric_benchmark import (
    DeepgramTranscriber,
    TestUtterance,
    measure_numeric_robustness,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


AMOUNTS: tuple[str, ...] = ("150", "1200", "2550", "42", "75", "520", "1500", "9999", "306", "18")
RATES: dict[str, int] = {"slow": 150, "normal": 180, "fast": 210, "very_fast": 240}


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    """One Deepgram-supported language and its native local voice."""

    code: str
    voice: str
    template: str
    number_words: Mapping[str, str]


LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        "en",
        "Samantha",
        "Transfer {amount} dollars to the test payee.",
        {
            "150": "one hundred fifty",
            "1200": "one thousand two hundred",
            "2550": "two thousand five hundred fifty",
            "42": "forty two",
            "75": "seventy five",
            "520": "five hundred twenty",
            "1500": "one thousand five hundred",
            "9999": "nine thousand nine hundred ninety nine",
            "306": "three hundred six",
            "18": "eighteen",
        },
    ),
    LanguageSpec(
        "es",
        "Mónica",
        "Transfiere {amount} dólares al beneficiario de prueba.",
        {
            "150": "ciento cincuenta",
            "1200": "mil doscientos",
            "2550": "dos mil quinientos cincuenta",
            "42": "cuarenta y dos",
            "75": "setenta y cinco",
            "520": "quinientos veinte",
            "1500": "mil quinientos",
            "9999": "nueve mil novecientos noventa y nueve",
            "306": "trescientos seis",
            "18": "dieciocho",
        },
    ),
    LanguageSpec(
        "fr",
        "Jacques",
        "Transfère {amount} euros au bénéficiaire de test.",
        {
            "150": "cent cinquante",
            "1200": "mille deux cents",
            "2550": "deux mille cinq cent cinquante",
            "42": "quarante-deux",
            "75": "soixante-quinze",
            "520": "cinq cent vingt",
            "1500": "mille cinq cents",
            "9999": "neuf mille neuf cent quatre-vingt-dix-neuf",
            "306": "trois cent six",
            "18": "dix-huit",
        },
    ),
    LanguageSpec(
        "de",
        "Anna",
        "Überweise {amount} Euro an den Testempfänger.",
        {
            "150": "einhundertfünfzig",
            "1200": "eintausendzweihundert",
            "2550": "zweitausendfünfhundertfünfzig",
            "42": "zweiundvierzig",
            "75": "fünfundsiebzig",
            "520": "fünfhundertzwanzig",
            "1500": "eintausendfünfhundert",
            "9999": "neuntausendneunhundertneunundneunzig",
            "306": "dreihundertsechs",
            "18": "achtzehn",
        },
    ),
    LanguageSpec(
        "hi",
        "Lekha",
        "परीक्षण प्राप्तकर्ता को {amount} रुपये भेजें।",
        {
            "150": "एक सौ पचास",
            "1200": "बारह सौ",
            "2550": "दो हजार पांच सौ पचास",
            "42": "बयालीस",
            "75": "पचहत्तर",
            "520": "पांच सौ बीस",
            "1500": "पंद्रह सौ",
            "9999": "नौ हजार नौ सौ निन्यानबे",
            "306": "तीन सौ छह",
            "18": "अठारह",
        },
    ),
    LanguageSpec(
        "ru",
        "Milena",
        "Переведи {amount} рублей тестовому получателю.",
        {
            "150": "сто пятьдесят",
            "1200": "тысяча двести",
            "2550": "две тысячи пятьсот пятьдесят",
            "42": "сорок два",
            "75": "семьдесят пять",
            "520": "пятьсот двадцать",
            "1500": "тысяча пятьсот",
            "9999": "девять тысяч девятьсот девяносто девять",
            "306": "триста шесть",
            "18": "восемнадцать",
        },
    ),
    LanguageSpec(
        "pt",
        "Luciana",
        "Transfira {amount} reais para o beneficiário de teste.",
        {
            "150": "cento e cinquenta",
            "1200": "mil e duzentos",
            "2550": "dois mil quinhentos e cinquenta",
            "42": "quarenta e dois",
            "75": "setenta e cinco",
            "520": "quinhentos e vinte",
            "1500": "mil e quinhentos",
            "9999": "nove mil novecentos e noventa e nove",
            "306": "trezentos e seis",
            "18": "dezoito",
        },
    ),
    LanguageSpec(
        "ja",
        "Kyoko",
        "テスト受取人に{amount}円を送金してください。",
        {
            "150": "百五十",
            "1200": "千二百",
            "2550": "二千五百五十",
            "42": "四十二",
            "75": "七十五",
            "520": "五百二十",
            "1500": "千五百",
            "9999": "九千九百九十九",
            "306": "三百六",
            "18": "十八",
        },
    ),
    LanguageSpec(
        "it",
        "Alice",
        "Trasferisci {amount} euro al beneficiario di prova.",
        {
            "150": "centocinquanta",
            "1200": "milleduecento",
            "2550": "duemilacinquecentocinquanta",
            "42": "quarantadue",
            "75": "settantacinque",
            "520": "cinquecentoventi",
            "1500": "millecinquecento",
            "9999": "novemilanovecentonovantanove",
            "306": "trecentosei",
            "18": "diciotto",
        },
    ),
    LanguageSpec(
        "nl",
        "Xander",
        "Maak {amount} euro over naar de testbegunstigde.",
        {
            "150": "honderdvijftig",
            "1200": "twaalfhonderd",
            "2550": "tweeduizend vijfhonderdvijftig",
            "42": "tweeënveertig",
            "75": "vijfenzeventig",
            "520": "vijfhonderdtwintig",
            "1500": "vijftienhonderd",
            "9999": "negen duizend negenhonderd negenennegentig",
            "306": "driehonderdzes",
            "18": "achttien",
        },
    ),
)


def build_language_utterances(spec: LanguageSpec) -> tuple[TestUtterance, ...]:
    """Build ten reproducible amount cases for one language."""
    return tuple(
        TestUtterance(
            id=f"{spec.code}-{amount}",
            text=spec.template.format(amount=spec.number_words[amount]),
            spoken_amount=amount,
        )
        for amount in AMOUNTS
    )


def synthesize_multilingual_sweep(
    output_dir: Path,
    *,
    specs: tuple[LanguageSpec, ...] = LANGUAGES,
    rates: Mapping[str, int] = RATES,
) -> Path:
    """Create 40 native-voice WAV clips for each supported language."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = output_dir / "manifest.json"
    expected_count = len(specs) * len(AMOUNTS) * len(rates)
    if existing_manifest.exists():
        cached = json.loads(existing_manifest.read_text())
        cached_samples = cached.get("samples", [])
        if (
            cached.get("sample_count") == expected_count
            and len(cached_samples) == expected_count
            and cached.get("languages") == [s.code for s in specs]
            and cached.get("rates") == dict(rates)
            and all((output_dir / str(sample["file"])).is_file() for sample in cached_samples)
        ):
            return existing_manifest
    samples: list[dict[str, object]] = []
    for spec in specs:
        for utterance in build_language_utterances(spec):
            for rate_name, words_per_minute in rates.items():
                stem = f"{utterance.id}__{rate_name}"
                aiff_path = output_dir / f"{stem}.aiff"
                wav_path = output_dir / f"{stem}.wav"
                subprocess.run(
                    [
                        "say",
                        "-v",
                        spec.voice,
                        "-r",
                        str(words_per_minute),
                        "-o",
                        str(aiff_path),
                        utterance.text,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(aiff_path),
                        "-ar",
                        "24000",
                        "-ac",
                        "1",
                        str(wav_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                aiff_path.unlink()
                samples.append(
                    {
                        "file": wav_path.name,
                        "id": utterance.id,
                        "text": utterance.text,
                        "spoken_amount": utterance.spoken_amount,
                        "language": spec.code,
                        "voice": spec.voice,
                        "rate": rate_name,
                    }
                )
    manifest = {
        "attack": "multilingual-numeric-robustness",
        "languages": [spec.code for spec in specs],
        "rates": dict(rates),
        "samples_per_language": len(AMOUNTS) * len(rates),
        "sample_count": len(samples),
        "samples": samples,
        "source": "macOS say + ffmpeg",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


def extract_multilingual_amount(transcript: str, spec: LanguageSpec) -> str | None:
    """Extract digits or one of the known language-specific amount phrases."""
    digit_matches = [
        re.sub(r"\D", "", match.group()) for match in re.finditer(r"\d(?:[\d,\s]*\d)*", transcript)
    ]
    if digit_matches:
        return max(digit_matches, key=len)
    normalized = _normalize(transcript)
    for amount, phrase in sorted(
        spec.number_words.items(), key=lambda item: -len(_normalize(item[1]))
    ):
        if _normalize(phrase) in normalized:
            return amount
    return None


def run_multilingual_numeric_benchmark(
    *,
    sweep_root: Path = Path("output/kokoro/multilingual-numeric"),
    results_root: Path = Path("output/stt-numeric-multilingual/deepgram-nova3"),
    fixture_root: Path = Path("fixtures/high_error/multilingual/deepgram-nova3"),
    repeats: int = 4,
    threshold: float = 0.5,
    api_key: str | None = None,
) -> Path:
    """Generate all 400 clips and score each language four times."""
    manifest_path = synthesize_multilingual_sweep(sweep_root)
    manifest = json.loads(manifest_path.read_text())
    for spec in LANGUAGES:
        language_dir = sweep_root / spec.code
        # The shared manifest is flat; materialize per-language manifests for
        # the existing scorer without changing its Gate-1 contract.
        language_samples = [
            sample for sample in manifest["samples"] if sample["language"] == spec.code
        ]
        language_dir.mkdir(parents=True, exist_ok=True)
        language_manifest = language_dir / "manifest.json"
        language_manifest.write_text(json.dumps({"samples": language_samples}, ensure_ascii=False))
        for sample in language_samples:
            source = sweep_root / sample["file"]
            destination = language_dir / sample["file"]
            if destination.is_symlink():
                destination.unlink()
            elif destination.exists():
                continue
            destination.symlink_to(source.resolve())

        def language_extractor(
            transcript: str, selected: LanguageSpec = spec
        ) -> str | None:
            return extract_multilingual_amount(transcript, selected)

        measure_numeric_robustness(
            language_dir,
            transcriber=DeepgramTranscriber(
                api_key=api_key,
                model_name="nova-3",
                language=spec.code,
                numerals=True,
            ),
            repeats=repeats,
            threshold=threshold,
            output_root=results_root,
            high_error_dir=fixture_root,
            amount_extractor=language_extractor,
        )
    return results_root


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().strip())
