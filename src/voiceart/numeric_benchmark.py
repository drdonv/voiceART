"""Local STT numeric-robustness benchmark for synthetic QA utterances."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from voiceart.kokoro_sweep import (
    DEFAULT_RATES,
    DEFAULT_VOICES,
    KOKORO_SAMPLE_RATE,
    synthesize_kokoro_sweep,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TestUtterance:
    """One controlled utterance in the numeric-robustness suite."""

    __test__ = False

    id: str
    text: str
    spoken_amount: str


@dataclass(frozen=True, slots=True)
class AmountCount:
    """Count of one extracted amount value."""

    amount: str | None
    count: int


@dataclass(frozen=True, slots=True)
class NumericBenchmarkRow:
    """One scored WAV clip from the numeric benchmark."""

    backend: str
    id: str
    text: str
    spoken_amount: str
    language: str | None
    voice: str
    rate: str
    wav_path: str
    repeats: int
    transcripts: tuple[str, ...]
    observations: tuple[str | None, ...]
    distribution: tuple[AmountCount, ...]
    error_rate: float
    most_common_misheard_amount: str | None
    is_high_error: bool


@runtime_checkable
class Transcriber(Protocol):
    """Pluggable STT backend interface."""

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV file to text."""


@dataclass(frozen=True, slots=True)
class FasterWhisperTranscriber:
    """Local faster-whisper STT backend."""

    model_name: str = "base.en"
    download_root: Path | None = None

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV file using faster-whisper."""
        model = _load_faster_whisper_model(self.model_name, self.download_root)
        segments, _info = model.transcribe(str(wav_path), language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()


@dataclass(frozen=True, slots=True)
class DeepgramTranscriber:
    """Deepgram STT backend used for vendor comparison."""

    api_key: str | None = None
    model_name: str = "nova-3"
    language: str = "en"
    punctuate: bool = True
    smart_format: bool = True
    numerals: bool = True

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV file using Deepgram."""
        api_key = self.api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY environment variable is not set")
        client = _load_deepgram_client(api_key)
        response = client.listen.v1.media.transcribe_file(
            request=wav_path.read_bytes(),
            model=self.model_name,
            language=self.language,
            punctuate=self.punctuate,
            smart_format=self.smart_format,
            numerals=self.numerals,
        )
        channels = response.results.channels
        if not channels:
            return ""
        alternatives = channels[0].alternatives
        if not alternatives:
            return ""
        return alternatives[0].transcript or ""


@dataclass(frozen=True, slots=True)
class DeepgramFluxTranscriber:
    """Deepgram Flux Multilingual streaming STT for voice-agent Gate 1 runs."""

    api_key: str | None = None
    model_name: str = "flux-general-multi"
    language_hint: str = "es"
    numerals: bool = True
    eot_timeout_ms: int = 1000
    endpoint: str = "wss://api.deepgram.com/v2/listen"
    pace_audio: bool = True

    def transcribe(self, wav_path: Path) -> str:
        """Stream one PCM WAV through Flux and return completed turn text."""
        key = _require_api_key(self.api_key, "DEEPGRAM_API_KEY", "Deepgram Flux")
        return asyncio.run(self._transcribe_async(wav_path, key))

    async def _transcribe_async(self, wav_path: Path, api_key: str) -> str:
        """Implement the Flux v2 WebSocket protocol with 80 ms audio chunks."""
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - dependency failure path
            raise RuntimeError(
                "websockets is not installed. Install with: "
                "pip install 'voiceart-benchmark[websockets]'"
            ) from exc

        audio, sample_rate, chunk_bytes = _read_flux_pcm(wav_path)
        query = urlencode(
            {
                "model": self.model_name,
                "encoding": "linear16",
                "sample_rate": sample_rate,
                "language_hint": self.language_hint,
                "numerals": str(self.numerals).lower(),
                "eot_timeout_ms": self.eot_timeout_ms,
            }
        )
        transcripts: list[str] = []
        end_of_turn = asyncio.Event()
        async with websockets.connect(
            f"{self.endpoint}?{query}",
            additional_headers={"Authorization": f"Token {api_key}"},
            open_timeout=30,
            close_timeout=10,
        ) as socket:
            async def receive_events() -> None:
                async for raw_message in socket:
                    if isinstance(raw_message, bytes):
                        continue
                    message = json.loads(raw_message)
                    if message.get("type") == "Error":
                        raise RuntimeError(
                            f"Deepgram Flux failed: {message.get('code')}: "
                            f"{message.get('description')}"
                        )
                    if (
                        message.get("type") == "TurnInfo"
                        and message.get("event") == "EndOfTurn"
                    ):
                        transcript = str(message.get("transcript") or "").strip()
                        if transcript:
                            transcripts.append(transcript)
                        end_of_turn.set()

            receiver = asyncio.create_task(receive_events())
            for offset in range(0, len(audio), chunk_bytes):
                chunk = audio[offset : offset + chunk_bytes]
                await socket.send(chunk)
                if self.pace_audio:
                    await asyncio.sleep(len(chunk) / (sample_rate * 2))
            # A short silence tail lets Flux's turn detector observe the natural
            # end of an isolated utterance before the stream is closed.
            silence = bytes(sample_rate * 2 * 2)
            for offset in range(0, len(silence), chunk_bytes):
                chunk = silence[offset : offset + chunk_bytes]
                await socket.send(chunk)
                if self.pace_audio:
                    await asyncio.sleep(len(chunk) / (sample_rate * 2))
            try:
                await asyncio.wait_for(end_of_turn.wait(), timeout=10)
            except TimeoutError:
                receiver.cancel()
                raise RuntimeError("Deepgram Flux produced no EndOfTurn event") from None
            await socket.send(json.dumps({"type": "CloseStream"}))
            await receiver
        return " ".join(transcripts)


@dataclass(frozen=True, slots=True)
class CartesiaTranscriber:
    """Cartesia Ink batch STT backend."""

    api_key: str | None = None
    model_name: str = "ink-whisper"
    api_version: str = "2026-03-01"

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV using Cartesia's batch STT endpoint."""
        key = _require_api_key(self.api_key, "CARTESIA_API_KEY", "Cartesia")
        payload = _post_multipart(
            "https://api.cartesia.ai/stt",
            headers={
                "Authorization": f"Bearer {key}",
                "Cartesia-Version": self.api_version,
            },
            fields={"model": self.model_name, "language": "en"},
            wav_path=wav_path,
        )
        return str(payload.get("text") or "")


@dataclass(frozen=True, slots=True)
class ElevenLabsTranscriber:
    """ElevenLabs Scribe batch STT backend."""

    api_key: str | None = None
    model_name: str = "scribe_v2"

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV using ElevenLabs Scribe."""
        key = _require_api_key(self.api_key, "ELEVENLABS_API_KEY", "ElevenLabs")
        payload = _post_multipart(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": key},
            fields={
                "model_id": self.model_name,
                "language_code": "eng",
                "tag_audio_events": "false",
                "diarize": "false",
            },
            wav_path=wav_path,
        )
        return str(payload.get("text") or "")


@dataclass(frozen=True, slots=True)
class OpenAITranscriber:
    """OpenAI GPT-4o batch transcription backend."""

    api_key: str | None = None
    model_name: str = "gpt-4o-transcribe"

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV using OpenAI's audio transcription endpoint."""
        key = _require_api_key(self.api_key, "OPENAI_API_KEY", "OpenAI")
        payload = _post_multipart(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            fields={
                "model": self.model_name,
                "language": "en",
                "temperature": "0",
                "response_format": "json",
            },
            wav_path=wav_path,
        )
        return str(payload.get("text") or "")


@dataclass(frozen=True, slots=True)
class GroqTranscriber:
    """Groq-hosted Whisper STT backend."""

    api_key: str | None = None
    model_name: str = "whisper-large-v3"

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV using Groq's OpenAI-compatible endpoint."""
        key = _require_api_key(self.api_key, "GROQ_API_KEY", "Groq")
        payload = _post_multipart(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            fields={
                "model": self.model_name,
                "language": "en",
                "temperature": "0",
                "response_format": "json",
            },
            wav_path=wav_path,
        )
        return str(payload.get("text") or "")


@dataclass(frozen=True, slots=True)
class AssemblyAITranscriber:
    """AssemblyAI asynchronous pre-recorded STT backend."""

    api_key: str | None = None
    speech_models: tuple[str, ...] = ("universal-3-pro", "universal-2")
    poll_interval: float = 0.5
    timeout: float = 180.0

    def transcribe(self, wav_path: Path) -> str:
        """Upload, submit, and poll one AssemblyAI transcription."""
        key = _require_api_key(self.api_key, "ASSEMBLYAI_API_KEY", "AssemblyAI")
        headers = {"Authorization": key}
        upload = _request_json(
            Request(
                "https://api.assemblyai.com/v2/upload",
                data=wav_path.read_bytes(),
                headers={**headers, "Content-Type": "application/octet-stream"},
                method="POST",
            )
        )
        upload_url = str(upload.get("upload_url") or "")
        if not upload_url:
            raise RuntimeError("AssemblyAI upload response contained no upload_url")
        submission = _request_json(
            Request(
                "https://api.assemblyai.com/v2/transcript",
                data=json.dumps(
                    {
                        "audio_url": upload_url,
                        "speech_models": list(self.speech_models),
                        "language_code": "en",
                        "format_text": True,
                    }
                ).encode(),
                headers={**headers, "Content-Type": "application/json"},
                method="POST",
            )
        )
        transcript_id = str(submission.get("id") or "")
        if not transcript_id:
            raise RuntimeError("AssemblyAI submission response contained no transcript id")

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = _request_json(
                Request(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers,
                )
            )
            status = str(result.get("status") or "")
            if status == "completed":
                return str(result.get("text") or "")
            if status == "error":
                raise RuntimeError(f"AssemblyAI transcription failed: {result.get('error')}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"AssemblyAI transcription timed out after {self.timeout}s")


DEFAULT_UTTERANCES: tuple[TestUtterance, ...] = (
    TestUtterance(
        id="numeric-150-digits",
        text="Please record 150 units for the QA fixture.",
        spoken_amount="150",
    ),
    TestUtterance(
        id="numeric-150-words",
        text="Please record one hundred fifty units for the QA fixture.",
        spoken_amount="150",
    ),
    TestUtterance(
        id="numeric-2550-digits",
        text="Please record 2550 units for the QA fixture.",
        spoken_amount="2550",
    ),
    TestUtterance(
        id="numeric-2550-words",
        text="Please record two thousand five hundred fifty units for the QA fixture.",
        spoken_amount="2550",
    ),
    TestUtterance(
        id="numeric-75-words",
        text="Please record seventy five units for the QA fixture.",
        spoken_amount="75",
    ),
    TestUtterance(
        id="numeric-42-words",
        text="Please record forty two units for the QA fixture.",
        spoken_amount="42",
    ),
    TestUtterance(
        id="numeric-1200-words",
        text="Please record one thousand two hundred units for the QA fixture.",
        spoken_amount="1200",
    ),
    TestUtterance(
        id="numeric-520-words",
        text="Please record five hundred twenty units for the QA fixture.",
        spoken_amount="520",
    ),
    TestUtterance(
        id="numeric-2550-spelled",
        text="Please record twenty five fifty units for the QA fixture.",
        spoken_amount="2550",
    ),
    TestUtterance(
        id="numeric-2550-separated",
        text="Please record two five five zero units for the QA fixture.",
        spoken_amount="2550",
    ),
)

_DIGIT_RUN = re.compile(r"\d(?:[\d,\s]*\d)*")
_TOKEN_RUN = re.compile(r"[a-z]+|\d+")
_CONNECTORS = {"and"}
_ALIASES = {"a": 1, "an": 1, "oh": 0, "o": 0, "zero": 0}
_SINGLE_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS_WORDS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALE_WORDS = {"hundred", "thousand", "million"}
_KNOWN_NUMBER_WORDS = set(_ALIASES) | set(_SINGLE_WORDS) | set(_TENS_WORDS) | _SCALE_WORDS


class _WhisperSegment(Protocol):
    """Protocol for faster-whisper segments."""

    text: str


class _WhisperModel(Protocol):
    """Protocol for the faster-whisper model object."""

    def transcribe(self, audio: str, *, language: str) -> tuple[Iterable[_WhisperSegment], object]:
        """Transcribe a WAV file."""


class _DeepgramAlternative(Protocol):
    """Protocol for a Deepgram transcript alternative."""

    transcript: str


class _DeepgramChannel(Protocol):
    """Protocol for a Deepgram result channel."""

    alternatives: list[_DeepgramAlternative]


class _DeepgramResults(Protocol):
    """Protocol for Deepgram's result wrapper."""

    channels: list[_DeepgramChannel]


class _DeepgramResponse(Protocol):
    """Protocol for Deepgram's response object."""

    results: _DeepgramResults


class _DeepgramMedia(Protocol):
    """Protocol for the Deepgram media endpoint."""

    def transcribe_file(
        self,
        *,
        request: bytes,
        model: str,
        language: str,
        punctuate: bool,
        smart_format: bool,
        numerals: bool,
    ) -> _DeepgramResponse:
        """Transcribe raw audio bytes."""


class _DeepgramV1(Protocol):
    """Protocol for the Deepgram v1 endpoint."""

    media: _DeepgramMedia


class _DeepgramListen(Protocol):
    """Protocol for the Deepgram listen endpoint."""

    v1: _DeepgramV1


class _DeepgramClient(Protocol):
    """Protocol for the Deepgram client used by the benchmark."""

    listen: _DeepgramListen


def benchmark_numeric_utterances(
    output_dir: Path,
    *,
    utterances: Sequence[TestUtterance] = DEFAULT_UTTERANCES,
    voices: Sequence[str] = DEFAULT_VOICES,
    rates: Mapping[str, float] = DEFAULT_RATES,
    cache_root: Path | None = None,
    language_code: str = "a",
) -> Path:
    """Synthesize the numeric benchmark sweep with exact-text caching."""
    cache_root = cache_root or Path("output/kokoro-cache/numeric")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, object]] = []
    for utterance in utterances:
        cache_dir = _ensure_cached_utterance(
            utterance, voices, rates, cache_root, language_code=language_code
        )
        text_hash = _text_hash(utterance.text)
        for voice in voices:
            for rate_name, speed in rates.items():
                cached_name = _sample_filename(text_hash, voice, rate_name, speed)
                cached_wav = cache_dir / cached_name
                final_name = _sample_filename(utterance.id, voice, rate_name)
                final_wav = output_dir / final_name
                shutil.copy2(cached_wav, final_wav)
                samples.append(
                    {
                        "file": final_name,
                        "id": utterance.id,
                        "text": utterance.text,
                        "spoken_amount": utterance.spoken_amount,
                        "voice": voice,
                        "rate": rate_name,
                        "speed": speed,
                        "text_hash": text_hash,
                    }
                )

    manifest = {
        "attack": "numeric-robustness-qa",
        "sample_count": len(samples),
        "sample_rate_hz": KOKORO_SAMPLE_RATE,
        "utterances": [asdict(utterance) for utterance in utterances],
        "voices": list(voices),
        "rates": dict(rates),
        "samples": samples,
        "cache_root": str(cache_root),
        "source": "kokoro",
        "language_code": language_code,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def measure_numeric_robustness(
    sweep_dir: Path,
    *,
    transcriber: Transcriber | None = None,
    repeats: int = 5,
    threshold: float = 0.5,
    output_root: Path | None = None,
    high_error_dir: Path | None = None,
    amount_extractor: Callable[[str], str | None] | None = None,
    workers: int = 1,
) -> Path:
    """Transcribe a sweep repeatedly and score amount extraction stability."""
    transcriber = transcriber or FasterWhisperTranscriber()
    amount_extractor = amount_extractor or extract_amount
    output_root = output_root or Path("output/stt-numeric")
    high_error_dir = high_error_dir or Path("fixtures/high_error")

    manifest_path = sweep_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {sweep_dir}")

    manifest = json.loads(manifest_path.read_text())
    def score(sample: dict[str, object]) -> NumericBenchmarkRow:
        return _score_sample(
            sweep_dir=sweep_dir,
            sample=sample,
            transcriber=transcriber,
            repeats=repeats,
            threshold=threshold,
            amount_extractor=amount_extractor,
        )

    samples = list(manifest["samples"])
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(score, samples))
    else:
        rows = [score(sample) for sample in samples]

    rows.sort(key=lambda row: (-row.error_rate, row.id, row.voice, row.rate))

    results_dir = output_root / sweep_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_row_to_dict(row)) + "\n")

    _copy_high_error_clips(rows, sweep_dir, high_error_dir)
    _write_summary(rows, repeats, threshold, results_dir)
    return results_dir


def run_numeric_benchmark(
    *,
    sweep_dir: Path = Path("corpus/numeric"),
    repeats: int = 5,
    threshold: float = 0.5,
    transcriber: Transcriber | None = None,
    output_root: Path | None = None,
    high_error_dir: Path | None = None,
    utterances: Sequence[TestUtterance] = DEFAULT_UTTERANCES,
    voices: Sequence[str] = DEFAULT_VOICES,
    rates: Mapping[str, float] = DEFAULT_RATES,
    cache_root: Path | None = None,
    language_code: str = "a",
    amount_extractor: Callable[[str], str | None] | None = None,
    workers: int = 1,
) -> Path:
    """Run the full synthesize → transcribe → measure pipeline."""
    if not _sweep_matches_configuration(
        sweep_dir,
        utterances=utterances,
        voices=voices,
        rates=rates,
        language_code=language_code,
    ):
        benchmark_numeric_utterances(
            sweep_dir,
            utterances=utterances,
            voices=voices,
            rates=rates,
            cache_root=cache_root,
            language_code=language_code,
        )
    return measure_numeric_robustness(
        sweep_dir,
        transcriber=transcriber,
        repeats=repeats,
        threshold=threshold,
        output_root=output_root,
        high_error_dir=high_error_dir,
        amount_extractor=amount_extractor,
        workers=workers,
    )


def _sweep_matches_configuration(
    sweep_dir: Path,
    *,
    utterances: Sequence[TestUtterance],
    voices: Sequence[str],
    rates: Mapping[str, float],
    language_code: str,
) -> bool:
    """Return whether an existing sweep is complete for the requested grid."""
    manifest_path = sweep_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples")
    expected_count = len(utterances) * len(voices) * len(rates)
    return (
        manifest.get("utterances") == [asdict(utterance) for utterance in utterances]
        and manifest.get("voices") == list(voices)
        and manifest.get("rates") == dict(rates)
        and manifest.get("language_code", "a") == language_code
        and manifest.get("sample_count") == expected_count
        and isinstance(samples, list)
        and len(samples) == expected_count
        and all((sweep_dir / str(sample["file"])).is_file() for sample in samples)
    )


def extract_amount(transcript: str) -> str | None:
    """Extract the most likely amount from a transcript."""
    normalized = _normalize_text(transcript)
    candidates: list[tuple[int, int, str]] = []

    for match in _DIGIT_RUN.finditer(normalized):
        digits = re.sub(r"\D", "", match.group())
        if digits:
            candidates.append((len(digits), match.start(), digits))

    for start, tokens in _numeric_spans(normalized):
        parsed = _parse_number_tokens(tokens)
        if parsed is not None:
            candidates.append((len(parsed), start, parsed))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _ensure_cached_utterance(
    utterance: TestUtterance,
    voices: Sequence[str],
    rates: Mapping[str, float],
    cache_root: Path,
    *,
    language_code: str,
) -> Path:
    """Synthesize one utterance into a cache directory keyed by text."""
    text_hash = _text_hash(utterance.text)
    cache_dir = cache_root / text_hash
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        phrase = cached.get("phrases", [{}])[0]
        if (
            phrase.get("text") == utterance.text
            and list(cached.get("voices", [])) == list(voices)
            and dict(cached.get("rates", {})) == dict(rates)
            and cached.get("language_code", "a") == language_code
        ):
            return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    synthesize_kokoro_sweep(
        cache_dir,
        attack="numeric-robustness-cache",
        phrases=[
            {
                "id": text_hash,
                "text": utterance.text,
                "spoken_amount": utterance.spoken_amount,
            }
        ],
        voices=voices,
        rates=rates,
        language_code=language_code,
    )
    return cache_dir


def _score_sample(
    *,
    sweep_dir: Path,
    sample: dict[str, object],
    transcriber: Transcriber,
    repeats: int,
    threshold: float,
    amount_extractor: Callable[[str], str | None],
) -> NumericBenchmarkRow:
    """Score one sample from the numeric benchmark sweep."""
    wav_name = str(sample["file"])
    wav_path = sweep_dir / wav_name
    spoken_amount = str(sample["spoken_amount"])
    utterance_id = str(sample["id"])
    text = str(sample["text"])
    voice = str(sample["voice"])
    rate = str(sample["rate"])
    language = str(sample["language"]) if sample.get("language") is not None else None

    transcripts = tuple(transcriber.transcribe(wav_path) for _ in range(repeats))
    observations = tuple(amount_extractor(transcript) for transcript in transcripts)
    counts = Counter(observations)
    distribution = tuple(
        AmountCount(amount=amount, count=count)
        for amount, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0] is None, str(item[0])),
        )
    )
    error_rate = sum(1 for amount in observations if amount != spoken_amount) / repeats
    misheard_counts = Counter(
        amount for amount in observations if amount not in {spoken_amount, None}
    )
    most_common_misheard_amount = (
        misheard_counts.most_common(1)[0][0] if misheard_counts else None
    )
    is_high_error = error_rate >= threshold

    return NumericBenchmarkRow(
        backend=type(transcriber).__name__.removesuffix("Transcriber").lower() or "unknown",
        id=utterance_id,
        text=text,
        spoken_amount=spoken_amount,
        language=language,
        voice=voice,
        rate=rate,
        wav_path=str(wav_path),
        repeats=repeats,
        transcripts=transcripts,
        observations=observations,
        distribution=distribution,
        error_rate=error_rate,
        most_common_misheard_amount=most_common_misheard_amount,
        is_high_error=is_high_error,
    )


def _copy_high_error_clips(
    rows: Sequence[NumericBenchmarkRow],
    sweep_dir: Path,
    high_error_dir: Path,
) -> None:
    """Copy high-error clips into a regression fixture directory."""
    high_error_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = sweep_dir / Path(row.wav_path).name
        destination = high_error_dir / source.name
        if row.is_high_error:
            shutil.copy2(source, destination)
        elif destination.exists():
            # Remove only this benchmark's previously generated copy when a
            # corrected/rerun result no longer meets the high-error threshold.
            destination.unlink()


def _write_summary(
    rows: Sequence[NumericBenchmarkRow],
    repeats: int,
    threshold: float,
    results_dir: Path,
) -> None:
    """Write a concise human-readable summary next to the JSONL results."""
    summary_path = results_dir / "summary.txt"
    observation_count = sum(row.repeats for row in rows)
    error_count = sum(round(row.error_rate * row.repeats) for row in rows)
    clip_count = len(rows)
    observation_denominator = observation_count or 1
    clip_denominator = clip_count or 1
    lines = [
        f"repeats={repeats} threshold={threshold}",
        f"clips={clip_count}",
        f"observations={observation_count}",
        f"gate1_failure_rate={error_count / observation_denominator:.6f}",
        "clips_with_any_error_rate="
        f"{sum(row.error_rate > 0 for row in rows) / clip_denominator:.6f}",
        f"high_error_rate={sum(row.is_high_error for row in rows) / clip_denominator:.6f}",
        f"high_error={sum(1 for row in rows if row.is_high_error)}",
        "",
    ]
    for row in rows[:10]:
        lines.append(
            f"{row.error_rate:.2f} {row.backend} {row.id} {row.voice}/{row.rate} "
            f"{row.spoken_amount} -> {row.most_common_misheard_amount} "
            f"({row.wav_path})"
        )
    summary_path.write_text("\n".join(lines) + "\n")


def _row_to_dict(row: NumericBenchmarkRow) -> dict[str, object]:
    """Serialize one result row for JSONL output."""
    return {
        "backend": row.backend,
        "id": row.id,
        "text": row.text,
        "spoken_amount": row.spoken_amount,
        "language": row.language,
        "voice": row.voice,
        "rate": row.rate,
        "wav_path": row.wav_path,
        "repeats": row.repeats,
        "transcripts": list(row.transcripts),
        "observations": list(row.observations),
        "distribution": [asdict(entry) for entry in row.distribution],
        "error_rate": row.error_rate,
        "most_common_misheard_amount": row.most_common_misheard_amount,
        "is_high_error": row.is_high_error,
    }


@lru_cache(maxsize=4)
def _load_faster_whisper_model(
    model_name: str,
    download_root: Path | None,
) -> _WhisperModel:
    """Load and cache a faster-whisper model."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError(
            "faster-whisper is not installed. Install with: "
            "pip install 'voiceart-benchmark[whisper]'"
        ) from exc

    kwargs: dict[str, object] = {"compute_type": "int8"}
    if download_root is not None:
        kwargs["download_root"] = str(download_root)
    return cast("_WhisperModel", WhisperModel(model_name, **kwargs))


@lru_cache(maxsize=4)
def _load_deepgram_client(api_key: str) -> _DeepgramClient:
    """Create and cache a Deepgram client."""
    try:
        from deepgram import DeepgramClient
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError(
            "deepgram-sdk is not installed. Install with: "
            "pip install 'voiceart-benchmark[deepgram]'"
        ) from exc

    return cast("_DeepgramClient", DeepgramClient(api_key=api_key))


def _require_api_key(explicit: str | None, environment_name: str, provider: str) -> str:
    """Resolve a provider key without ever including its value in an error."""
    key = explicit or os.environ.get(environment_name)
    if not key:
        raise RuntimeError(f"{environment_name} is not set; cannot run {provider} Gate 1")
    return key


def _read_flux_pcm(wav_path: Path) -> tuple[bytes, int, int]:
    """Read a mono 16-bit PCM WAV and derive an 80 ms Flux chunk size."""
    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError("Deepgram Flux fixtures must be mono 16-bit PCM WAV files")
        if wav_file.getcomptype() != "NONE":
            raise ValueError("Deepgram Flux fixtures must contain uncompressed PCM")
        sample_rate = wav_file.getframerate()
        audio = wav_file.readframes(wav_file.getnframes())
    chunk_bytes = round(sample_rate * 0.08) * 2
    return audio, sample_rate, chunk_bytes


def _post_multipart(
    url: str,
    *,
    headers: Mapping[str, str],
    fields: Mapping[str, str],
    wav_path: Path,
) -> dict[str, object]:
    """POST one WAV and simple text fields as deterministic multipart data."""
    boundary = "voiceart-gate1-boundary"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{wav_path.name}"\r\n'
            ).encode(),
            b"Content-Type: audio/wav\r\n\r\n",
            wav_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    request = Request(
        url,
        data=b"".join(chunks),
        headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return _request_json(request)


def _request_json(request: Request) -> dict[str, object]:
    """Execute one provider request and require a JSON object response."""
    with urlopen(request, timeout=180) as response:  # noqa: S310 - fixed provider URLs
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("STT provider returned a non-object JSON response")
    return payload


def _normalize_text(text: str) -> str:
    """Lowercase and flatten punctuation for extraction."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[$£€,;:(){}\[\]!?/\\]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _numeric_spans(text: str) -> list[tuple[int, list[str]]]:
    """Return contiguous number-word spans from normalized text."""
    spans: list[tuple[int, list[str]]] = []
    tokens = [(match.start(), match.group()) for match in _TOKEN_RUN.finditer(text)]
    index = 0
    while index < len(tokens):
        start_pos, token = tokens[index]
        if not _is_numeric_token(token):
            index += 1
            continue

        span_tokens = [token]
        end_index = index + 1
        while end_index < len(tokens):
            _next_pos, next_token = tokens[end_index]
            if not _is_numeric_token(next_token):
                break
            span_tokens.append(next_token)
            end_index += 1

        spans.append((start_pos, span_tokens))
        index = end_index
    return spans


def _is_numeric_token(token: str) -> bool:
    """Return whether a token participates in number parsing."""
    return token in _KNOWN_NUMBER_WORDS or token.isdigit()


def _parse_number_tokens(tokens: list[str]) -> str | None:
    """Parse a token span into a normalized digit string."""
    cleaned = [token for token in tokens if token not in _CONNECTORS]
    if not cleaned:
        return None

    if any(token in _SCALE_WORDS for token in cleaned):
        standard = _parse_standard_words(cleaned)
        if standard is not None:
            return standard
        compact = _parse_compact_words(cleaned)
        if compact is not None:
            return compact
        return None

    compact = _parse_compact_words(cleaned)
    if compact is not None:
        return compact

    standard = _parse_standard_words(cleaned)
    if standard is not None:
        return standard
    return None


def _parse_compact_words(tokens: list[str]) -> str | None:
    """Parse compact forms like 'twenty five fifty' or 'two 550'."""
    parts: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _TENS_WORDS and index + 1 < len(tokens):
            next_token = tokens[index + 1]
            if next_token in _SINGLE_WORDS or next_token in _ALIASES:
                parts.append(str(_TENS_WORDS[token] + _word_value(next_token)))
                index += 2
                continue
        if token in _SINGLE_WORDS or token in _TENS_WORDS or token in _ALIASES:
            parts.append(str(_word_value(token)))
            index += 1
            continue
        if token.isdigit():
            parts.append(token)
            index += 1
            continue
        return None

    return "".join(parts) if parts else None


def _parse_standard_words(tokens: list[str]) -> str | None:
    """Parse a conventional number phrase into digits."""
    via_library = _parse_with_word2number(tokens)
    if via_library is not None:
        return via_library

    total = 0
    current = 0
    saw_any = False
    for token in tokens:
        if token in _CONNECTORS:
            continue
        if token in _SINGLE_WORDS:
            current += _SINGLE_WORDS[token]
            saw_any = True
            continue
        if token in _TENS_WORDS:
            current += _TENS_WORDS[token]
            saw_any = True
            continue
        if token in _ALIASES:
            current += _ALIASES[token]
            saw_any = True
            continue
        if token == "hundred":
            current = max(1, current) * 100
            saw_any = True
            continue
        if token == "thousand":
            total += max(1, current) * 1000
            current = 0
            saw_any = True
            continue
        if token == "million":
            total += max(1, current) * 1_000_000
            current = 0
            saw_any = True
            continue
        if token.isdigit():
            current = current * 10 + int(token)
            saw_any = True
            continue
        return None

    if not saw_any:
        return None
    return str(total + current)


def _parse_with_word2number(tokens: list[str]) -> str | None:
    """Parse a number phrase with word2number when it is available."""
    try:
        from word2number import w2n  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        return str(w2n.word_to_num(" ".join(tokens)))
    except Exception:
        return None


def _word_value(token: str) -> int:
    """Return the numeric value of a one-token number word."""
    if token in _SINGLE_WORDS:
        return _SINGLE_WORDS[token]
    if token in _TENS_WORDS:
        return _TENS_WORDS[token]
    return _ALIASES[token]


def _text_hash(text: str) -> str:
    """Return a stable cache key for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sample_filename(
    sample_id: str,
    voice: str,
    rate_name: str,
    speed: float | None = None,
) -> str:
    """Build a stable WAV filename for one synth sample."""
    if speed is None:
        return f"{sample_id}__{voice}__{rate_name}.wav"
    speed_slug = f"{speed:.2f}".replace(".", "p")
    return f"{sample_id}__{voice}__{rate_name}-{speed_slug}.wav"
