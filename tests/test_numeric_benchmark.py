"""Tests for the numeric-robustness benchmark harness."""

import json
import sys
import wave
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voiceart.numeric_benchmark import (
    DEFAULT_UTTERANCES,
    AssemblyAITranscriber,
    CartesiaTranscriber,
    DeepgramFluxTranscriber,
    DeepgramTranscriber,
    ElevenLabsTranscriber,
    FasterWhisperTranscriber,
    GroqTranscriber,
    OpenAITranscriber,
    TestUtterance,
    Transcriber,
    _read_flux_pcm,
    _sweep_matches_configuration,
    benchmark_numeric_utterances,
    extract_amount,
    measure_numeric_robustness,
)


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("150", "150"),
        ("$150", "150"),
        ("2,550", "2550"),
        ("2 550", "2550"),
        ("two thousand five hundred fifty", "2550"),
        ("twenty-five fifty", "2550"),
        ("twenty five fifty", "2550"),
        ("two five five zero", "2550"),
        ("two 550", "2550"),
        ("Please record two 550 units for the QA fixture.", "2550"),
        ("one hundred fifty", "150"),
        ("forty two", "42"),
        ("one thousand two hundred", "1200"),
        ("No amount here", None),
    ],
)
def test_extract_amount_handles_common_number_forms(
    transcript: str, expected: str | None
) -> None:
    """Amount extraction should normalize digit and number-word transcripts."""
    assert extract_amount(transcript) == expected


def test_extract_amount_prefers_longer_numeric_candidate() -> None:
    """The extractor should prefer the longer amount-like candidate."""
    assert extract_amount("I heard 12 and then 340") == "340"


def test_benchmark_numeric_utterances_reuses_text_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical text should only synthesize once and then reuse the cache."""
    calls: list[Path] = []

    def fake_synthesize(
        output_dir: Path,
        *,
        attack: str,
        phrases: list[dict[str, object]],
            voices: tuple[str, ...] | list[str],
            rates: Mapping[str, float],
            language_code: str = "a",
    ) -> Path:
        calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_id = phrases[0]["id"]
        for voice in voices:
            for rate_name, speed in rates.items():
                speed_slug = f"{speed:.2f}".replace(".", "p")
                wav_name = f"{cache_id}__{voice}__{rate_name}-{speed_slug}.wav"
                (output_dir / wav_name).write_bytes(b"wav")
        manifest = {
            "phrases": list(phrases),
            "voices": list(voices),
                "rates": dict(rates),
                "language_code": language_code,
            "samples": [],
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest))
        return output_dir / "manifest.json"

    monkeypatch.setattr("voiceart.numeric_benchmark.synthesize_kokoro_sweep", fake_synthesize)
    utterances = (
        TestUtterance("one", "Please record 150 units.", "150"),
        TestUtterance("two", "Please record 150 units.", "150"),
    )

    manifest_path = benchmark_numeric_utterances(
        tmp_path / "sweep",
        utterances=utterances,
        voices=("af_bella",),
        rates={"fast": 1.35},
        cache_root=tmp_path / "cache",
    )

    manifest = json.loads(manifest_path.read_text())
    assert len(calls) == 1
    assert manifest["sample_count"] == 2
    assert (tmp_path / "sweep" / "one__af_bella__fast.wav").exists()
    assert (tmp_path / "sweep" / "two__af_bella__fast.wav").exists()


def test_complete_locked_sweep_is_reused_without_synthesis(tmp_path: Path) -> None:
    """A complete matching corpus should be accepted without regeneration."""
    sweep_dir = tmp_path / "numeric"
    sweep_dir.mkdir()
    utterance = TestUtterance("numeric-150", "Please record 150 units.", "150")
    wav_name = "numeric-150__af_bella__normal.wav"
    (sweep_dir / wav_name).write_bytes(b"locked-wav")
    (sweep_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sample_count": 1,
                "utterances": [
                    {
                        "id": utterance.id,
                        "text": utterance.text,
                        "spoken_amount": utterance.spoken_amount,
                    }
                ],
                "voices": ["af_bella"],
                "rates": {"normal": 1.0},
                "language_code": "a",
                "samples": [{"file": wav_name}],
            }
        )
    )

    assert _sweep_matches_configuration(
        sweep_dir,
        utterances=(utterance,),
        voices=("af_bella",),
        rates={"normal": 1.0},
        language_code="a",
    )

    (sweep_dir / wav_name).unlink()
    assert not _sweep_matches_configuration(
        sweep_dir,
        utterances=(utterance,),
        voices=("af_bella",),
        rates={"normal": 1.0},
        language_code="a",
    )


def test_measure_numeric_robustness_writes_rows_and_copies_high_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scorer should write JSONL rows and copy only high-error clips."""
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "one__af_bella__fast.wav").write_bytes(b"one")
    (sweep_dir / "two__af_bella__fast.wav").write_bytes(b"two")
    manifest = {
        "samples": [
            {
                "file": "one__af_bella__fast.wav",
                "id": "one",
                "text": "Please record 150 units.",
                "spoken_amount": "150",
                "voice": "af_bella",
                "rate": "fast",
                "speed": 1.35,
            },
            {
                "file": "two__af_bella__fast.wav",
                "id": "two",
                "text": "Please record 2550 units.",
                "spoken_amount": "2550",
                "voice": "af_bella",
                "rate": "fast",
                "speed": 1.35,
            },
        ]
    }
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest))

    transcripts = iter(
        [
            "150",
            "150",
            "151",
            "150",
            "150",
            "2550",
            "2550",
            "2600",
            "2600",
            "2600",
        ]
    )

    class DummyTranscriber:
        def transcribe(self, wav_path: Path) -> str:
            return next(transcripts)

    results_dir = measure_numeric_robustness(
        sweep_dir,
        transcriber=DummyTranscriber(),
        repeats=5,
        threshold=0.5,
        output_root=tmp_path / "results",
        high_error_dir=tmp_path / "high_error",
    )

    results_path = results_dir / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["error_rate"] == pytest.approx(0.6)
    assert rows[0]["is_high_error"] is True
    assert rows[0]["most_common_misheard_amount"] == "2600"
    assert rows[1]["error_rate"] == pytest.approx(0.2)
    assert rows[1]["is_high_error"] is False
    assert (tmp_path / "high_error" / "two__af_bella__fast.wav").exists()
    assert not (tmp_path / "high_error" / "one__af_bella__fast.wav").exists()
    assert (results_dir / "summary.txt").exists()


def test_measure_removes_stale_high_error_copy(tmp_path: Path) -> None:
    """A corrected rerun should not leave obsolete high-error regression WAVs."""
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    wav_name = "one__af_bella__fast.wav"
    (sweep_dir / wav_name).write_bytes(b"wav")
    (sweep_dir / "manifest.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "file": wav_name,
                        "id": "one",
                        "text": "Please record 150 units.",
                        "spoken_amount": "150",
                        "voice": "af_bella",
                        "rate": "fast",
                    }
                ]
            }
        )
    )
    high_error_dir = tmp_path / "high-error"
    high_error_dir.mkdir()
    stale = high_error_dir / wav_name
    stale.write_bytes(b"old")

    class CorrectTranscriber:
        def transcribe(self, wav_path: Path) -> str:
            return "150"

    measure_numeric_robustness(
        sweep_dir,
        transcriber=CorrectTranscriber(),
        repeats=1,
        output_root=tmp_path / "results",
        high_error_dir=high_error_dir,
    )

    assert not stale.exists()


def test_transcriber_protocol_accepts_faster_whisper_transcriber() -> None:
    """The concrete transcriber should satisfy the protocol shape."""
    assert isinstance(FasterWhisperTranscriber(), Transcriber)
    assert len(DEFAULT_UTTERANCES) == 10
    assert isinstance(DeepgramFluxTranscriber(), Transcriber)


def test_flux_pcm_reader_builds_80ms_chunks(tmp_path: Path) -> None:
    """Flux audio should be raw mono PCM split into its recommended chunk size."""
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(bytes(24_000 * 2))

    audio, sample_rate, chunk_bytes = _read_flux_pcm(wav_path)

    assert sample_rate == 24_000
    assert len(audio) == 48_000
    assert chunk_bytes == 3_840


def test_deepgram_transcriber_uses_sdk_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Deepgram adapter should read the transcript from the SDK response."""
    alt = MagicMock()
    alt.transcript = "150"
    channel = MagicMock()
    channel.alternatives = [alt]
    response = MagicMock()
    response.results.channels = [channel]

    media = MagicMock()
    media.transcribe_file.return_value = response
    client = MagicMock()
    client.listen.v1.media = media

    fake_sdk = MagicMock(DeepgramClient=MagicMock(return_value=client))
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"fake")

    with patch.dict(sys.modules, {"deepgram": fake_sdk}):
        transcriber = DeepgramTranscriber()
        assert transcriber.transcribe(wav_path) == "150"

    media.transcribe_file.assert_called_once()


@pytest.mark.parametrize(
    ("transcriber", "environment_name", "expected_url"),
    [
        (CartesiaTranscriber(), "CARTESIA_API_KEY", "api.cartesia.ai/stt"),
        (
            ElevenLabsTranscriber(),
            "ELEVENLABS_API_KEY",
            "api.elevenlabs.io/v1/speech-to-text",
        ),
        (GroqTranscriber(), "GROQ_API_KEY", "api.groq.com/openai/v1/audio/transcriptions"),
        (OpenAITranscriber(), "OPENAI_API_KEY", "api.openai.com/v1/audio/transcriptions"),
    ],
)
def test_http_provider_transcribers_use_structured_text_response(
    transcriber: Transcriber,
    environment_name: str,
    expected_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synchronous provider adapters should return the response text field."""
    monkeypatch.setenv(environment_name, "test-key")
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"wav")
    post = MagicMock(return_value={"text": "two thousand five hundred fifty"})
    monkeypatch.setattr("voiceart.numeric_benchmark._post_multipart", post)

    assert transcriber.transcribe(wav_path) == "two thousand five hundred fifty"
    assert expected_url in post.call_args.args[0]


def test_openai_transcriber_uses_reproducible_batch_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenAI adapter should use the quality model with deterministic settings."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"wav")
    post = MagicMock(return_value={"text": "forty two"})
    monkeypatch.setattr("voiceart.numeric_benchmark._post_multipart", post)

    assert OpenAITranscriber().transcribe(wav_path) == "forty two"
    assert post.call_args.kwargs["fields"] == {
        "model": "gpt-4o-transcribe",
        "language": "en",
        "temperature": "0",
        "response_format": "json",
    }


def test_assemblyai_transcriber_uploads_submits_and_polls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The AssemblyAI adapter should complete its asynchronous API lifecycle."""
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"wav")
    request_json = MagicMock(
        side_effect=[
            {"upload_url": "https://example.test/audio"},
            {"id": "transcript-id"},
            {"status": "processing"},
            {"status": "completed", "text": "forty two"},
        ]
    )
    monkeypatch.setattr("voiceart.numeric_benchmark._request_json", request_json)
    monkeypatch.setattr("voiceart.numeric_benchmark.time.sleep", MagicMock())

    transcript = AssemblyAITranscriber(poll_interval=0).transcribe(wav_path)

    assert transcript == "forty two"
    assert request_json.call_count == 4
