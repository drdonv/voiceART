"""Generate the canonical Gate 1 provider comparison from raw JSONL evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voiceart.numeric_benchmark import extract_amount

ROOT = Path(__file__).resolve().parents[1]
CORPUS_LOCK = ROOT / "benchmark/numeric-corpus-lock.json"
OUTPUT_JSON = ROOT / "benchmark/provider-comparison.json"
OUTPUT_MARKDOWN = ROOT / "benchmark/provider-comparison.md"


@dataclass(frozen=True, slots=True)
class ProviderRun:
    """One canonical provider run and its recorded API configuration."""

    provider: str
    display_name: str
    model: str
    results_path: Path
    settings: dict[str, object]


RUNS = (
    ProviderRun(
        provider="deepgram",
        display_name="Deepgram",
        model="nova-3",
        results_path=ROOT / "benchmark/runs/deepgram-nova3-4x/results.jsonl",
        settings={
            "language": "en",
            "punctuate": True,
            "smart_format": True,
            "numerals": True,
        },
    ),
    ProviderRun(
        provider="cartesia",
        display_name="Cartesia",
        model="ink-whisper",
        results_path=ROOT / "benchmark/runs/cartesia-ink-4x/results.jsonl",
        settings={
            "language": "en",
            "api_version": "2026-03-01",
        },
    ),
    ProviderRun(
        provider="elevenlabs",
        display_name="ElevenLabs",
        model="scribe_v2",
        results_path=ROOT / "benchmark/runs/elevenlabs-scribe-v2-4x/results.jsonl",
        settings={
            "language_code": "eng",
            "tag_audio_events": False,
            "diarize": False,
        },
    ),
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects from a JSONL file."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _verify_corpus_lock(lock: dict[str, Any]) -> set[tuple[str, str, str, str, str]]:
    """Verify the manifest, all WAV hashes, and the aggregate corpus digest."""
    manifest_record = lock["manifest"]
    manifest_path = ROOT / manifest_record["path"]
    if _sha256(manifest_path) != manifest_record["sha256"]:
        raise ValueError("numeric corpus manifest does not match its lock")

    manifest = _read_json(manifest_path)
    samples = manifest["samples"]
    file_records = {record["file"]: record["sha256"] for record in lock["files"]}
    if len(samples) != lock["clip_count"] or len(file_records) != lock["clip_count"]:
        raise ValueError("numeric corpus lock count does not match its manifest")

    ordered_hashes: list[str] = []
    expected_rows: set[tuple[str, str, str, str, str]] = set()
    for sample in samples:
        filename = sample["file"]
        wav_digest = _sha256(manifest_path.parent / filename)
        if wav_digest != file_records.get(filename):
            raise ValueError(f"{filename} does not match the numeric corpus lock")
        ordered_hashes.append(wav_digest)
        expected_rows.add(
            (
                str(sample["id"]),
                str(sample["voice"]),
                str(sample["rate"]),
                str(filename),
                str(sample["spoken_amount"]),
            )
        )

    aggregate = hashlib.sha256("".join(ordered_hashes).encode()).hexdigest()
    if aggregate != lock["aggregate"]["sha256"]:
        raise ValueError("numeric corpus aggregate does not match its lock")
    return expected_rows


def _summarize_run(
    run: ProviderRun,
    expected_rows: set[tuple[str, str, str, str, str]],
) -> dict[str, Any]:
    """Validate and summarize one fresh provider run."""
    rows = _read_jsonl(run.results_path)
    actual_rows: set[tuple[str, str, str, str, str]] = set()
    failures: list[dict[str, object]] = []
    affected_clips: set[tuple[str, str, str]] = set()

    for row in rows:
        if row.get("backend") != run.provider:
            raise ValueError(f"{run.results_path} contains a non-{run.provider} row")
        if row.get("repeats") != 4:
            raise ValueError(f"{run.results_path} contains a non-four-repeat row")

        transcripts = row.get("transcripts")
        observations = row.get("observations")
        if not isinstance(transcripts, list) or not isinstance(observations, list):
            raise ValueError(f"{run.results_path} is missing raw transcript evidence")
        if len(transcripts) != 4 or len(observations) != 4:
            raise ValueError(f"{run.results_path} contains an incomplete observation row")

        clip_key = (str(row["id"]), str(row["voice"]), str(row["rate"]))
        actual_rows.add(
            (
                *clip_key,
                Path(str(row["wav_path"])).name,
                str(row["spoken_amount"]),
            )
        )
        for repeat, (transcript, observation) in enumerate(
            zip(transcripts, observations, strict=True),
            start=1,
        ):
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(f"{run.results_path} contains an empty raw transcript")
            if extract_amount(transcript) != observation:
                raise ValueError(
                    f"{run.results_path} contains an observation inconsistent with the scorer"
                )
            if observation != row["spoken_amount"]:
                affected_clips.add(clip_key)
                failures.append(
                    {
                        "id": row["id"],
                        "voice": row["voice"],
                        "rate": row["rate"],
                        "repeat": repeat,
                        "spoken_amount": row["spoken_amount"],
                        "raw_transcript": transcript,
                        "observed_amount": observation,
                    }
                )

    if len(rows) != 90 or actual_rows != expected_rows:
        raise ValueError(f"{run.results_path} does not match the locked 90-clip corpus")

    observations = sum(int(row["repeats"]) for row in rows)
    high_error_clips = sum(bool(row["is_high_error"]) for row in rows)
    return {
        "provider": run.provider,
        "display_name": run.display_name,
        "model": run.model,
        "settings": run.settings,
        "run_date": "2026-07-23",
        "results_path": str(run.results_path.relative_to(ROOT)),
        "results_sha256": _sha256(run.results_path),
        "clips": len(rows),
        "repeats_per_clip": 4,
        "observations": observations,
        "incorrect_observations": len(failures),
        "failure_rate": len(failures) / observations,
        "affected_clips": len(affected_clips),
        "affected_clip_rate": len(affected_clips) / len(rows),
        "high_error_clips": high_error_clips,
        "failures": failures,
    }


def _render_markdown(comparison: dict[str, Any]) -> str:
    """Render the human-readable canonical comparison."""
    lines = [
        "# Gate 1 English numeric provider comparison",
        "",
        "This file is generated by `scripts/generate_provider_comparison.py`.",
        "Do not edit it manually.",
        "",
        f"- Corpus clips: {comparison['protocol']['clips']}",
        f"- Repeats per clip: {comparison['protocol']['repeats_per_clip']}",
        f"- Corpus SHA-256: `{comparison['corpus']['aggregate_sha256']}`",
        "",
        "| Provider | Model | Incorrect observations | Failure rate | "
        "Affected clips | High-error clips |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for provider in comparison["providers"]:
        lines.append(
            f"| {provider['display_name']} | `{provider['model']}` | "
            f"{provider['incorrect_observations']}/{provider['observations']} | "
            f"{provider['failure_rate']:.2%} | "
            f"{provider['affected_clips']}/{provider['clips']} | "
            f"{provider['high_error_clips']} |"
        )

    lines.extend(("", "## Verified failures", ""))
    failures = [
        (provider["display_name"], failure)
        for provider in comparison["providers"]
        for failure in provider["failures"]
    ]
    if not failures:
        lines.append("No failures.")
    else:
        lines.extend(
            (
                "| Provider | Clip | Voice/rate | Repeat | Expected | "
                "Observed | Raw transcript |",
                "|---|---|---|---:|---:|---:|---|",
            )
        )
        for provider_name, failure in failures:
            transcript = str(failure["raw_transcript"]).replace("|", "\\|")
            lines.append(
                f"| {provider_name} | `{failure['id']}` | "
                f"`{failure['voice']}/{failure['rate']}` | "
                f"{failure['repeat']} | {failure['spoken_amount']} | "
                f"{failure['observed_amount']} | {transcript} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Validate fresh evidence and write canonical comparison artifacts."""
    lock = _read_json(CORPUS_LOCK)
    expected_rows = _verify_corpus_lock(lock)
    comparison = {
        "schema_version": 1,
        "benchmark": "numeric-robustness-qa",
        "status": "canonical",
        "protocol": {
            "clips": 90,
            "repeats_per_clip": 4,
            "high_error_threshold": 0.5,
        },
        "corpus": {
            "lock_path": str(CORPUS_LOCK.relative_to(ROOT)),
            "manifest_sha256": lock["manifest"]["sha256"],
            "aggregate_sha256": lock["aggregate"]["sha256"],
        },
        "providers": [_summarize_run(run, expected_rows) for run in RUNS],
    }
    OUTPUT_JSON.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MARKDOWN.write_text(_render_markdown(comparison), encoding="utf-8")
    print(OUTPUT_MARKDOWN.relative_to(ROOT))


if __name__ == "__main__":
    main()
