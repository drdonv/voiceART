"""Generate the canonical Gate 1 provider comparison from raw JSONL evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voiceart.numeric_benchmark import extract_amount

ROOT = Path(__file__).resolve().parents[1]
CORPUS_LOCK = ROOT / "benchmark/numeric-corpus-lock.json"
OUTPUT_JSON = ROOT / "benchmark/provider-comparison.json"
OUTPUT_MARKDOWN = ROOT / "benchmark/provider-comparison.md"

AMBIGUOUS_RECOVERABLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("25 50", re.compile(r"\b25\s+50\b")),
    ("25-50", re.compile(r"\b25-50\b")),
    ("25.50", re.compile(r"\b25\.50\b")),
    ("two 550", re.compile(r"\btwo\s+550\b")),
    ("two 550s", re.compile(r"\btwo\s+550s\b")),
)


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
    ProviderRun(
        provider="openai",
        display_name="OpenAI",
        model="gpt-4o-transcribe",
        results_path=ROOT / "benchmark/runs/openai-gpt-4o-transcribe-4x/results.jsonl",
        settings={
            "language": "en",
            "temperature": 0,
            "response_format": "json",
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


def _ambiguous_recoverable_label(transcript: str) -> str | None:
    """Return a label when a transcript is numerically salvageable but ambiguous."""
    normalized = transcript.lower()
    for label, pattern in AMBIGUOUS_RECOVERABLE_PATTERNS:
        if pattern.search(normalized):
            return label
    return None


def _summarize_ambiguous_examples(
    observations: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return compact summaries and one representative example per surface form."""
    counts = Counter(str(observation["surface_form"]) for observation in observations)
    clip_sets: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    examples: dict[str, dict[str, object]] = {}
    for observation in observations:
        label = str(observation["surface_form"])
        clip_sets[label].add(
            (
                str(observation["id"]),
                str(observation["voice"]),
                str(observation["rate"]),
            )
        )
        examples.setdefault(label, observation)

    summaries = [
        {
            "surface_form": label,
            "observations": counts[label],
            "clips": len(clip_sets[label]),
        }
        for label in sorted(counts)
    ]
    representative_examples = [examples[label] for label in sorted(examples)]
    return summaries, representative_examples


def _verify_corpus_lock(
    lock: dict[str, Any],
) -> tuple[set[tuple[str, str, str, str, str]], dict[str, str]]:
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
    return expected_rows, file_records


def _summarize_run(
    run: ProviderRun,
    expected_rows: set[tuple[str, str, str, str, str]],
    file_digests: dict[str, str],
) -> dict[str, Any]:
    """Validate and summarize one fresh provider run."""
    rows = _read_jsonl(run.results_path)
    actual_rows: set[tuple[str, str, str, str, str]] = set()
    failures: list[dict[str, object]] = []
    affected_clips: set[tuple[str, str, str]] = set()
    ambiguous_observations: list[dict[str, object]] = []
    ambiguous_clips: set[tuple[str, str, str]] = set()

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

        wav_filename = Path(str(row["wav_path"])).name
        expected_digest = file_digests.get(wav_filename)
        if expected_digest is None:
            raise ValueError(f"{run.results_path}: {wav_filename} is not in the locked corpus")
        wav_path = Path(str(row["wav_path"]))
        if wav_path.exists() and _sha256(wav_path) != expected_digest:
            raise ValueError(f"{run.results_path}: {wav_filename} digest does not match the lock")

        clip_key = (str(row["id"]), str(row["voice"]), str(row["rate"]))
        actual_rows.add(
            (
                *clip_key,
                wav_filename,
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
                continue

            ambiguous_label = _ambiguous_recoverable_label(transcript)
            if ambiguous_label is not None:
                ambiguous_clips.add(clip_key)
                ambiguous_observations.append(
                    {
                        "id": row["id"],
                        "voice": row["voice"],
                        "rate": row["rate"],
                        "repeat": repeat,
                        "spoken_amount": row["spoken_amount"],
                        "raw_transcript": transcript,
                        "surface_form": ambiguous_label,
                    }
                )

    if len(rows) != 90 or actual_rows != expected_rows:
        raise ValueError(f"{run.results_path} does not match the locked 90-clip corpus")

    observations = sum(int(row["repeats"]) for row in rows)
    high_error_threshold = 0.5
    high_error_clips = 0
    for row in rows:
        errors = sum(
            1 for obs, amt in zip(row["observations"], [row["spoken_amount"]] * 4, strict=True)
            if obs != amt
        )
        if errors / int(row["repeats"]) >= high_error_threshold:
            high_error_clips += 1
    ambiguous_breakdown, ambiguous_examples = _summarize_ambiguous_examples(
        ambiguous_observations
    )

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
        "ambiguous_recoverable_observations": len(ambiguous_observations),
        "ambiguous_recoverable_rate": len(ambiguous_observations) / observations,
        "ambiguous_recoverable_clips": len(ambiguous_clips),
        "ambiguous_recoverable_clip_rate": len(ambiguous_clips) / len(rows),
        "ambiguous_recoverable_breakdown": ambiguous_breakdown,
        "ambiguous_recoverable_examples": ambiguous_examples,
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
        "| Provider | Model | Egregious errors | Error rate | "
        "Affected clips | Ambiguous but recoverable | Ambiguous clips | High-error clips |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in comparison["providers"]:
        lines.append(
            f"| {provider['display_name']} | `{provider['model']}` | "
            f"{provider['incorrect_observations']}/{provider['observations']} | "
            f"{provider['failure_rate']:.2%} | "
            f"{provider['affected_clips']}/{provider['clips']} | "
            f"{provider['ambiguous_recoverable_observations']}/{provider['observations']} | "
            f"{provider['ambiguous_recoverable_clips']}/{provider['clips']} | "
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
    lines.extend(("", "## Ambiguous but recoverable transcripts", ""))
    breakdown = [
        (provider["display_name"], row)
        for provider in comparison["providers"]
        for row in provider["ambiguous_recoverable_breakdown"]
    ]
    if not breakdown:
        lines.append("No ambiguous-but-recoverable transcripts.")
    else:
        lines.extend(
            (
                "| Provider | Surface form | Observations | Affected clips |",
                "|---|---|---:|---:|",
            )
        )
        for provider_name, row in breakdown:
            lines.append(
                f"| {provider_name} | `{row['surface_form']}` | "
                f"{row['observations']} | {row['clips']} |"
            )

        lines.extend(("", "Representative examples:", ""))
        lines.extend(
            (
                "| Provider | Clip | Voice/rate | Repeat | Expected | Surface form | Raw transcript |",
                "|---|---|---|---:|---:|---|---|",
            )
        )
        examples = [
            (provider["display_name"], observation)
            for provider in comparison["providers"]
            for observation in provider["ambiguous_recoverable_examples"]
        ]
        for provider_name, observation in examples:
            transcript = str(observation["raw_transcript"]).replace("|", "\\|")
            lines.append(
                f"| {provider_name} | `{observation['id']}` | "
                f"`{observation['voice']}/{observation['rate']}` | "
                f"{observation['repeat']} | {observation['spoken_amount']} | "
                f"`{observation['surface_form']}` | {transcript} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Validate fresh evidence and write canonical comparison artifacts."""
    lock = _read_json(CORPUS_LOCK)
    expected_rows, file_digests = _verify_corpus_lock(lock)
    comparison = {
        "schema_version": 2,
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
        "providers": [_summarize_run(run, expected_rows, file_digests) for run in RUNS],
    }
    OUTPUT_JSON.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MARKDOWN.write_text(_render_markdown(comparison), encoding="utf-8")
    print(OUTPUT_MARKDOWN.relative_to(ROOT))


if __name__ == "__main__":
    main()
