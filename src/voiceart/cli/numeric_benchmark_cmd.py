"""CLI for the local numeric-robustness benchmark."""

from pathlib import Path

import click

from voiceart.numeric_benchmark import (
    AssemblyAITranscriber,
    CartesiaTranscriber,
    DeepgramTranscriber,
    ElevenLabsTranscriber,
    FasterWhisperTranscriber,
    GroqTranscriber,
    OpenAITranscriber,
    Transcriber,
    run_numeric_benchmark,
)


@click.command()
@click.option(
    "--sweep-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("corpus/numeric"),
    show_default=True,
    help="Directory for synthesized WAVs and the sweep manifest.",
)
@click.option(
    "--results-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output/stt-numeric"),
    show_default=True,
    help="Directory that will contain the JSONL results.",
)
@click.option(
    "--high-error-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("fixtures/high_error"),
    show_default=True,
    help="Directory to receive high-error regression clips.",
)
@click.option(
    "--repeats",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="How many times to transcribe each clip.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum error-rate required to copy a clip into the regression set.",
)
@click.option(
    "--model-name",
    default="base.en",
    show_default=True,
    help="faster-whisper model to use for the reference STT pass.",
)
@click.option(
    "--backend",
    type=click.Choice(
        [
            "whisper",
            "deepgram",
            "cartesia",
            "assemblyai",
            "elevenlabs",
            "groq",
            "openai",
        ],
        case_sensitive=False,
    ),
    default="whisper",
    show_default=True,
    help="STT backend used for the benchmark run.",
)
@click.option(
    "--deepgram-api-key",
    envvar="DEEPGRAM_API_KEY",
    default=None,
    help="API key for the Deepgram backend.",
)
def numeric_benchmark_command(
    sweep_dir: Path,
    results_root: Path,
    high_error_dir: Path,
    repeats: int,
    threshold: float,
    model_name: str,
    backend: str,
    deepgram_api_key: str | None,
) -> None:
    """Run the numeric-robustness benchmark end to end."""
    backend = backend.lower()
    transcribers: dict[str, Transcriber] = {
        "whisper": FasterWhisperTranscriber(model_name=model_name),
        "deepgram": DeepgramTranscriber(api_key=deepgram_api_key),
        "cartesia": CartesiaTranscriber(),
        "assemblyai": AssemblyAITranscriber(),
        "elevenlabs": ElevenLabsTranscriber(),
        "groq": GroqTranscriber(),
        "openai": OpenAITranscriber(),
    }
    transcriber = transcribers[backend]

    results_dir = run_numeric_benchmark(
        sweep_dir=sweep_dir,
        repeats=repeats,
        threshold=threshold,
        transcriber=transcriber,
        output_root=results_root / backend,
        high_error_dir=high_error_dir / backend,
    )
    summary_path = results_dir / "summary.txt"
    click.echo(summary_path.read_text().rstrip())
