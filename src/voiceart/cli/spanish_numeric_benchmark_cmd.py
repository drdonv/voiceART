"""CLI for the Spanish Deepgram Nova-3 numeric benchmark."""

from pathlib import Path

import click

from voiceart.numeric_benchmark import DeepgramTranscriber, benchmark_numeric_utterances
from voiceart.spanish_numeric_benchmark import (
    SPANISH_RATES,
    SPANISH_UTTERANCES,
    SPANISH_VOICES,
    run_spanish_numeric_benchmark,
)


@click.command()
@click.option(
    "--sweep-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output/kokoro/spanish-numeric"),
    show_default=True,
)
@click.option(
    "--results-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output/stt-numeric-spanish/deepgram-nova3"),
    show_default=True,
)
@click.option(
    "--high-error-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("fixtures/high_error/spanish/deepgram-nova3"),
    show_default=True,
)
@click.option(
    "--cache-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output/kokoro-cache/spanish-numeric"),
    show_default=True,
)
@click.option("--repeats", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--threshold", type=float, default=0.5, show_default=True)
@click.option("--deepgram-api-key", envvar="DEEPGRAM_API_KEY", default=None)
@click.option(
    "--synthesize-only",
    is_flag=True,
    help="Create/reuse the 40 WAV fixtures without calling Deepgram.",
)
def spanish_numeric_benchmark_command(
    sweep_dir: Path,
    results_root: Path,
    high_error_dir: Path,
    cache_root: Path,
    repeats: int,
    threshold: float,
    deepgram_api_key: str | None,
    synthesize_only: bool,
) -> None:
    """Run 40 Spanish clips × four Nova-3 batch transcriptions."""
    if synthesize_only:
        manifest = benchmark_numeric_utterances(
            sweep_dir,
            utterances=SPANISH_UTTERANCES,
            voices=SPANISH_VOICES,
            rates=SPANISH_RATES,
            cache_root=cache_root,
            language_code="e",
        )
        click.echo(f"Spanish fixtures: 40 ({manifest})")
        return

    results_dir = run_spanish_numeric_benchmark(
        sweep_dir=sweep_dir,
        output_root=results_root,
        high_error_dir=high_error_dir,
        cache_root=cache_root,
        repeats=repeats,
        threshold=threshold,
        transcriber=DeepgramTranscriber(
            api_key=deepgram_api_key,
            model_name="nova-3",
            language="es",
            numerals=True,
        ),
    )
    click.echo((results_dir / "summary.txt").read_text().rstrip())
