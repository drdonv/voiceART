"""CLI for the all-language Deepgram Nova-3 numeric benchmark."""

from pathlib import Path

import click

from voiceart.multilingual_numeric_benchmark import run_multilingual_numeric_benchmark


@click.command()
@click.option(
    "--sweep-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output/kokoro/multilingual-numeric"),
    show_default=True,
)
@click.option(
    "--results-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output/stt-numeric-multilingual/deepgram-nova3"),
    show_default=True,
)
@click.option(
    "--fixture-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("fixtures/high_error/multilingual/deepgram-nova3"),
    show_default=True,
)
@click.option("--repeats", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--threshold", type=float, default=0.5, show_default=True)
@click.option("--deepgram-api-key", envvar="DEEPGRAM_API_KEY", default=None)
def multilingual_numeric_benchmark_command(
    sweep_root: Path,
    results_root: Path,
    fixture_root: Path,
    repeats: int,
    threshold: float,
    deepgram_api_key: str | None,
) -> None:
    """Run 40 clips × four Nova-3 transcriptions for all 10 languages."""
    output = run_multilingual_numeric_benchmark(
        sweep_root=sweep_root,
        results_root=results_root,
        fixture_root=fixture_root,
        repeats=repeats,
        threshold=threshold,
        api_key=deepgram_api_key,
    )
    click.echo(f"Multilingual results: {output}")
