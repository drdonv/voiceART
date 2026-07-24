"""VoiceART command-line interface."""

import click

from voiceart.cli.multilingual_numeric_benchmark_cmd import (
    multilingual_numeric_benchmark_command,
)
from voiceart.cli.numeric_benchmark_cmd import numeric_benchmark_command
from voiceart.cli.spanish_numeric_benchmark_cmd import spanish_numeric_benchmark_command


@click.group()
@click.version_option(package_name="voiceart-benchmark")
def cli() -> None:
    """VoiceART — reproducible STT robustness benchmarks."""


cli.add_command(numeric_benchmark_command, "numeric")
cli.add_command(spanish_numeric_benchmark_command, "numeric-spanish")
cli.add_command(multilingual_numeric_benchmark_command, "numeric-multilingual")
