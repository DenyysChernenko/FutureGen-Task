import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_pipeline_config
from src.pipeline import AudioPipeline
from src.utils.logger import configure_logging

app = typer.Typer(
    name="futugen-pipeline",
    help="FutuGen Audio Transcription & Classification Pipeline",
    add_completion=False,
)


@app.command()
def process(
    audio_file: Path = typer.Argument(
        ...,
        help="Path to audio file (WAV format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID for this processing run (auto-generated from filename if not provided)",
    ),
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to pipeline configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (no -v: quiet, -v: INFO, -vv: DEBUG)",
    ),
) -> None:
    configure_logging(verbose)
    asyncio.run(process_audio(audio_file, session_id, config))


async def process_audio(
    audio_file_path: Path, session_id: Optional[str], config_path: Path
) -> None:
    config = load_pipeline_config(config_path)
    pipeline = AudioPipeline(config)

    if session_id is None:
        session_id = f"session_{audio_file_path.stem}"

    await pipeline.process(
        audio_file_path=str(audio_file_path),
        session_id=session_id,
    )


if __name__ == "__main__":
    app()
