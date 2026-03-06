"""Typer CLI for the video unscrambling pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from .cluster_frames import main as cluster_main
from .compute_optimal_sequence import main as sequence_main
from .estimate_matches_motion import main as match_main
from .reconstruct_frames import main as reconstruct_main


console = Console()
app = typer.Typer(
    help="Unscramble a tampered video by clustering, matching, sequencing, and reconstruction.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


class Method(str, Enum):
    AKAZE = "AKAZE"
    RESNET = "RESNET"
    SIFT = "SIFT"
    COMBO = "COMBO"


def _run_step(title: str, argv: list[str], fn) -> None:
    console.print(Panel.fit(title, border_style="cyan"))
    fn(argv)


@app.command()
def pipeline(
    method: Method = typer.Option(Method.RESNET, help="Descriptor used for pairwise matching."),
    input: Path = typer.Option(Path("corrupted_video.mp4"), exists=False, help="Input video path."),
    output_dir: Path = typer.Option(Path("results"), help="Directory for intermediate artifacts."),
    fps: float = typer.Option(24.0, help="Output frame rate."),
    clusters: int = typer.Option(2, min=1, help="Number of frame clusters."),
    alpha: float = typer.Option(0.5, help="Motion penalty weight."),
    viz_tsne: bool = typer.Option(False, "--viz-tsne", help="Generate the t-SNE clustering view."),
) -> None:
    """Run the full reconstruction pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    inliers_dir = output_dir / "inliers"
    matches_path = output_dir / f"matches_{method.value}.npz"
    sequence_path = output_dir / f"sequence_{method.value}.npy"
    output_video = output_dir / f"reconstructed_video_{method.value}.mp4"
    save_frames_dir = output_dir / f"reconstructed_{method.value}"

    cluster_args = [
        "--input",
        str(input),
        "--output_dir",
        str(output_dir),
        "--clusters",
        str(clusters),
    ]
    if viz_tsne:
        cluster_args.append("--viz_tsne")

    _run_step("Step 1/4: Cluster frames", cluster_args, cluster_main)
    _run_step(
        "Step 2/4: Estimate matches and motion",
        [
            "--input_dir",
            str(inliers_dir),
            "--output",
            str(matches_path),
            "--descr",
            method.value,
        ],
        match_main,
    )
    _run_step(
        "Step 3/4: Compute optimal sequence",
        [
            "--input",
            str(matches_path),
            "--output",
            str(sequence_path),
            "--alpha",
            str(alpha),
            "--descr",
            method.value,
        ],
        sequence_main,
    )
    _run_step(
        "Step 4/4: Reconstruct video",
        [
            "--frames_dir",
            str(inliers_dir),
            "--sequence",
            str(sequence_path),
            "--output",
            str(output_video),
            "--fps",
            str(fps),
            "--save-frames-dir",
            str(save_frames_dir),
        ],
        reconstruct_main,
    )
    console.print(
        Panel.fit(
            f"[bold green]Done[/bold green]\nOutput: {output_video}",
            border_style="green",
        )
    )


@app.command("cluster")
def cluster_command(
    input: Path = typer.Option(..., help="Input video path."),
    output_dir: Path = typer.Option(..., help="Directory for extracted and clustered frames."),
    clusters: int = typer.Option(2, min=1, help="Number of clusters."),
    bins: int = typer.Option(64, min=1, help="Histogram bins."),
    viz_tsne: bool = typer.Option(False, "--viz-tsne", help="Generate the t-SNE clustering view."),
    resize: tuple[int, int] = typer.Option((256, 256), help="Resize width and height before histogram extraction."),
) -> None:
    """Extract frames and split inliers from outliers."""
    argv = [
        "--input",
        str(input),
        "--output_dir",
        str(output_dir),
        "--clusters",
        str(clusters),
        "--bins",
        str(bins),
        "--resize",
        str(resize[0]),
        str(resize[1]),
    ]
    if viz_tsne:
        argv.append("--viz_tsne")
    _run_step("Cluster frames", argv, cluster_main)


@app.command("match")
def match_command(
    input_dir: Path = typer.Option(..., help="Directory containing inlier .jpg frames."),
    output: Path = typer.Option(..., help="Output .npz file."),
    descr: Method = typer.Option(Method.RESNET, help="Descriptor to use."),
    ratio_thresh: float = typer.Option(0.75, help="Lowe ratio threshold for ResNet spatial matching."),
    alpha: float = typer.Option(0.5, help="Fusion weight for COMBO mode."),
) -> None:
    """Compute pairwise matches and motion matrices."""
    _run_step(
        "Estimate matches and motion",
        [
            "--input_dir",
            str(input_dir),
            "--output",
            str(output),
            "--descr",
            descr.value,
            "--ratio_thresh",
            str(ratio_thresh),
            "--alpha",
            str(alpha),
        ],
        match_main,
    )


@app.command("sequence")
def sequence_command(
    input: Path = typer.Option(..., help="Input .npz file containing matches and motion."),
    output: Path = typer.Option(..., help="Output .npy sequence file."),
    alpha: float = typer.Option(0.5, help="Motion penalty weight."),
    descr: Method = typer.Option(Method.RESNET, help="Descriptor label used for reporting."),
) -> None:
    """Recover an ordered frame sequence."""
    _run_step(
        "Compute optimal sequence",
        [
            "--input",
            str(input),
            "--output",
            str(output),
            "--alpha",
            str(alpha),
            "--descr",
            descr.value,
        ],
        sequence_main,
    )


@app.command("reconstruct")
def reconstruct_command(
    frames_dir: Path = typer.Option(..., help="Directory containing filtered frame images."),
    sequence: Path = typer.Option(..., help="Sequence .npy file."),
    output: Path = typer.Option(..., help="Output video path."),
    fps: float = typer.Option(30.0, help="Output frame rate."),
    save_frames_dir: Path | None = typer.Option(None, help="Optional directory for ordered JPEG frames."),
) -> None:
    """Write a reconstructed video from ordered frames."""
    argv = [
        "--frames_dir",
        str(frames_dir),
        "--sequence",
        str(sequence),
        "--output",
        str(output),
        "--fps",
        str(fps),
    ]
    if save_frames_dir is not None:
        argv.extend(["--save-frames-dir", str(save_frames_dir)])
    _run_step("Reconstruct video", argv, reconstruct_main)


def main() -> None:
    """Compatibility wrapper for non-Poetry entrypoints."""
    app()


if __name__ == "__main__":
    app()
