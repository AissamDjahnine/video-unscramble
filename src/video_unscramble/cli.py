"""Unified CLI for the video unscrambling pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from .cluster_frames import main as cluster_main
from .compute_optimal_sequence import main as sequence_main
from .estimate_matches_motion import main as match_main
from .reconstruct_frames import main as reconstruct_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-unscramble",
        description="Unscramble a tampered video by clustering, matching, sequencing, and reconstruction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline = subparsers.add_parser("pipeline", help="Run the full reconstruction pipeline.")
    pipeline.add_argument("--method", default="RESNET", choices=["AKAZE", "RESNET", "SIFT", "COMBO"])
    pipeline.add_argument("--input", default="corrupted_video.mp4", help="Input video path.")
    pipeline.add_argument("--output-dir", default="results", help="Directory for intermediate artifacts.")
    pipeline.add_argument("--fps", type=float, default=24.0, help="Output frame rate.")
    pipeline.add_argument("--clusters", type=int, default=2, help="Number of frame clusters.")
    pipeline.add_argument("--alpha", type=float, default=0.5, help="Motion penalty weight.")
    pipeline.add_argument("--viz-tsne", action="store_true", help="Generate the t-SNE clustering view.")

    cluster = subparsers.add_parser("cluster", help="Extract frames and split inliers/outliers.")
    cluster.add_argument("--input", required=True)
    cluster.add_argument("--output_dir", required=True)
    cluster.add_argument("--clusters", type=int, default=2)
    cluster.add_argument("--bins", type=int, default=64)
    cluster.add_argument("--viz_tsne", action="store_true")
    cluster.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=(256, 256))

    match = subparsers.add_parser("match", help="Compute pairwise matches and motion matrices.")
    match.add_argument("--input_dir", required=True)
    match.add_argument("--output", required=True)
    match.add_argument("--descr", default="RESNET", choices=["AKAZE", "SIFT", "ORB", "RESNET", "COMBO"])
    match.add_argument("--ratio_thresh", type=float, default=0.75)
    match.add_argument("--alpha", type=float, default=0.5)

    sequence = subparsers.add_parser("sequence", help="Recover an ordered frame sequence.")
    sequence.add_argument("--input", required=True)
    sequence.add_argument("--output", required=True)
    sequence.add_argument("--alpha", type=float, default=0.5)
    sequence.add_argument("--descr", type=str, default="AKAZE")

    reconstruct = subparsers.add_parser("reconstruct", help="Write a reconstructed video from ordered frames.")
    reconstruct.add_argument("--frames_dir", required=True)
    reconstruct.add_argument("--sequence", required=True)
    reconstruct.add_argument("--output", required=True)
    reconstruct.add_argument("--fps", type=float, default=30.0)
    reconstruct.add_argument("--save-frames-dir", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "pipeline":
        inliers_dir = f"{args.output_dir}/inliers"
        matches_path = f"{args.output_dir}/matches_{args.method}.npz"
        sequence_path = f"{args.output_dir}/sequence_{args.method}.npy"
        output_video = f"{args.output_dir}/reconstructed_video_{args.method}.mp4"
        save_frames_dir = f"{args.output_dir}/reconstructed_{args.method}"

        cluster_args = [
            "--input",
            args.input,
            "--output_dir",
            args.output_dir,
            "--clusters",
            str(args.clusters),
        ]
        if args.viz_tsne:
            cluster_args.append("--viz_tsne")

        cluster_main(cluster_args)
        match_main(
            [
                "--input_dir",
                inliers_dir,
                "--output",
                matches_path,
                "--descr",
                args.method,
            ]
        )
        sequence_main(
            [
                "--input",
                matches_path,
                "--output",
                sequence_path,
                "--alpha",
                str(args.alpha),
                "--descr",
                args.method,
            ]
        )
        reconstruct_main(
            [
                "--frames_dir",
                inliers_dir,
                "--sequence",
                sequence_path,
                "--output",
                output_video,
                "--fps",
                str(args.fps),
                "--save-frames-dir",
                save_frames_dir,
            ]
        )
        return

    if args.command == "cluster":
        cluster_args = [
            "--input",
            args.input,
            "--output_dir",
            args.output_dir,
            "--clusters",
            str(args.clusters),
            "--bins",
            str(args.bins),
            "--resize",
            str(args.resize[0]),
            str(args.resize[1]),
        ]
        if args.viz_tsne:
            cluster_args.append("--viz_tsne")
        cluster_main(cluster_args)
        return

    if args.command == "match":
        match_main(
            [
                "--input_dir",
                args.input_dir,
                "--output",
                args.output,
                "--descr",
                args.descr,
                "--ratio_thresh",
                str(args.ratio_thresh),
                "--alpha",
                str(args.alpha),
            ]
        )
        return

    if args.command == "sequence":
        sequence_main(
            [
                "--input",
                args.input,
                "--output",
                args.output,
                "--alpha",
                str(args.alpha),
                "--descr",
                args.descr,
            ]
        )
        return

    if args.command == "reconstruct":
        reconstruct_args = [
            "--frames_dir",
            args.frames_dir,
            "--sequence",
            args.sequence,
            "--output",
            args.output,
            "--fps",
            str(args.fps),
        ]
        if args.save_frames_dir is not None:
            reconstruct_args.extend(["--save-frames-dir", args.save_frames_dir])
        reconstruct_main(reconstruct_args)
        return

    parser.error(f"Unknown command: {args.command}")
