import argparse
import os
from .core import (
    extract_frames,
    compute_histogram_features,
    cluster_frames,
    filter_clusters,
    save_frames,
)

from .visualization import generate_plotly_visualization, load_images_from_folder

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=("Extract, cluster and save frames from a video."))
    parser.add_argument("--input", required=True, help="Path to the input video file.",)
    parser.add_argument("--output_dir", required=True, help="Directory to save results. Will be created if it doesn't exist.",)
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters to use for K‑means (default: 2).",)
    parser.add_argument("--bins", type=int, default=64, help="Number of bins for grayscale histograms (default: 64).",)
    parser.add_argument("--viz_tsne", action="store_true", help="Enable t-SNE visualization of clustered frames.")    
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=(256, 256), help="resize (width X height) before histogram computation.")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    orig_dir = os.path.join(args.output_dir, "original")
    inlier_dir = os.path.join(args.output_dir, "inliers")
    outlier_dir = os.path.join(args.output_dir, "outliers")

    print(f"Reading frames from {args.input}…")
    frames = extract_frames(args.input)
    image_paths = load_images_from_folder(orig_dir)
    
    n_frames = len(frames)
    if n_frames == 0:
        print("[ERROR] No frames were extracted from the video. Exiting.")
        return
    print(f"Extracted {n_frames} frames.")

    resize_dim = None if (args.resize[0] == 0 or args.resize[1] == 0) else tuple(args.resize)
    print("Computing histogram features…")
    features = compute_histogram_features(frames, bins=args.bins, resize=resize_dim)
    print(f"Computed features for {n_frames} frames.")

    labels = cluster_frames(
        features,
        n_clusters=args.clusters,
        max_iter=700,
    )
    if args.viz_tsne:
        generate_plotly_visualization(
            frames,
            labels,
            os.path.join(args.output_dir, "clustering_tsne.html"),
            image_paths=image_paths,
        )

    # Separate into dominant cluster (inliers) and outliers
    inliers, outliers, inlier_idx, outlier_idx, dominant_label = filter_clusters(frames, labels)
    print(
        f"Dominant cluster label: {dominant_label if dominant_label is not None else 'None'} "
        f"– {len(inliers)} inliers, {len(outliers)} outliers."
    )

    save_frames(frames, list(range(n_frames)), orig_dir)
    print(f"Saved {n_frames} original frames to {orig_dir}")

    # Save inliers and outliers
    print(f"Saving inliers {len(inliers)} inliers to {inlier_dir}")
    save_frames(inliers, inlier_idx, inlier_dir)

    print(f"Saving outliers {len(outliers)} outliers to {outlier_dir}")
    save_frames(outliers, outlier_idx, outlier_dir)

if __name__ == "__main__":
    main()
