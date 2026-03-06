import argparse
import numpy as np

from .core import (
    build_score_matrix,
    find_best_sequence,
)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute smart frame sequence (greedy + smoothing)")
    parser.add_argument("--input", required=True, help=".npz file with 'matches' and optionally 'motion'")
    parser.add_argument("--output", required=True, help=".npy file to save sequence")
    parser.add_argument("--alpha", type=float, default=0.5, help="Motion penalty weight (default 0.5)")
    parser.add_argument("--descr", type=str, default="AKAZE", help="Feature descriptor to use: AKAZE, SIFT, RESNET, or COMBO")
    
    args = parser.parse_args(argv)

    data = np.load(args.input)
    match_matrix = data["matches"]
    motion_matrix = data["motion"] if "motion" in data else np.zeros_like(match_matrix)

    score_matrix = build_score_matrix(match_matrix, motion_matrix, alpha=args.alpha)
    sequence = find_best_sequence(
        score_matrix=score_matrix,
        penalty_weight=35,
        lookahead_weight=0.5,
        max_starts=min(8, score_matrix.shape[0]),
    )
    if not sequence:
        raise ValueError("Failed to recover a valid frame sequence.")

    np.save(args.output, np.array(sequence, dtype=np.int32))
    print(f"Saved reordered sequence of {len(sequence)} frames to {args.output}")

if __name__ == "__main__":
    main()
