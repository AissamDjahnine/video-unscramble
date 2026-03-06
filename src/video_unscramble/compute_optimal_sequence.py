import argparse
import numpy as np

from .core import (
    smooth_temporal_coherence,
    remove_weak_links,
    total_sequence_score,
    two_opt,
    greedy_with_lookahead,
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

    score_matrix = match_matrix - args.alpha * motion_matrix

    score_matrix[~np.isfinite(score_matrix)] = -1e6
    
    sequence = greedy_with_lookahead(
        start_idx=np.argmax(score_matrix.sum(axis=1) - score_matrix.sum(axis=0)),
        score_matrix=score_matrix,
        penalty_weight=35,
        lookahead_weight=0.5
    )
    
    sequence = two_opt(sequence, score_matrix)
    sequence = smooth_temporal_coherence(sequence, score_matrix)
    sequence = remove_weak_links(sequence, score_matrix)

    np.save(args.output, np.array(sequence[::-1], dtype=np.int32))
    print(f"Saved reordered sequence of {len(sequence)} frames to {args.output}")

if __name__ == "__main__":
    main()
