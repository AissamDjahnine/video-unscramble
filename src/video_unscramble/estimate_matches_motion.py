import argparse
import glob
import os
import cv2
import numpy as np
from .core import (
    combine_global_local,
    compute_feature_matches_AKAZE,
    compute_feature_matches_ResNet_spatial,
    compute_feature_matches_SIFT,
)

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute pairwise frame similarity using various descriptors"
    )
    parser.add_argument('--input_dir', required=True, help='Directory of .jpg frames')
    parser.add_argument('--output', required=True, help='Output .npz file')
    parser.add_argument('--descr', default='RESNET', choices=['AKAZE','SIFT','ORB','RESNET','COMBO'], help='Descriptor to use')
    parser.add_argument('--ratio_thresh', type=float, default=0.75, help='Lowe ratio for ResNet spatial')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for global/local combine')
    args = parser.parse_args(argv)

    frame_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg files in {args.input_dir}")
    frames = [cv2.imread(p) for p in frame_paths]

    if args.descr == 'RESNET':
        match_matrix, motion_matrix = compute_feature_matches_ResNet_spatial(frames, ratio_thresh=args.ratio_thresh)
    elif args.descr == 'AKAZE':
        match_matrix, motion_matrix = compute_feature_matches_AKAZE(frames)
    elif args.descr == 'SIFT':
        match_matrix, motion_matrix = compute_feature_matches_SIFT(frames)
    elif args.descr == 'COMBO':
        g_scores, g_motion = compute_feature_matches_ResNet_spatial(frames)
        l_counts, l_motion = compute_feature_matches_AKAZE(frames)
        
        match_matrix = combine_global_local(g_scores, l_counts, args.alpha)
        motion_matrix = combine_global_local(g_motion, l_motion, args.alpha)
    else:
        print(f"Unsupported descriptor: {args.descr}")
        return
    np.savez_compressed(args.output, matches=match_matrix, motion=motion_matrix)
    print(f"Saving matches & motion matrices to {args.output}")

if __name__ == '__main__':
    main()
