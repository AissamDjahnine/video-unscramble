import argparse
import glob
import os

import cv2
import numpy as np

from .core import reconstruct_video

def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruct video from ordered frames.")
    parser.add_argument("--frames_dir", required=True, help="Directory containing filtered frame images (.jpg).")
    parser.add_argument("--sequence", required=True, help=".npy file containing the frame order (list of indices).")
    parser.add_argument("--output", required=True, help="Path to the reconstructed video file.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate of the output video.")
    parser.add_argument("--save-frames-dir", default=None,help="directory to save ordered frames as JPEGs.")
    args = parser.parse_args(argv)

    pattern = os.path.join(args.frames_dir, "*.jpg")
    frame_paths = sorted(glob.glob(pattern))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in {args.frames_dir}")

    frames = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read frame image: {path}")
        frames.append(img)

    sequence = np.load(args.sequence).tolist()
    if not sequence:
        raise ValueError(f"Sequence file {args.sequence} is empty.")

    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)
        for idx, i in enumerate(sequence):
            out_path = os.path.join(args.save_frames_dir, f"{idx:04d}.jpg")
            cv2.imwrite(out_path, frames[i])
        print(f"Saved ordered frames to folder: {args.save_frames_dir}")

    print(f"Reconstructing video from {len(sequence)} ordered frames...")
    reconstruct_video(frames, sequence, args.output, fps=args.fps)
    print(f"Reconstructed video saved to: {args.output}")

if __name__ == "__main__":
    main()
