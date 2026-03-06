import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for matching/motion")
    parser.add_argument("--method", type=str, required=True, help="Method name (e.g., RESNET, SIFT...)")
    args = parser.parse_args()
    
    method = args.method.upper()
    npz_filename = f"results/matches_{method}.npz"

    if not os.path.exists(npz_filename):
        print(f"Error: File '{npz_filename}' not found.")
        return

    data = np.load(npz_filename)
    matches = data["motion"]
    medians = data["matches"]

    output_dir = f"results/heatmaps/{method}"
    os.makedirs(output_dir, exist_ok=True)

    # Match Count Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(matches, aspect="equal", cmap="viridis")
    plt.title(f"{method} - Match Counts")
    plt.colorbar(label="# of Matches")
    plt.xlabel("Frame index")
    plt.ylabel("Frame index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"matches_heatmap_{method}.png"), dpi=200)
    plt.close()

    # Median Displacement Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(medians, aspect="equal", cmap="magma")
    plt.title(f"{method} - Median Keypoint Displacement")
    plt.colorbar(label="Pixels")
    plt.xlabel("Frame index")
    plt.ylabel("Frame index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"medians_heatmap_{method}.png"), dpi=200)
    plt.close()

    print(f"Saved heatmaps to {output_dir}/")

if __name__ == "__main__":
    main()
