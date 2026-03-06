#!/bin/bash
set -e

METHOD=${1:-RESNET}

if [[ "$METHOD" != "AKAZE" && "$METHOD" != "RESNET" && "$METHOD" != "SIFT" && "$METHOD" != "COMBO" ]]; then
    echo "Error: Method must be AKAZE, RESNET, or SIFT"
    echo "Usage: $0 [METHOD]"
    echo "Example: $0 RESNET"
    exit 1
fi

echo "Starting video processing pipeline with method: $METHOD"

if [[ ! -f "corrupted_video.mp4" ]]; then
echo "Error: corrupted_video.mp4 not found!"
exit 1
fi

start_time=$(date +%s)
if ! PYTHONPATH=src python -m video_unscramble.cli pipeline --method "$METHOD" --input corrupted_video.mp4 --output-dir results --fps 24 --clusters 2 --alpha 0.5 --viz-tsne; then
echo "Error: Pipeline failed!"
exit 1
fi
end_time=$(date +%s)

echo "Total pipeline time: $((end_time - start_time)) seconds"
