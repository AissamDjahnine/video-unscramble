#!/bin/bash
set -e

METHODS=(RESNET)
INPUT_DIR="videos"
RESULTS_DIR="results_multiple_videos"

run_pipeline_for_video() {
  local input_file=$1
  local METHOD=$2

  echo "=================================="
  local filename=$(basename "$input_file")
  local base_name="${filename%_tampered.mp4}"
  local output_file="$INPUT_DIR/${base_name}_reconstructed_${METHOD}.mp4"
  local method_results="$RESULTS_DIR/${base_name}/$METHOD"
  mkdir -p "$method_results"

  echo "Processing Input: $input_file"
  echo "Output: $output_file"
  local start_time=$(date +%s)

  PYTHONPATH=src python -m video_unscramble.cli pipeline \
    --method "$METHOD" \
    --input "$input_file" \
    --output-dir "$method_results" \
    --fps 24 \
    --clusters 2 \
    --alpha 0.5
  local end_time=$(date +%s)

  local total_time=$((end_time - start_time))
  echo "Total pipeline time for $filename with $METHOD: $total_time seconds ($((total_time / 60))m $((total_time % 60))s)"

  echo "Cleaning up temporary files..."
  rm -rf "$method_results/reconstructed_${METHOD}"
  rm -f "$method_results/matches.npz" "$method_results/sequence.npy"
  mv "$method_results/reconstructed_video_${METHOD}.mp4" "$output_file"

  echo "=================================="
}

if [ ! -d "$INPUT_DIR" ]; then
  echo "Directory $INPUT_DIR not found!"
  exit 1
fi

tampered_files=($(find "$INPUT_DIR" -name "*_tampered.mp4" -type f))
if [ ${#tampered_files[@]} -eq 0 ]; then
  echo "No *_tampered.mp4 files found in $INPUT_DIR"
  exit 1
fi

for METHOD in "${METHODS[@]}"; do
  echo "=============================="
  echo "Running pipeline with method: $METHOD"
  echo "=============================="
  for input_file in "${tampered_files[@]}"; do
    run_pipeline_for_video "$input_file" "$METHOD"
  done
done

echo "All videos processed successfully for all methods!"
