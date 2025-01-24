#!/bin/bash
#./run_pipeline.sh

# Set paths
INPUT_FILE="/Users/gl665/Documents/Code_Repo/CLDMakerRAG/datasets/121124/Synthetic_sample_data.csv"
OUTPUT_FOLDER="/Users/gl665/Documents/Code_Repo/CLDMakerRAG/datasets/121124/1156pm"
NUM_ITERATIONS=1

# Step 1: Generate synthetic data
echo "Generating synthetic data..."
python generate_synthetic_data_v3.py \
    --input $INPUT_FILE \
    --output_folder $OUTPUT_FOLDER \
    --num_iterations $NUM_ITERATIONS

# Step 2: Calculate metrics
echo "Calculating metrics..."
python get_metric_scores.py \
    --input "${OUTPUT_FOLDER}/Synthetic_sample_data_synthetic_data.csv" \
    --output_folder $OUTPUT_FOLDER
