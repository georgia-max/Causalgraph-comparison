# Set paths
$INPUT_FILE = "C:/Users/GL665/Desktop/CLDMakerRAG/datasets/Synthetic_sample_data.csv"
$OUTPUT_FOLDER = "C:/Users/GL665/Desktop/CLDMakerRAG/datasets/121224/11am"
$NUM_ITERATIONS = 1000 # Set the the number of iterations the LLM model is called. The more iterations the larger sample size. 

# Step 1: Generate synthetic data
Write-Host " Step 1. Generating synthetic data..."
python generate_synthetic_data.py --input $INPUT_FILE --output_folder $OUTPUT_FOLDER --num_iterations $NUM_ITERATIONS

# Step 2: Calculate metrics
Write-Host "Step 2. Calculating metrics..."
python get_metric_scores.py --input "$OUTPUT_FOLDER/Synthetic_sample_data_synthetic_data.csv" --output_folder $OUTPUT_FOLDER
