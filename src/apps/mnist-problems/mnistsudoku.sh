#!/bin/bash


script_dir=$(cd "$(dirname "$0")" && pwd)

base_dir="${script_dir}/../../../data"


config_file="${script_dir}/../../../config.json"

dimensions_values=$(jq -r '.mnist_sudoku_dimensions' "$config_file")

read -r -a dimensions <<< "$dimensions_values"

probposs_transformation_methods_values=$(jq -r '.PiNeSy_probposs_transformation_methods' "$config_file")

read -r -a PROBPOSS_TRANSFORMATION_METHODS <<< "$probposs_transformation_methods_values"


for dimension in "${dimensions[@]}"; do
    echo "Mnist-sudoku $dimension will be performed."
done


for probposs in "${PROBPOSS_TRANSFORMATION_METHODS[@]}"; do
    echo "by Pi-NeSy-$probposs."
done

MAX_RETRIES=50
RETRY_DELAY=20


find_split_directories() {
    local base_dir="$1"
    find "$base_dir" -type d -name "split::*"
}

sanitize_filename() {
    echo "$1" | sed 's/[\/:]/_/g' | tr -dc 'A-Za-z0-9_'
}

tmp_dir="${script_dir}/../../../tmp"
mkdir -p "$tmp_dir"


for dimension in "${dimensions[@]}"; do
    dimension_dir="$base_dir/$dimension"
    dimension_number="${dimension##*::}"  
    split_dirs=$(find_split_directories "$dimension_dir")
    for split_dir in $split_dirs; do
        for probposs_transformation_method in "${PROBPOSS_TRANSFORMATION_METHODS[@]}"; do
            attempt=0
            while [ $attempt -lt $MAX_RETRIES ]; do
                ((attempt=attempt+1))
                echo "Attempt $attempt of $MAX_RETRIES for Dimension=$dimension_number, SplitDir=$split_dir, ProbPoss=$probposs_transformation_method"
                
                sanitized_filename=$(sanitize_filename "$split_dir")
                STATUS_FILE="${tmp_dir}/experiment_status_${dimension_number}_${sanitized_filename}_${probposs_transformation_method}.txt"
                echo "Experiment not completed" > "$STATUS_FILE"
                
                
                python3 "${script_dir}/mnistsudoku_single.py" "$dimension_number" "$split_dir" "$probposs_transformation_method" "$STATUS_FILE"
                
                # Check if the Python script has updated the file
                if grep -q "experiment was ok" "$STATUS_FILE"; then
                    echo "Script executed successfully."
                    rm -f "$STATUS_FILE"
                    break
                else
                    echo "Script failed. Retrying in $RETRY_DELAY seconds..."
                    sleep $RETRY_DELAY
                fi
                
                if [ $attempt -eq $MAX_RETRIES ]; then
                    echo "Maximum number of retries reached. Moving to next set."
                fi
            done
            
            if [ $attempt -eq $MAX_RETRIES ]; then
                echo "Failed after maximum retries for Dimension=$dimension_number, SplitDir=$split_dir, ProbPoss=$probposs_transformation_method"
                rm -f "$STATUS_FILE"  
                exit -1;
            fi
        done
    done
done
