#!/bin/bash

script_dir=$(cd "$(dirname "$0")" && pwd)

config_file="${script_dir}/../../../config.json"

k_values=$(jq -r '.mnist_addition_k' "$config_file")

read -r -a K <<< "$k_values"

probposs_transformation_methods_values=$(jq -r '.PiNeSy_probposs_transformation_methods' "$config_file")

read -r -a PROBPOSS_TRANSFORMATION_METHODS <<< "$probposs_transformation_methods_values"



for value in "${K[@]}"; do
    echo "mnist-addition-k=$value will be performed."
done

for probposs in "${PROBPOSS_TRANSFORMATION_METHODS[@]}"; do
    echo "by Pi-NeSy-$probposs."
done

NUMBER_RUNS=10


MAX_RETRIES=50
RETRY_DELAY=20

for k_id in "${!K[@]}"; do
    k=${K[$k_id]}
    for probposs_transformation_method in "${PROBPOSS_TRANSFORMATION_METHODS[@]}"; do
        for id in $(seq 1 $NUMBER_RUNS); do
            attempt=0
            while [ $attempt -lt $MAX_RETRIES ]; do
                ((attempt=attempt+1))
                echo "Attempt $attempt of $MAX_RETRIES for k=$k, ID=$id, ProbPoss=$probposs_transformation_method"
                
                # Create a file to be checked later
                STATUS_FILE="${script_dir}/../../../tmp/experiment_status_${k}_${id}_${probposs_transformation_method}.txt"
                echo "Experiment not completed" > "$STATUS_FILE"
                
                python3 ${script_dir}/mnistadd_single.py $k $id $probposs_transformation_method "$STATUS_FILE"
                
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
                echo "Failed after maximum retries for for k=$k, id=$id, ProbPoss=$probposs_transformation_method"
                rm -f "$STATUS_FILE" 
                exit -1;
            fi
        done
    done
done
