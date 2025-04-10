#!/bin/bash

source .venv/bin/activate
echo "$(python3 --version)"


python3 src/apps/mnist-problems/parse_results_and_logs/mnistadd_inference-and-learning-time.py
python3 src/apps/mnist-problems/parse_results_and_logs/mnistadd_parse-results.py

python3 src/apps/mnist-problems/parse_results_and_logs/mnistadd_possibilistic-learning-parse-stats.py
python3 src/apps/mnist-problems/parse_results_and_logs/mnistadd_possibilistic-learning-parse-thresholds-err-mnist-add.py


python3 src/apps/mnist-problems/parse_results_and_logs/mnistsudoku_inference-and-learning-time.py
python3 src/apps/mnist-problems/parse_results_and_logs/mnistsudoku_parse-results.py

python3 src/apps/mnist-problems/parse_results_and_logs/mnistsudoku_possibilistic-learning-parse-thresholds-err-mnist-sudoku.py
python3 src/apps/mnist-problems/parse_results_and_logs/mnistsudoku_possibilistic-learning-parse-stats.py


deactivate
exit 0
