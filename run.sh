#!/bin/bash


source .venv/bin/activate

echo $(python3 --version)


echo "Starting experiments..."



bash src/apps/mnist-problems/mnistadd.sh 1> out-mnist-add.txt 2> error-mnist-add.err

bash src/apps/mnist-problems/mnistsudoku.sh 1> out-mnist-sudoku.txt 2> error-mnist-sudoku.err


deactivate
echo "Experiments completed."

exit 0