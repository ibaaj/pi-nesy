#!/bin/bash


directories=(
    "./src/nabla/cpu"
    "./src/nabla/metal"
    "./src/nabla/opencl"
    "./src/nabla/cuda"
    "./src/nabla/cpu/build"
    "./src/nabla/metal/build"
    "./src/nabla/opencl/build"
    "./src/nabla/cuda/build"
)


clear_cmake() {
    if [[ -d $1 ]]; then
        echo "Clearing CMake-related content in directory $1"
        find $1 -name 'CMakeCache.txt' -delete
        find $1 -name 'cmake_install.cmake' -delete
        find $1 -name 'Makefile' -delete
        find $1 -name '*.so' -delete
        find $1 -name '*.a' -delete
        find $1 -type d -name 'CMakeFiles' -exec rm -rf {} +
        find $1 -type f -perm +111 -delete
    else
        echo "Directory does not exist: $1"
    fi
}


for dir in "${directories[@]}"; do
    clear_cmake $dir
done

rm -rf ./src/nabla/metal/build/*
rm -rf ./src/nabla/metal/lib/*
rm -rf ./src/nabla/metal/shaders/*.ir
rm -rf ./src/nabla/metal/shaders/*.metallib
rm -rf ./src/nabla/metal/shaders/*.dummy

echo "Removing all files in ./tmp/, ./results/, ./lib/, ./extern/, and ./data/"
rm -rf ./tmp/*
rm -rf ./results/*
rm -rf ./lib/*
rm -rf ./extern/*
rm -rf ./data/*
rm -rf ./aggregated_results/*


echo "Removing the shaders directory and .venv"
rm -rf ./shaders
rm -rf ./.venv
rm -rf ./.venv_vspc


echo "Removing all __pycache__ directories in src/*"
find ./src/ -type d -name '__pycache__' -exec rm -rf {} +


echo "Removing all .DS_Store files from all subdirectories"
find . -name '.DS_Store' -type f -delete

echo "Removing experiments logs."
rm out-mnist-add.txt
rm out-mnist-sudoku.txt
rm error-mnist-sudoku.err
rm error-mnist-add.err



CONFIG_FILE="config.json"
jq '.DeepSoftLog_checkpointmnistaddfile = "" |
    .DeepSoftLog_mnistadd_validloader_file = ""' "$CONFIG_FILE" > temp.json && mv temp.json "$CONFIG_FILE"


echo "Cleanup completed."
