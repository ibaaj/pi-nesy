#!/usr/bin/env bash


cd "$(dirname "$0")"



# Function to check Python3 version (between 3.7 and 3.12)
check_python_version() {
    if command -v python3 &> /dev/null; then
        version=$(python3 --version 2>&1 | awk '{print $2}')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        
        if [ "$major" -eq 3 ] && [ "$minor" -ge 7 ] && [ "$minor" -le 12 ]; then
            echo "Python3 version $version is installed and supported."
        else
            echo "Python3 is installed but the version is not in the supported range (3.7 - 3.12). Installed version: $version"
            exit 1
        fi
    else
        echo "Python3 is not installed."
        exit 1
    fi
}



# Function to check if GCC is installed
check_gcc() {
    if command -v gcc &> /dev/null; then
        echo "GCC is installed. Version: $(gcc --version | head -n 1)"
    else
        echo "GCC is not installed."
        exit 1
    fi
}

# Function to check if CMake is installed
check_cmake() {
    if command -v cmake &> /dev/null; then
        echo "CMake is installed. Version: $(cmake --version | head -n 1)"
    else
        echo "CMake is not installed."
        exit 1
    fi
}

# Function to check Bash version
check_bash_version() {
    if command -v bash &> /dev/null; then
        version=$(bash --version | head -n 1 | awk '{print $4}')
        major=$(echo "$version" | cut -d. -f1)
        if [ "$major" -ge 5 ]; then
            echo "Bash version 5 or higher is installed. Version: $version"
        else
            echo "Bash version is less than 5. Installed version: $version"
            exit 1
        fi
    else
        echo "Bash is not installed."
        exit 1
    fi
}

# Function to check if pybind11 is available for CMake
check_pybind11() {
    pybind11_dir=$(cmake --find-package -DNAME=pybind11 -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST 2>/dev/null)
    if [ -n "$pybind11_dir" ]; then
        echo "pybind11 library is available for CMake."
    else
        echo "pybind11 library is not available for CMake. Ensure it is installed."
        exit 1
    fi
}



check_python_version
check_gcc
check_cmake
check_pybind11
check_bash_version

mkdir -p lib
mkdir -p data
mkdir -p results
mkdir -p aggregated_results
mkdir -p extern
mkdir -p tmp


choice=""
opt=""
nablaDir=""
buildDir=""


options_backend=("Apple Metal" "OpenCL" "CPU" "Cuda")

echo "Select which *_computation_py (nabla) library to install:"
echo "Options:"

i=1
for o in "${options_backend[@]}"; do
    echo "$i) $o"
    (( i++ ))
done

while true; do
    echo -n "Enter the number corresponding to your choice: "
    read -r choice

    # Check that 'choice' is a valid integer in range.
    if [ "$choice" -eq "$choice" ] 2>/dev/null && \
       (( choice >= 1 && choice <= ${#options_backend[@]} )); then

        # Store the user‐selected text in $opt
        opt="${options_backend[$(( choice - 1 ))]}"

        case "$opt" in
            "Apple Metal")
                echo "You chose Apple Metal. This requires a recent Apple computer."
                nablaDir="./src/nabla/metal/"
                buildDir="build"
                ;;
            "OpenCL")
                echo "You chose OpenCL. You need OpenCL C++ headers and OpenCL installed."
                nablaDir="./src/nabla/opencl/"
                ;;
            "CPU")
                echo "You chose CPU."
                nablaDir="./src/nabla/cpu/"
                ;;
            "Cuda")
                echo "You chose Cuda. You need Cuda installed."
                nablaDir="./src/nabla/cuda/"
                ;;
        esac
        break
    else
        echo "Invalid choice. Please enter a number between 1 and ${#options_backend[@]}."
    fi
done

echo "Selected computation library: $opt"
echo "nablaDir: $nablaDir"
echo "buildDir: $buildDir"




echo "Compiling $opt..."
if [[ $opt == "Apple Metal" ]]; then
    destinationDir="$nablaDir/lib/metal-cmake"
    url="https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip"
    mkdir -p "$destinationDir"
    echo "Downloading from $url"
    curl -o "$destinationDir/metal-cpp.zip" "$url"
    echo "Unzipping the file to $destinationDir"
    unzip "$destinationDir/metal-cpp.zip" -d "$destinationDir"
    rm "$destinationDir/metal-cpp.zip"
    # Create CMakeLists.txt file
    cat << EOF > "$destinationDir/CMakeLists.txt"
# Library definition
add_library(METAL_CPP
        \${CMAKE_CURRENT_SOURCE_DIR}/definition.cpp
        )

# Metal cpp headers
target_include_directories(METAL_CPP PUBLIC
        "\${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
        )

# Metal cpp library (linker)
target_link_libraries(METAL_CPP
        "-framework Metal"
        "-framework MetalKit"
        "-framework AppKit"
        "-framework Foundation"
        "-framework QuartzCore"
        )
EOF

    # Create definition.cpp file
    cat << EOF > "$destinationDir/definition.cpp"
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
EOF
    mkdir -p "$nablaDir/$buildDir"
    cd "$nablaDir/$buildDir"

    echo "Running cmake for $opt..."
    cmake -DPython3_EXECUTABLE=$(which python3) .. || { echo "Error: cmake failed for $opt"; exit 1; }

    echo "Running make for $opt..."
    make || { echo "Error: make failed for $opt"; exit 1; }
else
    cd "$nablaDir"
    
    echo "Running cmake for $opt..."
    cmake -DPython3_EXECUTABLE=$(which python3) . || { echo "Error: cmake failed for $opt"; exit 1; }

    echo "Running make for $opt..."
    make || { echo "Error: make failed for $opt"; exit 1; }
fi


# Find the .so file
soFile=$(find . -name "*.so" | head -n 1)

# Move the .so file to the program directory
if [[ $opt == "Apple Metal" ]]; then
    echo "Moving $soFile to the program directory..."
    mv "$soFile" ../../../../lib/
    cd ../../../..
else 
    mv "$soFile" ../../../lib/
    cd ../../..
fi

echo $(pwd)
# Step 5: Update config.json with the user's choice
echo "Updating config.json with your choice..."
# Detect the operating system (Linux or Darwin (macOS))
os="$(uname -s)"

# Apply the appropriate sed command based on the operating system
case "$os" in
    Linux*)
        sed -i "s/\"library\": \".*\"/\"library\": \"$opt\"/" config.json
        ;;
    Darwin*)
        sed -i '' "s/\"library\": \".*\"/\"library\": \"$opt\"/" config.json
        ;;
    *)
        echo "Unknown operating system. Manual edit required for config.json."
        ;;
esac

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

deactivate

echo "Installation and configuration of Pi-NeSy completed."


read -p "Do you want to generate visual-sudoku-puzzle classification data now? (Y/N): " answer

if [[ "$answer" =~ ^[Nn]$ ]]; then
    echo "Exiting."
    exit 0
fi



echo "Cloning linqs/visual-sudoku-puzzle-classification and generating datasets..."



cd extern

git clone https://github.com/linqs/visual-sudoku-puzzle-classification/


cd ..


python3 -m venv .venv_vspc
source .venv_vspc/bin/activate

python3 -m pip install -r ./extern/visual-sudoku-puzzle-classification/requirements.txt 


function sed_inplace() {
    case "$(uname)" in
        Darwin) sed -i '' "$@" ;;  # macOS requires an empty string argument with -i
        *) sed -i "$@" ;;          # Linux and other UNIX-like systems
    esac
}

# Extract values from config.json using jq
mnistsudoku_train_size=$(jq -r '.mnist_train_size_for_mnistsudoku' config.json)
mnistsudoku_validtest_size=$(jq -r '.mnist_validtest_size_for_mnistsudoku' config.json)



# Modify the generator script using the cross-platform 'sed_inplace' function
sed_inplace 's/readonly NUM_SPLITS='\''11'\''/readonly NUM_SPLITS='\''10'\''/' ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh

sed_inplace "s/NUM_TRAIN_PUZZLES='001 002 005 010 020 030 040 050 100'/NUM_TRAIN_PUZZLES='$mnistsudoku_train_size'/" ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh
sed_inplace "s/NUM_TEST_VALID_PUZZLE='100'/NUM_TEST_VALID_PUZZLE='$mnistsudoku_validtest_size'/" ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh


sed_inplace 's/OVERLAP_PERCENTS='\''0.00 0.50 1.00 2.00'\''/OVERLAP_PERCENTS='\''0.00'\''/' ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh
sed_inplace 's/STRATEGIES='\''simple r_split r_puzzle r_cell transfer'\''/STRATEGIES='\''simple'\''/' ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh
sed_inplace 's/SINGLE_DATASETS='\''mnist emnist fmnist kmnist'\''/SINGLE_DATASETS='\''mnist'\''/' ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh
sed_inplace 's/LARGE_DATASETS='\''emnist'\''/LARGE_DATASETS='\'''\''/' ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh
sed_inplace 's/ALL_DATASETS='\''mnist emnist fmnist kmnist mnist,emnist mnist,fmnist mnist,kmnist emnist,fmnist emnist,kmnist fmnist,kmnist emnist,fmnist,kmnist mnist,fmnist,kmnist mnist,emnist,fmnist mnist,emnist,fmnist,kmnist'\''/ALL_DATASETS='\'''\''/' ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh


# seed
sed_inplace $'/random\\.seed(seed)/a\\\n    print("seed:" + str(seed))' ./extern/visual-sudoku-puzzle-classification/scripts/generate-split.py


sed_inplace $'/^declare -A ALLOWED_DATASETS$/a\\
readonly SEED=42' \
  ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh

sed_inplace $'/^[[:space:]]*trap exit SIGINT$/a\\
    id=0' \
  ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh


sed_inplace $'/--strategy "\${strategy}"$/s/$/ \\\\/' \
  ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh

sed_inplace $'/--strategy "\${strategy}" \\\\/a\\
                                    --seed "\$((SEED + id))"' \
  ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh

sed_inplace 's/\(--seed "\$((SEED + id))"\)[[:space:]]*done/\1/' \
  ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh


sed_inplace $'/--seed "\\$((SEED + id))"/a\\
                                id=\\$((id+1))\\
                                done' \
  ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh


bash_version=$(bash --version | head -n 1 | awk '{print $4}' | cut -d '.' -f 1)

echo "Current Bash version: $bash_version"

bash ./extern/visual-sudoku-puzzle-classification/scripts/generate-data.sh

if [ $? -eq 0 ]; then
  echo "Visual sudoku generation executed successfully."
else
  echo "Visual sudoku generation execution failed."

  # Provide additional info if Bash version is less than 5
  if [ "$bash_version" -lt 5 ]; then
    echo "Note: Bash version is less than 5. The visual sudoku puzzle generator script may require Bash version >= 5 to work correctly."
  fi
fi

deactivate

exit 0
