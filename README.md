# Π-NeSy

Π-NeSy is a research software developed for performing neuro-symbolic computations. See [ArXiv Paper](https://arxiv.org/abs/2504.07055).

In Π-NeSy, high-performance processing is carried out using a GPU or multithreaded CPU.  The program supports Apple Metal, CPU, OpenCL, and CUDA, depending on the system's capabilities and user preferences.

## Requirements

- **OS:** macOS or Linux  
- **Bash:** Version 5 or higher  
- **CMake**  
- **GCC**  
- **Python:** Versions 3.9 to 3.12  
- **Pybind11:** Compatible with your Python version
- **Apple Metal** or **OpenCL** or **CUDA** for high-performance processing

## Installation

Clone the repository:
```bash
git clone https://github.com/ibaaj/pi-nesy
```

Run the installation script:

```bash
./install.sh
```

The script performs the following steps:

1. **Dependency Check:** Verifies that all required dependencies are installed.
2. **Backend Selection:** Prompts you to choose a computational backend (Apple Metal, CPU, OpenCL, or CUDA). Based on your selection:
   - **Apple Metal:**  
     - Downloads the Apple C++ Framework for Apple Metal (execution implies acceptance of the license).  
     - Compiles a Python 3 library named `metal_computation_py` in the `./lib` directory.  
     - *Requirements:* A recent MacBook Pro with the Apple Metal Framework, CommandLine Tools installed, and xcrun activated for compiling shaders.
   - **OpenCL:**  
     - Compiles a Python 3 library named `opencl_computation_py` in the `./lib` directory.  
     - *Requirements:* OpenCL 1.2.
   - **CUDA:**  
     - Compiles a Python 3 library named `cuda_computation_py` in the `./lib` directory.  
     - *Requirements:* An Nvcc compiler and Nvidia hardware. You may need to modify the line `set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_75 -Xcompiler -fPIC")` in the CUDA library's CMakeLists.txt to match your Nvidia GPU architecture (default is set to sm_75 for testing).
   - **CPU:**  
     - Compiles a Python 3 library named `cpu_computation_py` in the `./lib` directory.
3. **Configuration Update:** Updates your selection in `config.json`.
4. **Python Dependencies:** Creates a virtual environment and installs the required Python packages (e.g., `torch`, `torchvision`, `numpy`, `scikit-learn`, etc.) from `requirements.txt` using pip.
5. **Dataset Generation:**  
   - Generates the MNIST-Sudoku dataset using [linqs/visual-sudoku-puzzle-classification](https://github.com/linqs/visual-sudoku-puzzle-classification) based on parameters specified in `config.json` (e.g., `mnist_train_size_for_mnistsudoku` (this number indicates both the count of correct train puzzles and the count of incorrect train puzzles) for training and `mnist_validtest_size_for_mnistsudoku` for testing/validation (this number indicates both the count of correct test puzzles and the count of incorrect test puzzles)).  
   - Modifies the `generated-data.sh` script within the cloned repository to adjust the data splits correctly (10 splits, dimensions of 4x4 and 9x9, corruption chance of 0.5, with no overlap).

Ensure that the installation script runs without errors; otherwise, verify your dependencies.

*Note:* The experiments described in the article were conducted using Apple Metal on a Mac M2 under macOS Sequoia 15.2 using python 3.12.2. The other backends (OpenCL, CPU, CUDA) have been tested on synthetic data, and each library includes its own test program.


## Running Experiments

To run experiments with **Π-NeSy**, simply execute:

```bash
./run.sh
```

### Single MNIST-Additions-k Experiment

If you want to run a single MNIST-Additions-k experiment, follow these steps:

   ```bash
   source .venv/bin/activate
   python3 src/apps/mnist-problems/mnistadd_single.py k id probposs_transformation_method file_txt
   deactivate
   ```

**Parameter Details:**

- **k:** The number of digits in the input numbers.
- **id:** The experiment index (choose any number).
- **probposs_transformation_method:** Use `1` for the antipignistic method or `2` for the method obeying the minimum specificity principle.
- **file_txt:** A text file to log when the experiment is complete.

This experiment is performed using the parameters defined in config.json.

### Additional Experiment Configurations

- **Without Thresholds:**  
  Set `"use_thresholds"` to `false` in `config.json` and then run:
  ```bash
  ./run.sh
  ```

- **Without Possibilistic Learning:**  
  Set `"perform_possibilistic_learning"` to `false` in `config.json` and then run:
  ```bash
  ./run.sh
  ```

- **Using DeepSoftLog's CNN:**  
  To run experiments using the convolutional neural network of [DeepSoftLog](https://github.com/jjcmoon/DeepSoftLog), update `config.json` with:
  ```json
  {
    "same_NN_as_DeepSoftLog": true,
  }
  ```

### Configurable Parameters in `config.json`

- **MNIST-Addition Dataset Sizes:**  
  Modify the following parameters to adjust the training, validation, and test sizes:
  - `"mnist_train_size_for_mnistadd"`
  - `"mnist_valid_size_for_mnistadd"`
  - `"mnist_test_size_for_mnistadd"`

- **MNIST-Sudoku Dataset Dimensions:**
  Modify `"mnist_sudoku_dimensions"`.
  
- **MNIST-Sudoku Dataset Sizes:**  
  Adjust the sizes by modifying:
  - `"mnist_train_size_for_mnistsudoku"` (this number indicates both the count of correct train puzzles and the count of incorrect train puzzles) 
  - `"mnist_validtest_size_for_mnistsudoku"` (this number indicates both the count of correct test puzzles and the count of incorrect test puzzles) 

- **MNIST-Addition-k Values:**  
  To experiment with different values of `k`, modify `"mnist_addition_k"` (the default is set to `100 15 4 2 1`).

The `config.json` file also contains hyperparameters for Π-NeSy's neural network, settings for possibilistic learning, and directory paths for data, temporary files, and results. **Do not modify** the following parameters:
- `"library"` — Specifies your chosen backend.
- `"problem_studied"` — Indicates the current neuro-symbolic problem being executed.


See in `./config_files` for config examples.


## Parse and aggregate the results

To parse and aggregate the results obtained with Π-NeSy, you can run:

```bash
./parse_results.sh
```
Then go to ./aggregated_results/.


##  Cleaning

If you want to revert to the initial state, you can run:

```bash
./cleaning.sh
```

This script removes all results and installation files, restoring the initial state.

## Known Issue

Python scripts that load the backend libraries `*_computation_py` may experience a segmentation fault upon closing. This issue is related to deallocation within pybind11 (investigated with lldb). It occurs after the experiments are complete and does not impact the experiment results.

Update (04-11-2025): The bug is linked to the pybind11 global static object (``py::float_ float DEFAULT_EPSILON``) in the main.cpp file in the `*_computation_py` backend libraries. Replacing it with ``constexpr float DEFAULT_EPSILON`` seems to avoid the bug (but it is not implemented since the experiments reported in the article were performed with ``py::float_ float DEFAULT_EPSILON``).


## Citation 

To cite Π-NeSy, use:

```
@article{baaj2025pinesy,
      title={$\Pi$-NeSy: A Possibilistic Neuro-Symbolic Approach}, 
      author={Ismaïl Baaj and Pierre Marquis},
      year={2025},
      eprint={2504.07055},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.07055}, 
}
```

##  License

Π-NeSy is licensed under the Apache License 2.0. For more details, see the LICENSE file.
