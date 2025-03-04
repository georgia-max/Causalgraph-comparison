# Measuring Similarity in Causal Graphs: A Framework for Semantic and Structural Analysis 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository investigates how to compare causal graphs—diagrams that map out how factors influence each other—when those graphs vary in structure or in how variables are named. Drawing upon synthetic datasets created by AI to simulate multiple independent attempts at modeling the same complex system, it reviews a range of comparison methods. The findings underscore that each metric has strengths and weaknesses depending on whether the priority is capturing meaningful relationships among variables or the particular ways those relationships are structured.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Getting Started](#start)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)


## Installation

1. Clone the repository:

    ```bash
    https://github.com/georgia-max/Causalgraph-comparison.git
    ```
2. Create a a virtual environment in your terminal:
   ```
   python3.11 -m venv myenv
   ```
2. Activate virtual environment in your terminal:
   - for mac 
    ```
    source myenv/bin/activate
    ```
   - for windows
     ```
     .\myenv\Scripts\Activate
     ```
4. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```


## Usage

To generate synthetic data and get the metrics scores for each causal graph: 

- For Windows:
    1. Go to `run_pipeline.ps1` and change the INPUT_PATH and OUTPUT_PATH variables to the path of your choice, specify the number of iterations you want to run. 
    2. Run `run_pipeline.ps1` in powershell. 
    3. In the end, you will have two files: 
        - `Synthetic_sample_data_synthetic_data.csv`: contains the synthetic data for each CLD.
        - `metrics_results.xlsx`: contains the metrics scores for each CLD.

- For Mac:
    1. Go to `run_pipeline.sh` and change the INPUT_PATH and OUTPUT_PATH variables to the path of your choice, specify the number of iterations you want to run. 
    2. Run `run_pipeline.sh` in powershell. 
    3. In the end, you will have two files: 
        - `Synthetic_sample_data_synthetic_data.csv`: contains the synthetic data for each CLD.
        - `metrics_results.xlsx`: contains the metrics scores for each CLD.

## Getting Started

	•	Installation: Clone this repository and install all Python dependencies (see requirements.txt).
	•	Documentation: Refer to the docs/ folder for details on each comparison method and guidelines for extending or customizing the metrics.
	•	Examples: The examples/ folder contains Jupyter notebooks demonstrating how to generate synthetic data, run comparisons, and visualize results.
## Examples

TODO: Add examples of how to use the package.
