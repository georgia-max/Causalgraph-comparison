# Measuring Similarity in Causal Graphs: A Framework for Semantic and Structural Analysis 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

#TODO update the overview section 

Causal graphs are diagrams that help us understand how different factors are connected and influence one another, making them an essential tool for studying complex problems like climate change, public health, and social dynamics. However, comparing these graphs—especially when they are created using artificial intelligence—can be tricky. Differences in variable names, graph structures, or the way relationships are represented can make it hard to assess whether two graphs are saying the same thing or something completely different. In this study, we explore a variety of metrics to compare causal graphs, focusing on their structure and the meaning of their variable names. We tested these methods using synthetic datasets created by artificial intelligence, simulating what might happen if hundreds of people independently tried to map out the same complex system. Our results show that no single method works perfectly in all situations. Some methods are better at identifying meaningful connections between variables, while others focus on how the graphs are built. This work highlights the need for combining different approaches to more accurately compare causal graphs, paving the way for better tools to analyze and understand complex systems.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
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


## Examples

TODO: Add examples of how to use the package.
