# # Generating synthetic data using LLM
# - Creation: 2024-11-22
# - Update: 2024-12-11
# - author: Georgia
# - description: generating synthetic data using LLM with few shots, using the originial LLM model instead of chat models.
#  - Version: 0.3
# - testing if we can generate a better normal distribution of data
# - License: MIT
#

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from my_key import MY_OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = MY_OPENAI_API_KEY


# Suppress the specific FutureWarning
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning
)


def main():
    args = parse_args()

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Setup output paths
    name = os.path.splitext(os.path.basename(args.input))[0]
    output_file = f"{name}_synthetic_data.csv"
    output_path = os.path.join(args.output_folder, output_file)

    # Read sample data
    df = pd.read_csv(args.input, encoding='utf-8')

    # Process data and generate synthetic samples
    print("Numbers of iterations: ", args.num_iterations)
    output_dict = generate_synthetic_data(df, args.num_iterations)

    # Convert to DataFrame and save
    df_synthetic = process_output(output_dict)
    df_synthetic.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic CLD data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input sample CSV file')
    parser.add_argument('--output_folder', type=str,
                        required=True, help='Output folder path')
    parser.add_argument('--num_iterations', type=int, default=2,
                        help='Number of iterating on the LLM to generate')
    return parser.parse_args()


def check_syntax(r: str):
    r = r.strip()
    nq = len(re.findall(re.compile(r'\"'), r))
    nb = len(re.findall(re.compile(r'\[|\]'), r))
    if nq % 2 != 0 or nb % 2 != 0:
        return '\n'
    # TODO check if missing a bracket [ or ] add it to the sentence.
    elif nb == 0 and '\"' in r:
        return '\n'
    r = re.sub(re.compile('arrowhead='), 'arrowhead =', r)
    return r


def create_example_list(df):
    """Convert DataFrame to list of examples with proper formatting"""

    dict_df = df[['input', 'output', 'Score']].to_dict(orient='index')

    example_list = []

    for key, values in dict_df.items():
        cld_examples = {}

        # add { before the { sign
        digraph = str(values['output'])
        # add { before the { sign
        digraph = digraph.replace("{", "{{")
        digraph = digraph.replace("}", "}}")

        cld_examples["input"] = str(values['input'])
        cld_examples["output"] = digraph
        cld_examples["Score"] = str(values['Score'])

        example_list.append(cld_examples)

    return example_list


def create_prompt_templates(example_list):
    """ Create and return the few-shot prompt template """

    # Define the example prompt
    example_prompt = PromptTemplate(
        input_variables=["input", "output", "Score"],
        template="""
        Given the following dynamic hypothesis:
        {input} 
        
        Task: Generate a causal loop diagram (CLD) in DOT format that represents the causal relationships.
        
        Guidelines:
        1. Format: Use DOT digraph syntax
        2. Variables: Extract key variable names as nodes
        3. Relationships:
            - Positive influence: [arrowhead = vee]
            - Negative influence: [arrowhead = tee]
        4. Quality Criteria:
            - Correct variable naming
            - Valid causal links
   
        
        Generated CLD:
        {output}
    
        The ground truth CLD is:  
        
        "digraph [
        "Population" -> "Net Increase" [arrowhead=vee]
        "Net Increase" -> "Population" [arrowhead=vee]
        "Population" -> "Resources per Capita" [arrowhead=tee]
        "Resources per Capita" -> "Fractional Net Increase" [arrowhead=vee]
        "Fractional Net Increase" -> "Net Increase" [arrowhead=vee]
        "Resources per Capita" -> "Carrying Capacity" [arrowhead=vee]
        "Population" -> "Carrying Capacity" [arrowhead=tee]
        "Carrying Capacity" -> "Resources per Capita" [arrowhead=vee]
        ]"
        
        The similairity score between the Generated and the ground truth CLD considers both semantic similarity of the 
        variable names and the graph strucuture similiary considering both the vertex and edge labels : {Score}
       
        """
    )

    # Define Few-Shot Prompt Template
    SYNTHETIC_FEW_SHOT_PREFIX = """
    
    You are a helpful assistant that generates synthetic data of causal loop diagrams based on given hypothesis. 

    """
    SYNTHETIC_FEW_SHOT_SUFFIX = """
        Given hypothesis: {input}

        Generate as many CLDs as possible representing different student understanding levels, where the examples are uniformly distributed across:
        - High understanding (Score > 0.7)
        - Medium understanding (0.2 < Score < 0.7)
        - Low understanding (Score < 0.2)

        Each CLD should:
        - Be syntactically valid DOT format
        - represent the causal relationships in the hypothesis
        - Vary in complexity and completeness 
        - consider the negative and positive causal relationships between each link

        Format each response as:
        1.
        Digraph: <DOT code>
        Score: <similarity_score>
        2.
        Digraph: <DOT code>
        Score: <similarity_score>
        3.
        Digraph: <DOT code>
        Score: <similarity_score>

    """

    few_shot_prompt = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=example_list,
        example_prompt=example_prompt,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["input"]
    )

    return example_prompt, few_shot_prompt


def generate_single_response(few_shot_prompt, subject, temperature):

    llm = ChatOpenAI(model="gpt-4o", temperature=temperature)

    chain = few_shot_prompt | llm
    response = chain.invoke({"input": subject})
    return response.content


def generate_synthetic_data(df, NUM_OF_ITERATIONS):

    df['generated_cld_DOT'] = df['generated_cld_DOT'].astype(str)
    df['ID'] = df['ID'].astype(int)

    df = df.rename(columns={'generated_cld_DOT': 'output', 'text': 'input'})
    # parse all the \n in the input and output
    df['input'] = df['input'].apply(lambda x: x.replace('\n', ''))
    df['output'] = df['output'].apply(lambda x: x.replace('\n', ''))

    # Create examples and prompts
    example_list = create_example_list(df)
    example_prompt, few_shot_prompt = create_prompt_templates(example_list)

    # The dynamic hypothesis
    subject = """
    Population grows based on the net increase,
    which is influenced by the fractional net increase—the rate at which
    the population grows per person. This rate isn’t constant; it depends on
    the resources available per person. When resources are plentiful,
    the rate remains high, allowing growth to continue.
    However, as resources are stretched thinner with a growing population, the rate slows down,
    reducing the pace of growth. Additionally, the growing population depletes and degrades these resources,
    creating a cycle where fewer resources lead to slower growth, ultimately limiting how much the population can expand.
    """

    # Generate responses
    output_dict = {}

    pbar = tqdm(total=NUM_OF_ITERATIONS, desc="Generate synthetic data")

    for i in range(NUM_OF_ITERATIONS):
        temperature = np.random.uniform(0.8, 1.2)
        output_dict[i] = generate_single_response(
            few_shot_prompt, subject, temperature)
        pbar.update(1)

    pbar.close()
    return output_dict


def process_output(output_dict):

    pattern = re.compile(r"""
    \d+\.\s+              # Match the number and period
    Digraph:\s+```\s+     # Match 'Digraph:' and opening triple backticks
    (.*?)                 # Capture the Digraph content (non-greedy)
    \s+```\s+Score:\s+    # Match closing triple backticks and 'Score:'
    ([0-9.]+)             # Capture the Score (decimal number)
    """, re.DOTALL | re.VERBOSE)

    # Initialize an empty DataFrame
    df_synthetic = pd.DataFrame(columns=["text", "Score"])

    # Parse the data using regex
    for i in range(len(output_dict)):
        matches = pattern.findall(output_dict[i])

        # Convert to a DataFrame
        df = pd.DataFrame(matches, columns=["text", "Score"])
        df["Score"] = df["Score"].astype(float)
        df = df.dropna(how='all')  # Remove rows where all values are NA
        if not df.empty:
            df_synthetic = pd.concat([df_synthetic, df], ignore_index=True)

    label_cld = """
        
        digraph {\n    
        \"Population\" -> \"Net Increase\" [arrowhead=vee]\n    
        \"Net Increase\" -> \"Population\" [arrowhead=vee]\n    
        \"Population\" -> \"Resources per Capita\" [arrowhead=tee]\n    
        \"Resources per Capita\" -> \"Fractional Net Increase\" [arrowhead=vee]\n    
        \"Fractional Net Increase\" -> \"Net Increase\" [arrowhead=vee]\n    
        \"Resources per Capita\" -> \"Carrying Capacity\" [arrowhead=vee]\n    
        \"Population\" -> \"Carrying Capacity\" [arrowhead=tee]\n    
        \"Carrying Capacity\" -> \"Resources per Capita\" [arrowhead=vee]\n
        
        }

        """
    df_synthetic['label_cld'] = label_cld
    df_synthetic['ID'] = df_synthetic.index
    df_synthetic = df_synthetic.rename(
        columns={'text': 'generated_cld_DOT', 'index': 'ID'})

    print("final data size", df_synthetic.shape)
    return df_synthetic


if __name__ == "__main__":
    main()
