"""
This script calculates various metrics for CLD data.

"""
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from graph_metrics import calculate_graph_scores
from my_functions import (
    digraph_to_nxgraph,
    get_correct_G_format,
    nx_2_graphviz,
    substitute_nodes,
)
from nlp_metrics import calculate_nlp_score

warnings.filterwarnings(
    "ignore", message="`clean_up_tokenization_spaces` was not set")

# Suppress specific warning related to "changing format from 'adjacency' to 'all'"
warnings.filterwarnings(
    "ignore", message='changing format from "adjacency" to "all"', category=UserWarning
)


# Suppress the specific UserWarning about overriding edge labels
warnings.filterwarnings(
    "ignore", message="overriding existing edgelabels for indexes", category=UserWarning
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate metrics for CLD data")
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file path")

    # write where to store the result csv
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Output folder path"
    )

    # optional arguments
    parser.add_argument(
        "--swap_nodes", action="store_true", required=False, help="Enable node substitution"
    )
    parser.add_argument(
        "--nlp_threshold", type=float, default=0.8, help="NLP similarity threshold for node substitution"
    )
    parser.add_argument(
        "--nlp_metric", type=str, default="cosine", choices=["cosine", "dot", "fuzzy"], help="NLP metric to use for similarity"
    )
    return parser.parse_args()


# # Main Code
def main():
    args = parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Setup output paths
    output_excel = os.path.join(args.output_folder, "metrics_results.xlsx")
    OUTPUT_FILE_PATH = os.path.join(
        args.output_folder, "distribution_plot.png")
    print(f"Output file path: {OUTPUT_FILE_PATH}")

    # Read input CSV
    df = pd.read_csv(args.input, encoding="utf-8")

    # no subsitution
    print("Get metrics scores without node substitution")
    df_results_no = get_metric_scores(df, swap_nodes=False)

    df_results_no["type"] = "before"

    if args.swap_nodes:
        print("Substitute nodes")
        df_results_sub = get_metric_scores(
            df, swap_nodes=True, nlp_threshold=args.nlp_threshold, nlp_metric=args.nlp_metric)
        df_results_sub["type"] = "after"
        df_result_all = pd.concat(
            [df_results_no, df_results_sub], ignore_index=True, sort=False, axis=0)
    else:
        df_result_all = df_results_no

    print("final shape of the dataset", df_result_all.shape)

    selected_metrics = [
        "fuzzy_score",
        "sent_dot_score",
        "cosine_score",
        "bleu_score",
        "pm_label",
        "sp_label",
        "wl_vh",
        "sub_graph",
    ]  # 'wl_sp', 'wl_eh',

    selected_nlp_metrics = [
        "fuzzy_score",
        "sent_dot_score",
        "cosine_score",
        "bleu_score",
    ]
    selected_graph_metrics = [
        "pm_label",
        "sp_label",
        "wl_vh",
        "sub_graph",
    ]  # 'wl_sp', 'wl_eh',

    df_result_all["sum_score_nlp"] = df_result_all[selected_nlp_metrics].sum(
        axis=1)
    df_result_all["sum_score_graph"] = df_result_all[selected_graph_metrics].sum(
        axis=1)
    df_result_all["sum_scores"] = df_result_all[selected_metrics].sum(axis=1)

    df_result_all["norm_sum_score_nlp"] = df_result_all["sum_score_nlp"] / len(
        selected_nlp_metrics
    )
    df_result_all["norm_sum_score_graph"] = df_result_all["sum_score_graph"] / len(
        selected_graph_metrics
    )
    df_result_all["norm_sum_score"] = df_result_all["sum_scores"] / len(
        selected_metrics
    )

    # print the distribution of the norm_sum_score_9
    print("Selected metrics: ", selected_metrics)
    print("Distribution of normalized sum scores on metrics: ", selected_metrics)

    print()

    print(df_result_all["norm_sum_score"].describe())
    # print ditribution plot
    df_result_all.to_excel(output_excel, index=False)

    # plot_distribution(df_result_all, selected_metrics, OUTPUT_FILE_PATH = OUTPUT_FILE_PATH)


def plot_distribution(df, seleted_columns, OUTPUT_FILE_PATH=None):
    """
    Plots the distribution of a given metric for a given metric type.
    """
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots()
    # plot the distribution of the sum of the scores
    sns.histplot(
        data=df,
        x=df["norm_sum_score"],
        ax=ax,
        fill=True,
        bins=20,
        multiple="dodge",
        shrink=0.8,
    )
    # , bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.figtext(0.5, -0.025, str(seleted_columns), ha="center", fontsize=12)

    ax.set_xlim(0, 1)
    ax.text(
        1.3, 0.9, f"Number of samples: {len(df)}", transform=ax.transAxes, ha="center"
    )

    plt.xlabel("Normalized Sum of Scores")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Normalized Sum of Scores")
    plt.tight_layout()
    if OUTPUT_FILE_PATH:
        plt.savefig(OUTPUT_FILE_PATH, dpi=300)
    else:
        plt.show()


def get_metric_scores(
    df, swap_nodes: bool = False, nlp_threshold: float = 0.5, nlp_metric: str = "fuzzy"
):
    """
    Get metrics scores for both NLP metrics and graph metrics.

    Args:
        df: DataFrame containing CLD data
        swap_nodes: Boolean flag to enable node substitution based on semantic similarity
        nlp_threshold: Threshold value for NLP similarity comparison
        nlp_metric: Type of NLP metric to use (e.g., "fuzzy", "cosine", "dot)

    Returns:
        DataFrame with calculated metric scores
    """
    my_df = df.copy()
    pbar = tqdm(total=len(my_df), desc="Calculating Metrics")

    for idx, row in my_df.iterrows():
        # Covert digraph to nxgraph for G truth
        G_truth = digraph_to_nxgraph(row["label_cld"])
        G_truth = get_correct_G_format(G_truth)

        # Covert digraph to nxgraph for G generated

        G_generated = digraph_to_nxgraph(row["generated_cld_DOT"])

        # skip the row if G_generated is None
        if G_generated is None:
            # delete the row from the dataframe
            my_df = my_df.drop(idx)
            continue
        else:
            G_generated = get_correct_G_format(G_generated)

        # if swap_nodes is True, swap nodes in G_generated to the node names in G_truth, if they are consider semantic similar.
        if swap_nodes:
            G_generated = substitute_nodes(
                G_truth, G_generated, nlp_threshold, nlp_metric
            )
            my_df.at[idx, "generated_cld_DOT"] = nx_2_graphviz(G_generated)

            G_generated = get_correct_G_format(G_generated)

        # Calculate NLP scores
        nlp_scores = calculate_nlp_score(G_truth, G_generated)

        # Add dictionary values as new columns
        for metric, score in nlp_scores.items():
            my_df.loc[idx, metric] = score

        # # Calculate Graph Metrics Scores
        G_truth = nx.convert_node_labels_to_integers(G_truth)
        G_generated = nx.convert_node_labels_to_integers(G_generated)

        graph_metrics_dict = calculate_graph_scores(G_truth, G_generated)

        for metric, score in graph_metrics_dict.items():
            my_df.loc[idx, metric] = score

        pbar.update(1)

    pbar.close()

    return my_df


if __name__ == "__main__":
    main()
