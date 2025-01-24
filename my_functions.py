import ast
from io import StringIO

import networkx as nx
import pandas as pd

from nlp_metrics import calculate_fuzzy_score, get_sentence_transformer_score


def change_arrowhead_to_label(graph_json):
    """
    Changes the 'arrowhead' attribute of each link in the graph JSON to a 'cLink' attribute.
    """
    for link in graph_json["links"]:
        arrowhead = link.pop("arrowhead")
        if arrowhead == "vee":
            link["cLink"] = "+"
        elif arrowhead == "tee":
            link["cLink"] = "-"
        else:
            # Keep original value if not 'vee' or 'tee'
            link["cLink"] = "o"
    return graph_json


def digraph_to_nx_json(digraph_str):
    # TODO: redundant code, fix this later.
    """
    Converts a DOT string representation of a directed graph to a NetworkX graph object and a JSON representation of the graph.

    Args:
        digraph_str (str): A string containing the DOT representation of a directed graph.

    Returns:
        tuple: A tuple containing the following:
            - nx.DiGraph: The NetworkX directed graph object.
            - dict: A JSON representation of the graph.
    """
    try:
        # Replace escaped newline characters with actual newlines
        cleaned_digraph_str = digraph_str.replace(r"\n", "\n")

        # Create a NetworkX graph from the cleaned DOT string
        dot_file = StringIO(cleaned_digraph_str)
        nx_graph = nx.nx_pydot.read_dot(dot_file)
        G = nx.DiGraph(nx_graph)

        graph_json = nx.node_link_data(G)
        new_graph_json = change_arrowhead_to_label(graph_json)

        # Get nx graph from new_graph_json
        H = nx.node_link_graph(new_graph_json)

        return H, new_graph_json

    except Exception as e:
        print(f"Error parsing DOT string: {e}")
        print(f"Invalid DOT string:\n{digraph_str}")
        return None, None


def digraph_to_nxgraph(digraph_str):
    """
    Converts a DOT string representation of a directed graph to a NetworkX graph object and a JSON representation of the graph.

    Args:
        digraph_str (str): A string containing the DOT representation of a directed graph.

    Returns:
        tuple: A tuple containing the following:
            - nx.DiGraph: The NetworkX directed graph object.
    """
    try:
        # Replace escaped newline characters with actual newlines
        cleaned_digraph_str = digraph_str.replace(r"\n", "\n")

        # Create a NetworkX graph from the cleaned DOT string
        dot_file = StringIO(cleaned_digraph_str)
        nx_graph = nx.nx_pydot.read_dot(dot_file)
        G = nx.DiGraph(nx_graph)

        graph_json = nx.node_link_data(G)
        new_graph_json = change_arrowhead_to_label(graph_json)
        # Get nx graph from new_graph_json
        H = nx.node_link_graph(new_graph_json, directed=True, multigraph=False)

        return H

    except Exception as e:
        print(f"Error parsing DOT string: {e}")
        print(f"Invalid DOT string:\n{digraph_str}")
        return None


def get_nxgraph_info(nxgraph):
    """
    Get the number of nodes and edges in the graph
    """
    print(nxgraph.nodes(data=True))
    print(nxgraph.edges(data=True))
    print(nx.get_edge_attributes(nxgraph, "weight"))


def nx_2_graphviz(nxg):
    """
    get a graphviz dot string from a networkx graph
    """
    g = "digraph {\n"
    edges = list(nxg.edges())
    weights = list(nx.get_edge_attributes(nxg, "weight").values())
    digraph_weights = [
        "vee" if w == 1 else "tee" if w == -1 else "none" for w in weights
    ]

    for edge, w in zip(edges, digraph_weights):
        g += '"%s" -> "%s" [arrowhead=%s]\n' % (*edge, w)
    g += "}"
    return g


def get_correct_G_format(G_nx):
    # Ensure all nodes and edges in both graphs are properly labeled
    G_nx = clean_invalid_nodes(G_nx)
    G_nx = ensure_node_labels(G_nx)
    G_nx = ensure_networkx_graph_type(G_nx)
    G_nx = set_edge_weights(G_nx)

    return G_nx


# Function to clean invalid nodes like '\\n'
def clean_invalid_nodes(G):
    nodes_to_remove = [
        node for node in G.nodes if isinstance(node, str) and node.strip() == "\\n"
    ]
    G.remove_nodes_from(nodes_to_remove)
    return G


def ensure_node_labels(G):
    for node in G.nodes:
        G.nodes[node]["label"] = node
    return G


def ensure_networkx_graph_type(G):
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
        print("The graph type is: ", type(G))
    return G


def set_edge_weights(G):
    for u, v, data in G.edges(data=True):
        if "cLink" in data:
            if data["cLink"] == "+":
                data["weight"] = 1
            elif data["cLink"] == "-":
                data["weight"] = -1
            else:
                data["weight"] = 0
    return G


def substitute_nodes(G_truth, G_generated, threshold: float, criterion="cosine"):
    """
    Substitute nodes in the generated graph based on the highest similarity criterion.

    :param G_truth: Ground truth NetworkX graph
    :param G_generated: Generated NetworkX graph
    :param threshold: Similarity threshold for substitution
    :param criterion: Criterion for substitution ("webbert" or "fuzzy")
    :return: Updated generated graph after substitution
    """
    substitution_map = {}
    used_truth_nodes = set()

    # Calculate similarity between each node in G_generated and G_truth
    for g_node in G_generated.nodes():
        best_match = None
        best_similarity = 0

        for t_node in G_truth.nodes():
            if t_node in used_truth_nodes:
                continue  # Skip nodes that have already been used in a substitution

            # Choose the similarity calculation based on the criterion
            if criterion == "fuzzy":
                similarity = calculate_fuzzy_score(g_node, t_node)
            if criterion == "dot":
                similarity = get_sentence_transformer_score(
                    g_node, t_node, sim_metric="dot"
                )
            else:  # default to cosine similarity
                similarity = get_sentence_transformer_score(
                    g_node, t_node, sim_metric="cosine"
                )

            # print(f"Similarity between '{g_node}' and '{t_node}': {similarity:.4f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = t_node

        # Check if the best match exceeds the threshold and hasn't been used yet
        if best_match and best_similarity > threshold:
            substitution_map[g_node] = best_match
            used_truth_nodes.add(best_match)  # Mark the truth node as used
            # print(f"Substituting '{g_node}' with '{best_match}'")

    # Apply substitutions to the generated graph
    # print("Substitution Map:", substitution_map)
    G_fixed = nx.relabel_nodes(G_generated, substitution_map, copy=True)

    return G_fixed


def process_digraphs(df):
    """
    Processes the digraphs in the input DataFrame and returns two dictionary of NetworkX graphs for the ground truth and generated digraphs.
    """

    # Covert digraph to nxgraph
    G_groundtruth = df["label_cld"].apply(lambda x: digraph_to_nx_json(x)[0])

    # Convert JSON to nxgraph
    if "generated_cld" in df.columns and df["generated_cld"].notna().any():
        G_generated_nx = nx_json_to_nx_graph(df["generated_cld"])
        G_generated = G_generated_nx.apply(nx.node_link_graph)
    else:
        G_generated = df["generated_cld_DOT"].apply(
            lambda x: digraph_to_nx_json(x)[0])

    # Create result dictionary with non-None graphs
    dict_G_groundtruth = {
        idx: graph for idx, graph in enumerate(G_groundtruth) if graph is not None
    }

    dict_G_generated = {
        idx: graph for idx, graph in enumerate(G_generated) if graph is not None
    }

    return dict_G_groundtruth, dict_G_generated


def nx_json_to_nx_graph(df_json):
    """
    Converts a JSON representation of a NetworkX graph to a NetworkX graph object.
    """

    series_generated_cld = df_json.astype("str")
    series_generated_cld = series_generated_cld.apply(
        lambda x: ast.literal_eval(x))
    return series_generated_cld


def log_edges(G, label):
    print(f"{label} edges:")
    for u, v, data in G.edges(data=True):
        print(f"  {u} -> {v}, attributes: {data}")
