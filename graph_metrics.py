import grakel as gk
import networkx as nx
import numpy as np
from grakel import GraphKernel
from grakel.kernels import (
    CoreFramework,
    EdgeHistogram,
    GraphletSampling,
    NeighborhoodHash,
    NeighborhoodSubgraphPairwiseDistance,
    PropagationAttr,
    PyramidMatch,
    RandomWalkLabeled,
    ShortestPathAttr,
    SubgraphMatching,
    VertexHistogram,
    WeisfeilerLehman,
)


# https://ysig.github.io/GraKeL/0.1a8/graph_kernel.html
# Kernel score calculation using the Shortest Path Kernel
def calculate_graph_scores(G_truth, G_generated):

    G_nx_list = [G_truth, G_generated]

    Gk = convert_to_grakel_format(G_nx_list)

    ged = calculate_ged_score(G_truth, G_generated)

    # for i in range(len(Gk)):
    #     print(Gk[i])
    #     print(type(Gk[i]))
    # Gk[i] = Gk[i].desired_format("all")

    core_sp = get_kernel_score(Gk, kernel_type='core_sp')
    sub_graph = get_kernel_score(Gk, kernel_type='sub_graph')

    wl_sp = get_kernel_score(Gk, kernel_type='wl_sp')
    wl_eh = get_kernel_score(Gk, kernel_type='wl_eh')
    wl_vh = get_kernel_score(Gk, kernel_type='wl_vh')
    pm_label = get_kernel_score(Gk, kernel_type='pm_label')
    sp_label = get_kernel_score(Gk, kernel_type='sp_label')
    # propagt = get_kernel_score(Gk, kernel_type='propagt')

    def safe_round(value, decimals=3, default=0):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return np.round(value, decimals)

    graph_dict = {
        'ged': safe_round(ged, 3),
        'wl_sp': safe_round(wl_sp, 3),
        'wl_eh': safe_round(wl_eh, 3),

        'wl_vh': safe_round(wl_vh, 3),
        'pm_label': safe_round(pm_label, 3),
        'sp_label': safe_round(sp_label, 3),
        # 'propagt': safe_round(propagt,3),
        'core_sp': safe_round(core_sp, 3),
        'sub_graph': safe_round(sub_graph, 3)
    }

    return graph_dict


def get_kernel_score(G_list, kernel_type):
    # print("current kernel type: ", kernel_type)

    def custom_edge_kernel(label1, label2):
        # Example: Assign partial similarity for mismatched labels
        # print(label1, label2)
        if label1 == label2:
            return 1  # Perfect match
        else:
            return 0  # Partial similarity for mismatched labels

    def custom_label_kernel(label1, label2):
        # Example: Assign partial similarity for mismatched labels
        # print(label1, label2)
        if label1 == label2:
            return 1  # Perfect match
        else:
            return 0  # Partial similarity for mismatched labels

    try:
        if len(G_list) != 2:
            raise ValueError("Expected two graphs for comparison")

        # NOTE: correct! Consider both labels and edge labels.

        # https://ysig.github.io/GraKeL/0.1a8/generated/grakel.SubgraphMatching.html?highlight=subgraph_matching

        if kernel_type == 'wl_sp':  # NOTE: not considering edge labels.
            kernel = gk.WeisfeilerLehman(
                n_iter=15, normalize=True, base_graph_kernel=ShortestPathAttr)
        if kernel_type == 'wl_eh':  # NOTE: not considering edge labels.
            kernel = gk.WeisfeilerLehman(
                n_iter=30, normalize=True, base_graph_kernel=EdgeHistogram)

        if kernel_type == 'wl_vh':
            kernel = gk.WeisfeilerLehman(
                n_iter=30, normalize=True, base_graph_kernel=VertexHistogram)
        if kernel_type == 'pm_label':  # NOTE: not considering edge labels.

            # change format to all

            kernel = GraphKernel(
                {"name": "pyramid_match", "with_labels": True}, normalize=True)
        if kernel_type == 'sp_label':  # NOTE: cosidering edge labels.
            kernel = GraphKernel({"name": "shortest_path",
                                 "with_labels": True,
                                  "algorithm_type": "floyd_warshall",
                                  }, normalize=True)
        if kernel_type == 'propagt':
            # need label to be different so that the edge lable is different when the label is different. only work with attribute.
            kernel = PropagationAttr(normalize=True, t_max=10)
        if kernel_type == 'core_sp':
            kernel = CoreFramework(
                normalize=True, base_graph_kernel=SubgraphMatching)

        # NOTE: not considering edge labels. #bug in subgraph_matching function, don't use it.
        elif kernel_type == 'sub_graph':
            kernel = gk.SubgraphMatching(
                normalize=True, kv=custom_label_kernel, ke=custom_edge_kernel)

        # Ensure the input is an iterable (list)
        G_groundtruth_grakel = G_list[0]
        G_generated_grakel = G_list[1]

        if G_groundtruth_grakel and G_generated_grakel:

            # Fit and transform the ground truth graph
            kernel.fit_transform([G_groundtruth_grakel])
            # Calculate the kernel score between ground truth and generated graphs
            K_score = kernel.transform([G_generated_grakel])

            # Return the scalar kernel score
            if K_score is not None:
                return np.round(K_score[0][0], 3)

            else:
                return None
        else:
            print("Graph conversion to GraKeL format failed.")
            return None

    except Exception as e:
        print(f"Error calculating kernel score: {e}")
        return None


def convert_to_grakel_format(G_nx):

    # Check if the generated CLD is empty
    if len(G_nx[1].nodes()) == 0 or len(G_nx[1].edges()) == 0:
        print("Warning: Empty graph detected")
        return None

    # Verify edge attributes exist
    for _, _, data in G_nx[1].edges(data=True):
        if 'cLink' not in data or 'weight' not in data:
            print("Warning: Missing edge attributes")
            return None

    try:

        Gk = gk.graph_from_networkx(
            G_nx, node_labels_tag='label', edge_labels_tag='cLink', edge_weight_tag='weight', as_Graph=True)

        Gk_list = list(Gk)

        return Gk_list

    except Exception as e:
        print(f"Error converting from networkX to GraKeL format:  {e}")

        return None


def calculate_ged_score(G_truth, G_generated):
    ged = nx.graph_edit_distance(
        G_truth, G_generated, node_match=node_match, edge_match=edge_match)
    return ged


# Define a node matching function with a default value for 'label'
def node_match(n1, n2):
    return n1.get('label', '') == n2.get('label', '')

# Define an edge matching function with a default value for 'cause'


def edge_match(e1, e2):
    return e1.get('cLink', '') == e2.get('cLink', '')
