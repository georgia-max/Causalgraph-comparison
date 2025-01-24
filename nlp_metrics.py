import numpy as np
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, SimilarityFunction
from transformers import BertModel, BertTokenizer


def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    # print(reference)
    candidate = candidate.split()
    # print(candidate)
    return sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)


def calculate_fuzzy_score(reference, candidate):
    # TODO: check if this is the right way to calculate fuzzy score,
    return fuzz.token_sort_ratio(reference, candidate)


def get_sentence_transformer_score(sent1, sent2, sim_metric):

    base_model = 'sentence-transformers/all-MiniLM-L6-v2'

    if sim_metric == 'dot':

        model = SentenceTransformer(
            base_model, similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
    elif sim_metric == 'cosine':
        model = SentenceTransformer(
            base_model, similarity_fn_name=SimilarityFunction.COSINE)
    elif sim_metric == 'euclidean':
        model = SentenceTransformer(
            base_model, similarity_fn_name=SimilarityFunction.EUCLIDEAN)
    elif sim_metric == 'manhattan':
        model = SentenceTransformer(
            base_model, similarity_fn_name=SimilarityFunction.MANHATTAN)

    # Compute embeddings for both lists
    embeddings1 = model.encode(sent1)
    embeddings2 = model.encode(sent2)

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)
    score = similarities[0][0].numpy().item()
    return score


def calculate_nlp_score(G_truth, G_generated):

    G_truth_nodes = str(G_truth.nodes())
    G_generated_nodes = str(G_generated.nodes())

    # cosine_sim_score = cosine_sim(G_truth_nodes, G_generated_nodes)
    bleu_score = calculate_bleu(G_truth_nodes, G_generated_nodes)
    fuzzy_score = calculate_fuzzy_score(G_truth_nodes, G_generated_nodes)/100
    sent_dot_score = get_sentence_transformer_score(
        G_truth_nodes, G_generated_nodes, sim_metric="dot")
    cosine_sim_score = get_sentence_transformer_score(
        G_truth_nodes, G_generated_nodes, sim_metric="cosine")
    sent_euc_score = get_sentence_transformer_score(
        G_truth_nodes, G_generated_nodes, sim_metric="euclidean")
    sent_man_score = get_sentence_transformer_score(
        G_truth_nodes, G_generated_nodes, sim_metric="manhattan")

    # node_precision, node_recall, node_f1_score = node_similarity_metrics(
    #     G_truth, G_generated)

    # edge_precision, edge_recall, edge_f1 = edge_similarity_metrics(
    #     G_truth, G_generated)

    # Helper function to handle None or NaN
    def safe_round(value, decimals=3, default=0):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return np.round(value, decimals)

    # Create a dictionary with safe rounding
    nlp_dict = {
        # "cosine_score": safe_round(cosine_sim_score),
        "bleu_score": safe_round(bleu_score),
        "fuzzy_score": safe_round(fuzzy_score),
        "sent_dot_score": safe_round(sent_dot_score),
        "cosine_score": safe_round(cosine_sim_score),
        "sent_euc_score": safe_round(sent_euc_score),
        "sent_man_score": safe_round(sent_man_score),

        # "node_precision": safe_round(node_precision),
        # "node_recall": safe_round(node_recall),
        # "node_f1_score": safe_round(node_f1_score),
        # "edge_precision": safe_round(edge_precision),
        # "edge_recall": safe_round(edge_recall),
        # "edge_f1_score": safe_round(edge_f1)
    }

    return nlp_dict


def node_similarity_metrics(G_truth, G_generated):
    truth_nodes = set(normalize_label(node) for node in G_truth.nodes())
    generated_nodes = set(normalize_label(node)
                          for node in G_generated.nodes())

    matched_nodes = truth_nodes.intersection(generated_nodes)
    # if there is 1-to-many or many-to-1 mapping, we need to consider both the node precision and recall
    node_precision = len(matched_nodes) / \
        len(generated_nodes) if len(generated_nodes) > 0 else 0
    node_recall = len(matched_nodes) / \
        len(truth_nodes) if len(truth_nodes) > 0 else 0
    node_f1 = 2 / (1 / node_precision + 1 /
                   node_recall) if node_precision + node_recall > 0 else 0

    return node_precision, node_recall, node_f1


def edge_similarity_metrics(G_truth, G_generated):
    truth_edges = {(normalize_label(u), normalize_label(v), frozenset(
        d.items())) for u, v, d in G_truth.edges(data=True)}
    generated_edges = {(normalize_label(u), normalize_label(v), frozenset(
        d.items())) for u, v, d in G_generated.edges(data=True)}

    matched_edges = truth_edges.intersection(generated_edges)
    edge_precision = len(matched_edges) / \
        len(generated_edges) if len(generated_edges) > 0 else 0
    edge_recall = len(matched_edges) / \
        len(truth_edges) if len(truth_edges) > 0 else 0
    edge_f1 = 2 / (1 / edge_precision + 1 /
                   edge_recall) if (edge_precision + edge_recall) > 0 else 0

    return edge_precision, edge_recall, edge_f1


def normalize_label(label):
    """Normalize the node or edge label for comparison."""
    return label.strip().lower()


def normalize_edge(edge):
    """Normalize an edge tuple for comparison by normalizing its labels."""
    u, v, d = edge
    return (normalize_label(u), normalize_label(v), frozenset(d.items()))
