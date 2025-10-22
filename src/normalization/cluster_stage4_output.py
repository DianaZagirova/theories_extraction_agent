#!/usr/bin/env python3
"""
Script to cluster theory names based on embeddings and paper counts.

Requirements:
- torch
- sentence_transformers
- scikit-learn
- pandas
- numpy
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


def normalize_text(text: str) -> str:
    """Normalize theory name text for better clustering."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(theory|model|concept|framework|hypothesis)\b", "", text)
    text = re.sub(r"\bageing\b", "aging", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json_file(filepath: str) -> dict:
    """Load JSON data from a file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_valid_entries(theory_lineage: dict) -> Tuple[Dict[str, dict], List[Optional[str]]]:
    """
    Filter entries with final_status == 'valid' and collect others' statuses.
    """
    final_mapping = {}
    status = []
    for key, entry in theory_lineage.items():
        if entry.get("final_status") == "valid":
            final_mapping[key] = {
                "final_name": entry.get("final_name"),
                "final_name_normalized": entry.get("final_name_normalized"),
                "final_status": entry.get("final_status"),
                "name_warning": entry.get("name_warning"),
                "status_warning": entry.get("status_warning"),
                "doi": entry.get("doi"),
                "paper_focus": entry.get("paper_focus"),
                "mode": entry.get("mode"),
                "stage1_name": entry.get("stage1_name"),
            }
        else:
            status.append(entry.get("final_status"))
    return final_mapping, status


def group_by_final_name(data: Dict[str, dict]) -> Dict[str, List[dict]]:
    """Group entries by their normalized final name."""
    grouped = defaultdict(list)
    for key, value in data.items():
        final_name = value.get("final_name_normalized")
        if final_name:
            grouped[final_name].append({key: value})
    return dict(grouped)


def filter_by_max_focus(data: Dict[str, dict]) -> Dict[str, dict]:
    """
    For each DOI, keep only the entry with the max paper_focus.
    """
    doi_max_focus = {}
    for key, info in data.items():
        doi = info.get("doi")
        focus = info.get("paper_focus", 0)
        if doi:
            if doi not in doi_max_focus or focus > doi_max_focus[doi]:
                doi_max_focus[doi] = focus

    filtered_data = {}
    for key, info in data.items():
        doi = info.get("doi")
        focus = info.get("paper_focus", 0)
        if doi and doi_max_focus.get(doi) == focus:
            filtered_data[key] = info

    with open('data/clustering_data/filtered_paper_data.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
    return filtered_data


def count_unique_dois_per_name(filtered_data: Dict[str, dict]) -> Dict[str, int]:
    """Count unique DOIs per normalized final name."""
    doi_per_name = defaultdict(set)
    for entry in filtered_data.values():
        name_norm = entry.get("final_name_normalized")
        doi = entry.get("doi")
        if name_norm and doi:
            doi_per_name[name_norm].add(doi)
    return {name: len(dois) for name, dois in doi_per_name.items()}


def cluster_names(
    names: List[str], model: SentenceTransformer, distance_threshold: float = 1.8, min_cluster_size: int = 5
) -> Dict[int, List[str]]:
    """
    Cluster theory names using sentence embeddings and agglomerative clustering.

    Small clusters are reassigned to nearest large clusters.
    """
    logging.info(f"Encoding {len(names)} names with model {model.__class__.__name__}")
    embeddings = model.encode(names, batch_size=32, device=model.device, show_progress_bar=True)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    labels = clustering.fit_predict(embeddings)

    groups = defaultdict(list)
    groups_embeddings = defaultdict(list)
    for label, name, emb in zip(labels, names, embeddings):
        groups[label].append(name)
        groups_embeddings[label].append(emb)

    # Separate large and small clusters
    large_clusters = {}
    large_clusters_emb = {}
    small_clusters = {}
    small_clusters_emb = {}

    for label, members in groups.items():
        if len(members) >= min_cluster_size:
            large_clusters[label] = members
            large_clusters_emb[label] = np.mean(groups_embeddings[label], axis=0)
        else:
            small_clusters[label] = members
            small_clusters_emb[label] = np.mean(groups_embeddings[label], axis=0)

    if small_clusters:
        logging.info(f"Reassigning {len(small_clusters)} small clusters to nearest large clusters")
        large_labels = list(large_clusters_emb.keys())
        large_centroids = np.stack([large_clusters_emb[l] for l in large_labels])

        small_labels = list(small_clusters_emb.keys())
        small_centroids = np.stack([small_clusters_emb[l] for l in small_labels])

        dists = cosine_distances(small_centroids, large_centroids)
        nearest_large_idx = np.argmin(dists, axis=1)

        for i, small_label in enumerate(small_labels):
            assigned_large_label = large_labels[nearest_large_idx[i]]
            large_clusters[assigned_large_label].extend(small_clusters[small_label])
    else:
        logging.info("No small clusters to reassign.")

    return large_clusters


def save_clusters_to_json(
    clusters: Dict[int, List[str]], sorted_counts: Dict[str, int], output_path: str
) -> None:
    """Save clusters with paper counts to a JSON file."""
    clusters_json = {}
    for label, cluster_members in clusters.items():
        members_list = [{"theory_name": name, "paper_count": sorted_counts.get(name, 0)} for name in cluster_members]
        members_list.sort(key=lambda x: x["paper_count"], reverse=True)
        clusters_json[str(label)] = {"size": len(cluster_members), "members": members_list}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2, ensure_ascii=False)
    logging.info(f"Clusters saved to {output_path}")


def main(
    input_json_path: str,
    output_json_path: str,
    device_id: Optional[int] = None,
    distance_threshold: float = 1.8,
    min_cluster_size: int = 5,
) -> None:
    """Main function to run the clustering pipeline."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if device_id is not None:
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available, falling back to CPU")
            device = "cpu"
        else:
            device = f"cuda:{device_id}"
            torch.cuda.set_device(device_id)
        logging.info(f"Using device: {device}")
    else:
        device = "cpu"
        logging.info("Using CPU device")

    # Load input data
    logging.info(f"Loading input JSON file: {input_json_path}")
    theory_tracking_report = load_json_file(input_json_path)

    theory_lineage = theory_tracking_report.get("theory_lineage")
    if theory_lineage is None:
        raise KeyError("Input JSON does not contain 'theory_lineage' key")

    # Filter valid entries
    data, status = filter_valid_entries(theory_lineage)
    logging.info(f"Filtered {len(data)} valid entries, {len(status)} invalid or other statuses")

    # Group by final_name_normalized
    grouped = group_by_final_name(data)
    logging.info(f"Grouped into {len(grouped)} unique normalized final names")

    # Filter by max paper_focus per DOI
    filtered_data = filter_by_max_focus(data)
    grouped_filtered = group_by_final_name(filtered_data)
    assert len(grouped_filtered) == len(grouped), "Filtered grouping size mismatch"

    # Count unique DOIs per final_name_normalized
    doi_counts = count_unique_dois_per_name(filtered_data)
    sorted_counts = dict(sorted(doi_counts.items(), key=lambda x: x[1], reverse=True))
    logging.info(f"Counted unique DOIs for {len(sorted_counts)} theory names")

    # Load SentenceTransformer model
    logging.info("Loading SentenceTransformer model 'all-mpnet-base-v2'")
    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    names = list(sorted_counts.keys())

    # Cluster names
    clusters = cluster_names(names, model, distance_threshold=distance_threshold, min_cluster_size=min_cluster_size)

    # Log cluster info
    total_unique_names = sum(len(members) for members in clusters.values())
    logging.info(f"Total unique names across all clusters: {total_unique_names}")
    for label, cluster_members in clusters.items():
        logging.info(f"Cluster {label} size: {len(cluster_members)}")

    # Save clusters to JSON
    save_clusters_to_json(clusters, sorted_counts, output_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster theory names based on embeddings and paper counts.")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to input theory_tracking_report.json file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to output clusters JSON file",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="CUDA device ID to use (e.g. 0, 1, 2). If not set or CUDA not available, CPU is used.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=1.8,
        help="Distance threshold for agglomerative clustering",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size to avoid reassignment",
    )

    args = parser.parse_args()

    main(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        device_id=args.device_id,
        distance_threshold=args.distance_threshold,
        min_cluster_size=args.min_cluster_size,
    )