#!/usr/bin/env python3
"""
Stage 7 - Step 1: Cluster Stage 6 consolidated names.

- Input:  output/stage6_consolidated_final_theories.json
- Output: data/stage7/clusters_from_stage6_names.json

Approach: mirror of cluster_stage4_output.py but source is the
final_name_summary from Stage 6 consolidated output.
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


def load_stage6_consolidated(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stage 6 consolidated file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("final_name_summary", [])


def cluster_names(names: List[str], model: SentenceTransformer,
                  distance_threshold: float = 1.8,
                  min_cluster_size: int = 5) -> Dict[int, List[str]]:
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

    # Reassign small clusters to nearest large ones
    if small_clusters and large_clusters:
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
    elif small_clusters and not large_clusters:
        # If all clusters are small, just keep them as is
        logging.info("All clusters are small; keeping original small clusters")
        large_clusters = small_clusters

    return large_clusters


def save_clusters_to_json(clusters: Dict[int, List[str]],
                          name_to_count: Dict[str, int],
                          output_path: str) -> None:
    clusters_json = {}
    for label, cluster_members in clusters.items():
        members_list = [{"theory_name": name, "paper_count": name_to_count.get(name, 0)}
                        for name in cluster_members]
        members_list.sort(key=lambda x: x["paper_count"], reverse=True)
        clusters_json[str(label)] = {"size": len(cluster_members), "members": members_list}

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2, ensure_ascii=False)
    logging.info(f"Clusters saved to {output_path}")


def main(input_stage6: str,
         output_json: str,
         device_id: Optional[int],
         distance_threshold: float,
         min_cluster_size: int) -> None:
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

    logging.info(f"Loading Stage 6 consolidated: {input_stage6}")
    entries = load_stage6_consolidated(input_stage6)

    names: List[str] = []
    name_to_count: Dict[str, int] = {}

    for entry in entries:
        name = entry.get("final_name")
        count = entry.get("theory_ids_count", 0)
        if name:
            names.append(name)
            name_to_count[name] = count

    logging.info(f"Loaded {len(names)} unique final names from Stage 6")

    logging.info("Loading SentenceTransformer model 'all-mpnet-base-v2'")
    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    clusters = cluster_names(names, model, distance_threshold=distance_threshold, min_cluster_size=min_cluster_size)
    total_names = sum(len(members) for members in clusters.values())
    logging.info(f"Total names across clusters: {total_names}")

    save_clusters_to_json(clusters, name_to_count, output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7 - Cluster Stage 6 consolidated names")
    parser.add_argument("--input-stage6", type=str, default="output/stage6_consolidated_final_theories.json",
                        help="Path to Stage 6 consolidated file")
    parser.add_argument("--output-json", type=str, default="data/stage7/clusters_from_stage6_names.json",
                        help="Path to output clusters JSON file")
    parser.add_argument("--device-id", type=int, default=None, help="CUDA device ID to use")
    parser.add_argument("--distance-threshold", type=float, default=1.8,
                        help="Distance threshold for agglomerative clustering")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="Minimum cluster size to avoid reassignment")

    args = parser.parse_args()
    main(
        input_stage6=args.input_stage6,
        output_json=args.output_json,
        device_id=args.device_id,
        distance_threshold=args.distance_threshold,
        min_cluster_size=args.min_cluster_size,
    )
