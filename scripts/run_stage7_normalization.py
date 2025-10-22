#!/usr/bin/env python3
"""
Run Stage 7 normalization pipeline:
  1) Cluster Stage 6 consolidated names (embeddings clustering)
  2) Gather small names (size<=rare_max_size) within each cluster under broader names

Outputs:
  - data/stage7/clusters_from_stage6_names.json
  - output/stage7_name_normalizations.json
  - output/stage7_consolidated_final_theories.json
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, text=True)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    p = argparse.ArgumentParser(description="Run Stage 7 normalization pipeline")
    p.add_argument('--stage6', default='output/stage6_consolidated_final_theories.json',
                   help='Path to Stage 6 consolidated file')
    p.add_argument('--clusters-out', default='data/stage7/clusters_from_stage6_names.json',
                   help='Path to save Stage 7 clusters (from names)')
    p.add_argument('--distance-threshold', type=float, default=1.8,
                   help='Agglomerative clustering distance threshold')
    p.add_argument('--min-cluster-size', type=int, default=5,
                   help='Minimum cluster size to avoid reassignment')
    p.add_argument('--device-id', type=int, default=None,
                   help='CUDA device id (or omit for CPU)')

    p.add_argument('--stage0', default='output/stage0_filtered_theories.json',
                   help='Path to Stage0 theories file')
    p.add_argument('--rare-max-size', type=int, default=1,
                   help='Max size of names to gather (<=)')
    p.add_argument('--reference-min-size', type=int, default=3,
                   help='Min size of names considered as references (>=)')
    p.add_argument('--max-concurrent', type=int, default=8,
                   help='Max concurrent LLM calls in Stage 7 refinement')

    p.add_argument('--out-norms', default='output/stage7_name_normalizations.json',
                   help='Where to save name normalizations')
    p.add_argument('--out-consolidated', default='output/stage7_consolidated_final_theories.json',
                   help='Where to save Stage 7 consolidated file')

    args = p.parse_args()

    # Step 1: Cluster Stage 6 names
    Path(args.clusters_out).parent.mkdir(parents=True, exist_ok=True)
    cmd1 = [
        sys.executable,
        str(ROOT / 'src/normalization/cluster_stage7_from_stage6.py'),
        '--input-stage6', args.stage6,
        '--output-json', args.clusters_out,
        '--distance-threshold', str(args.distance_threshold),
        '--min-cluster-size', str(args.min_cluster_size),
    ]
    if args.device_id is not None:
        cmd1 += ['--device-id', str(args.device_id)]
    run(cmd1)

    # Step 2: Gather small names within clusters
    Path(args.out_norms).parent.mkdir(parents=True, exist_ok=True)
    cmd2 = [
        sys.executable,
        str(ROOT / 'src/normalization/stage7_cluster_refinement.py'),
        '--clusters', args.clusters_out,
        '--stage6', args.stage6,
        '--stage0', args.stage0,
        '--rare-max-size', str(args.rare_max_size),
        '--reference-min-size', str(args.reference_min_size),
        '--max-concurrent', str(args.max_concurrent),
        '--out-norms', args.out_norms,
        '--out-consolidated', args.out_consolidated,
    ]
    run(cmd2)

    print("\n✓ Stage 7 completed")
    print(f"  - Clusters: {args.clusters_out}")
    print(f"  - Name normalizations: {args.out_norms}")
    print(f"  - Consolidated: {args.out_consolidated}")


if __name__ == '__main__':
    main()
