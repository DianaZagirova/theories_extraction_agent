



python scripts/run_stage7_normalization.py \
  --stage6 output/stage6_consolidated_final_theories.json \
  --clusters-out data/stage7/clusters_from_stage6_names.json \
  --distance-threshold 2.0 \
  --min-cluster-size 9 \
  --stage0 output/stage0_filtered_theories.json \
  --rare-max-size 1 \
  --reference-min-size 3 \
  --max-concurrent 8 \
  --out-norms output/stage7_name_normalizations.json \
  --out-consolidated output/stage7_consolidated_final_theories.json


  python src/normalization/cluster_stage7_from_stage6.py \
  --input-stage6 output/stage6_consolidated_final_theories.json \
  --output-json data/stage7/clusters_from_stage6_names.json \
  --distance-threshold 2.0 \
  --min-cluster-size 9


  python src/normalization/stage7_cluster_refinement.py \
  --clusters data/stage7/clusters_from_stage6_names.json \
  --stage6 output/stage6_consolidated_final_theories.json \
  --stage0 output/stage0_filtered_theories.json \
  --rare-max-size 1 \
  --reference-min-size 3 \
  --max-concurrent 8 \
  --out-norms output/stage7_name_normalizations.json \
  --out-consolidated output/stage7_consolidated_final_theories.json