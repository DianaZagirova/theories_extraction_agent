0 - rerun

python scripts/export_invalid_preprocessing_dois.py --results-db theories.db --output invalid_dois.txt

1- Export DB to JSON
python scripts/export_db_to_json.py --db theories_abstract.db --output theories_abstract_per_paper.json

python scripts/export_db_to_json.py --db theories.db --output theories_per_paper.json

2 - Quality Filter
python src/normalization/stage0_quality_filter.py

3 - Fuzzy Matching
python src/normalization/stage1_fuzzy_matching.py

4 - LLM Mapping
python src/normalization/stage1_5_llm_mapping.py

4.* check mapping

python scripts/export_checkpoint_to_json.py --include-batches
python scripts/export_checkpoint_to_json.py

python scripts/export_checkpoint_to_json.py \
  --output output/for_next_stage.json