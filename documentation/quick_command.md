


python scripts/answer_questions_per_paper.py \
    --dois-file data/sample_from_ext.txt \
    --reset-db \
    --only-dois-in-file

python scripts/evaluate_qa_results.py \
    --validation-file data/qa_validation_set_extended.json \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_evaluation_results_extended.json
    
python scripts/export_qa_results.py \
    --dois-file data/sample_from_ext.txt \
    --validation-file data/qa_validation_set_extended.json \
    --output-file qa_exported_results_extended.json