# Datasets (not committed; re-download with the commands below)

- `olb_old/contents.parquet` - Open LLM Leaderboard v1 table (HF `open-llm-leaderboard-old/contents`)
- `olb_v2/contents.parquet` - Open LLM Leaderboard v2 table (HF `open-llm-leaderboard/contents`; v2 per-item details are gated, unused)
- `v1_items/*.pkl` - per-item correctness for 241 v1 models x 62 tasks: `python src/download_v1_details.py`
- `raw_eval/` - ARC-Easy/Challenge test (`allenai/ai2_arc`) and MMLU test (`cais/mmlu`, config `all`) parquet files
- `eval_items/items.jsonl` - 500-item samples per benchmark used in Exp 3 (seed 42)
- `metabench/` - metabench item subsets (inspected, not used: items only, no responses)
