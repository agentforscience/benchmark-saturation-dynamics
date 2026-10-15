#!/bin/bash
# Run Exp 3 seed evaluation for a list of models on one GPU. Usage: run_seed_eval.sh <gpu> model1 model2 ...
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME=$PWD/.hf_cache CUDA_VISIBLE_DEVICES=$1
shift
for m in "$@"; do
  tag=$(basename $m)
  [ -f results/seed_eval/$tag.parquet ] && { echo "skip $m"; continue; }
  python src/seed_eval.py --model $m --seeds 10 --batch 32 2>&1 | grep -v -i "warn\|Loading weights"
done
