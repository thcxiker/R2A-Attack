#!/usr/bin/env bash
# Pre-download the public checkpoints used by R2A into the Hugging Face cache
# (HF_HOME). Everything is also fetched lazily on first use; this script just
# does it up front. Meta-Llama-3-8B is gated: accept its license on the Hub and
# run `huggingface-cli login` (or export HF_TOKEN) first.
#
#   bash scripts/download_models.sh            # all routers
#   bash scripts/download_models.sh small      # only what configs/smoke.yaml needs
set -euo pipefail

SMALL=(
  sentence-transformers/all-MiniLM-L6-v2      # lightweight router + GraphRouter
  microsoft/mdeberta-v3-base                  # RouterDC backbone
  routellm/bert_gpt4_augmented                # RouteLLM-BERT
  routellm/causal_llm_gpt4_augmented          # RouteLLM-Causal (Llama-3-8B, ~16 GB)
  meta-llama/Meta-Llama-3-8B                  # tokenizer of RouteLLM-Causal (gated)
  lmarena-ai/p2l-0.5b-bt-01132025
)
FULL=(
  routellm/mf_gpt4_augmented                  # RouteLLM-MF (target only; needs OPENAI_API_KEY)
  lmarena-ai/p2l-1.5b-bt-01132025
  lmarena-ai/p2l-7b-grk-02222025              # P2L target / member (~15 GB)
)

repos=("${SMALL[@]}")
if [[ "${1:-all}" != "small" ]]; then repos+=("${FULL[@]}"); fi

for repo in "${repos[@]}"; do
  echo ">>> ${repo}"
  if [[ "${repo}" == meta-llama/* ]]; then
    huggingface-cli download "${repo}" --include "tokenizer*" "*.json"
  else
    huggingface-cli download "${repo}"
  fi
done

cat <<'MSG'

RouterDC has no public checkpoint. Train it with the official code
(https://github.com/shuhao02/RouterDC) and place the weights at
    checkpoints/routerdc/best_model.pth
or point routers.routerdc.args.checkpoint_path to them.
MSG
