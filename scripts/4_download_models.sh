#!/bin/bash
# 실험에 쓰는 Llama 모델을 $MODELS_DIR에 받는다. meta-llama 접근 승인 + `hf auth login`(또는 HF_TOKEN)이 필요하다.
set -eo pipefail
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
source $SCRIPTS/env.sh

hf download meta-llama/Llama-2-7b-hf --local-dir $MODELS_DIR/Llama-2-7b-hf \
  --include "*.json" "*.safetensors" "tokenizer.model"
hf download meta-llama/Meta-Llama-3-8B --local-dir $MODELS_DIR/Meta-Llama-3-8B \
  --exclude "original/*"
