#!/usr/bin/env bash
set -e

echo "TRAINING SCRIPT EXECUTING"

python train_adv_bert.py \
  --bert_model_name_or_path bert-ipa-model \
  --train_data corpus_clean_train.csv \
  --test_data corpus_clean_test.csv \
  --output trained_adv_model

echo "TEST SCRIPT EXECUTING"

python infer_adv_bert.py \
  --model_dir trained_adv_model \
  --bert_model_name_or_path bert-ipa-model \
  --test_data corpus_clean_test.csv \
  --output inference_results

echo "PROGRAM COMPLETE"