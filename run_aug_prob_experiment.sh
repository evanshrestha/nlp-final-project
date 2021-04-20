#!/bin/bash

for RUN in 1 2 3
  for AUG in 0.1 0.2 0.3 0.4 0.5
  do
    # Random Embedding + GloVe Tokenization + WordNet Augmentation
    python main.py \
      --use_gpu \
      --train_path "datasets/squad_train.jsonl.gz" \
      --dev_path "datasets/squad_dev.jsonl.gz" \
      --hidden_dim 256 \
      --batch_size 192 \
      --bidirectional \
      --augment_prob "$AUG" \
      --do_train \
      --do_test \
      --model "baseline" \
      --model_path "random_glove_aug_${AUG}_run_$RUN.pt" \
      --output_path "random_glove_aug_${AUG}_run_$RUN.txt" \
      --dataset "baseline" \
      --use_random_embeddings \
      --question_augmentation

    ./evaluate.sh "random_glove_aug_${AUG}_run_$RUN.txt"

    # GloVe Embedding + GloVe Tokenization + WordNet Augmentation
    python main.py \
      --use_gpu \
      --train_path "datasets/squad_train.jsonl.gz" \
      --dev_path "datasets/squad_dev.jsonl.gz" \
      --hidden_dim 256 \
      --batch_size 192 \
      --bidirectional \
      --augment_prob "$AUG" \
      --do_train \
      --do_test \
      --model "baseline" \
      --model_path "glove_glove_aug_${AUG}_run_$RUN.pt" \
      --output_path "glove_glove_aug_${AUG}_run_$RUN.txt" \
      --dataset "baseline" \
      --question_augmentation

    ./evaluate.sh "glove_glove_aug_${AUG}_run_$RUN.txt"

    # Random Embedding + Hugging Face Tokenization + WordNet Augmentation
    python main.py \
      --use_gpu \
      --train_path "datasets/squad_train.jsonl.gz" \
      --dev_path "datasets/squad_dev.jsonl.gz" \
      --hidden_dim 256 \
      --batch_size 192 \
      --bidirectional \
      --augment_prob "$AUG" \
      --do_train \
      --do_test \
      --model "baseline" \
      --model_path "random_hf_aug_${AUG}_run_$RUN.pt" \
      --output_path "random_hf_aug_${AUG}_run_$RUN.txt" \
      --dataset "bert" \
      --use_random_embeddings \
      --question_augmentation

    ./evaluate.sh "random_hf_aug_${AUG}_run_$RUN.txt"

    # DistillBERT Embedding + Hugging Face Tokenization + WordNet Augmentation
    python main.py \
      --use_gpu \
      --train_path "datasets/squad_train.jsonl.gz" \
      --dev_path "datasets/squad_dev.jsonl.gz" \
      --hidden_dim 256 \
      --batch_size 192 \
      --bidirectional \
      --augment_prob "$AUG" \
      --do_train \
      --do_test \
      --model "bert" \
      --model_path "bert_hf_aug_${AUG}_run_$RUN.pt" \
      --output_path "bert_hf_aug_${AUG}_run_$RUN.txt" \
      --dataset "bert" \
      --question_augmentation

    ./evaluate.sh "bert_hf_aug_${AUG}_run_$RUN.txt"
  done
done