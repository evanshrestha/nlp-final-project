#!/bin/bash

for RUN in 1 2 3
do
  # Random Embedding + GloVe Tokenization + No Augmentation
  python main.py \
    --use_gpu \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --hidden_dim 256 \
    --batch_size 192 \
    --bidirectional \
    --augment_prob 0.1 \
    --do_train \
    --do_test \
    --model "baseline" \
    --model_path "random_glove_noaug_run_$RUN.pt" \
    --output_path "random_glove_noaug_run_$RUN.txt" \
    --dataset "baseline" \
    --use_random_embeddings

  # Random Embedding + GloVe Tokenization + WordNet Augmentation
  python main.py \
    --use_gpu \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --hidden_dim 256 \
    --batch_size 192 \
    --bidirectional \
    --augment_prob 0.1 \
    --do_train \
    --do_test \
    --model "baseline" \
    --model_path "random_glove_aug_run_$RUN.pt" \
    --output_path "random_glove_aug_run_$RUN.txt" \
    --dataset "baseline" \
    --use_random_embeddings \
    --question_augmentation

  # GloVe Embedding + GloVe Tokenization + No Augmentation
  python main.py \
    --use_gpu \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --hidden_dim 256 \
    --batch_size 192 \
    --bidirectional \
    --augment_prob 0.1 \
    --do_train \
    --do_test \
    --model "baseline" \
    --model_path "glove_glove_noaug_run_$RUN.pt" \
    --output_path "glove_glove_noaug_run_$RUN.txt" \
    --dataset "baseline"

  # GloVe Embedding + GloVe Tokenization + WordNet Augmentation
  python main.py \
    --use_gpu \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --hidden_dim 256 \
    --batch_size 192 \
    --bidirectional \
    --augment_prob 0.1 \
    --do_train \
    --do_test \
    --model "baseline" \
    --model_path "glove_glove_aug_run_$RUN.pt" \
    --output_path "glove_glove_aug_run_$RUN.txt" \
    --dataset "baseline" \
    --question_augmentation

  # Random Embedding + Hugging Face Tokenization + No Augmentation
  python main.py \
    --use_gpu \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --hidden_dim 256 \
    --batch_size 192 \
    --bidirectional \
    --augment_prob 0.1 \
    --do_train \
    --do_test \
    --model "baseline" \
    --model_path "random_hf_noaug_run_$RUN.pt" \
    --output_path "random_hf_noaug_run_$RUN.txt" \
    --dataset "bert" \
    --use_random_embeddings

  # Random Embedding + Hugging Face Tokenization + WordNet Augmentation
  python main.py \
    --use_gpu \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --hidden_dim 256 \
    --batch_size 192 \
    --bidirectional \
    --augment_prob 0.1 \
    --do_train \
    --do_test \
    --model "baseline" \
    --model_path "random_hf_aug_run_$RUN.pt" \
    --output_path "random_hf_aug_run_$RUN.txt" \
    --dataset "bert" \
    --use_random_embeddings \
    --question_augmentation

  # DistillBERT Embedding + Hugging Face Tokenization + No Augmentation
  python main.py \
    --use_gpu \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --hidden_dim 256 \
    --batch_size 192 \
    --bidirectional \
    --augment_prob 0.1 \
    --do_train \
    --do_test \
    --model "bert" \
    --model_path "bert_hf_noaug_run_$RUN.pt" \
    --output_path "bert_hf_noaug_run_$RUN.txt" \
    --dataset "bert"

  # DistillBERT Embedding + Hugging Face Tokenization + WordNet Augmentation
  python main.py \
    --use_gpu \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --hidden_dim 256 \
    --batch_size 192 \
    --bidirectional \
    --augment_prob 0.1 \
    --do_train \
    --do_test \
    --model "bert" \
    --model_path "bert_hf_aug_run_$RUN.pt" \
    --output_path "bert_hf_aug_run_$RUN.txt" \
    --dataset "bert" \
    --question_augmentation

done