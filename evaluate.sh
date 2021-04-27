#!/bin/bash

python evaluate.py --dataset_path datasets/squad_dev.jsonl.gz --output_path $1 > "$1_metrics.json"
