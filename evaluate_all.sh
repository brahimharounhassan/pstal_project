#!/bin/bash

TEST_FILE="data/sequoia/sequoia-ud.parseme.frsemcor.simple.test"
TRAIN_FILE="data/sequoia/sequoia-ud.parseme.frsemcor.simple.train"
PREDICTIONS_DIR="predictions"

if [ ! -f "$TEST_FILE" ]; then
    echo "Erreur : Le fichier de test est introuvable : $TEST_FILE"
    exit 1
fi

echo "Evaluation start..."
echo "-----------------------------------"

for pred_file in "$PREDICTIONS_DIR"/*.conllu; do
    if [ -f "$pred_file" ]; then
        echo "Evaluating file : $(basename "$pred_file")"
        python lib/evaluate.py --gold "$TEST_FILE" --pred "$pred_file" --train "$TRAIN_FILE" --tagcolumn frsemcor:noun --upos-filter NOUN PROPN NUM
        echo "-----------------------------------"
    else
        echo "No .conllu found in $PREDICTIONS_DIR"
    fi
done

echo "All evaluations are completed."