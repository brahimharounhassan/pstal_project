#!/usr/bin/bash

mkdir -p pred
for model in bow gru cnn; do
  for in_type in word char; do
    echo "Evaluating model model/model-${model}-${in_type}.pt" | tee pred/dumas_test_pred-${model}-${in_type}.acc
    ./predict_textclass.py dumas_test.txt model/model-${model}-${in_type}.pt > pred/dumas_test_pred-${model}-${in_type}.txt
    ./eval_textclass.py dumas_test.txt pred/dumas_test_pred-${model}-${in_type}.txt | tee pred/dumas_test_pred-${model}-${in_type}.acc       
  done
done
