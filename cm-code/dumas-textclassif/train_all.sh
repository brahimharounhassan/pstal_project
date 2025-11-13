#!/usr/bin/bash

mkdir -p model
for model in bow gru cnn; do
  for in_type in word char; do
    ./train_textclass.py data/dumas_train.txt data/dumas_dev.txt ${model} ${in_type}
    mv model.pt model/model-${model}-${in_type}.pt          
  done    
done
