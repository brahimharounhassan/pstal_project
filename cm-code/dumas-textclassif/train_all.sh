#!/usr/bin/bash

mkdir -p model
for model in bow gru cnn; do
  for in_type in word char; do
    ./train_textclass.py dumas_train.txt dumas_dev.txt ${model} ${in_type}
    mv model.pt model/model-${model}-${in_type}.pt          
  done    
done
