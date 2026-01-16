# PSTAL - Prédiction Structurée pour le Traitement Automatique des Langues

Pedagogical materials for the advanced NLP course of Master 2 in AI and ML, 
Aix Marseille University and Centrale Mediterranée.

* `sequoia`: (simplified) Sequoia corpus used for all lab exercises (TP)
* `lib`: code given to speed up system development, includes CONLL-U library `conllulib.py` and evaluation script `evaluate.py`
* `cm-code`: code snippets shown during theoretical course (CM)


## execution :
### BASE

- python src/train_ssense.py \
  --train data/sequoia/sequoia-ud.parseme.frsemcor.simple.train \
  --dev data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
  --output models/ssense_base.pt \
  --model-name almanach/camembert-base \
  --epochs 20 \
  --batch-size 64 \
  --dropout 0.3 \
  --lr 0.001 \
  <!-- --hidden-dim 256 \ -->


- python src/predict_ssense.py \
  --model models/ssense_base.pt \
  --input data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
  --output predictions/ssense_base_dev.conllu 
    

- python lib/evaluate.py \
    --pred predictions/ssense_base_dev.conllu \
    --gold data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
    --tagcolumn frsemcor:noun \
    --train data/sequoia/sequoia-ud.parseme.frsemcor.simple.train \
    --upos-filter NOUN,PROPN,NUM

### FINE TUNED

- python src/train_finetuned.py \
  --train data/sequoia/sequoia-ud.parseme.frsemcor.simple.train \
  --dev data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
  --output models/ssense_finetuned.pth \
  --finetuned-model models/final_model_epochs_50.pth \
  --epochs 20 \
  --batch-size 64 \
  --dropout 0.3 \
  --lr 0.001 \


- python src/predict_ssense.py \
  --model models/ssense_finetuned.pth \
  --input data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
  --finetuned-model models/final_model_epochs_50.pth \
  --output predictions/ssense_finetuned_dev.conllu 
    

- python lib/evaluate.py \
    --pred predictions/ssense_finetuned_dev.conllu \
    --gold data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
    --tagcolumn frsemcor:noun \
    --train data/sequoia/sequoia-ud.parseme.frsemcor.simple.train \
    --upos-filter NOUN PROPN NUM
