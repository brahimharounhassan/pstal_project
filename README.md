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
  --output models/ssense_baseline.pt \
  --model-name almanach/camembert-base \
  --n-epochs 20 \
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

### FINE TUNED WITHOUT DORA

python src/train_finetuned.py \
  --train data/sequoia/sequoia-ud.parseme.frsemcor.simple.train \
  --dev data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
  --output models/ssense_finetuned_dora.pt \
  --finetuned-model models/peft_adapter_dora/ \
  --n-epochs 10 \
  --batch-size 32


python src/predict_ssense.py \
  --input data/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
  --finetuned-model models/final_model_epochs_50.pt \
  --output predictions/ssense_finetuned_dev.conllu \
  --normalize



    



Modèle LoRA fine-tuné (figé : Le backbone (CamemBERT original) est gelé seules les petites matrices LoRA apprennent)
          ↓
Extraction d'embeddings contextuels
          ↓
MLP SuperSenseClassifier (entraînable)
          ↓
Prédiction supersense



Phase 1 - Fine-tuning LoRA (hp_tuning.py / fine_tuning.py):
┌─────────────────────────────────────────────────┐
│ CamemBERT backbone (gelé)                       │
│         +                                       │
│ Matrices LoRA (entraînables)                    │  ← Tâche: POS tagging
│         ↓                                       │     (25 labels UPOS)
│ Classifier head temporaire (TOKEN_CLS)          │
└─────────────────────────────────────────────────┘
           Sauvegarde du modèle fine-tuné
                      ↓
Phase 2 - Extraction + MLP (train_finetuned.py):
┌────────────────────────────────────────────────┐
│ Modèle LoRA fine-tuné (FIGÉ - aucun gradient)  │
│         ↓                                      │
│ Extraction embeddings contextuels (768D)       │  ← Mode: eval(), torch.no_grad()
│         ↓                                      │
│ MLP SuperSenseClassifier (ENTRAÎNABLE)         │  ← Tâche: Super-sense
│   - Linear(768 → 256)                          │     (24 labels)
│   - ReLU + Dropout                             │
│   - Linear(256 → 24)                           │
└────────────────────────────────────────────────┘


DoRA (Weight-Decomposed Low-Rank Adaptation) est une variante de LoRA qui décompose les poids en magnitude et direction :

W' = W + ΔW = W + B × A  # LoRA standard
W' = m · (W + B × A) / ||W + B × A||  # DoRA (magnitude × direction normalisée)

Avantages de DoRA
+0.5-2% F1 sur certaines tâches (paper: Liu et al., 2024)
Meilleure convergence sur des tâches très spécifiques
Apprentissage plus stable de la magnitude et direction séparément
❌ Inconvénients de DoRA (pourquoi désactivé)
Instabilité avec CamemBERT

DoRA est optimisé pour LLaMA/GPT-style models
RoBERTa-based (CamemBERT) a une architecture différente
Risque de divergence pendant l'entraînement
Coût computationnel

# LoRA standard
output = W × input + (B × A) × input  # 1 normalisation

# DoRA
output = magnitude × normalize(W + B × A) × input  # 2 normalisations + calcul magnitude



Input tokens
    ↓
[ENCODEUR] (roberta) → Produit les embeddings contextuels
    ↓                   Dimension: (batch, seq_len, 768)
    |
    ├─→ [Embeddings contextuels] ← Ce qu'on veut !
    |
    ↓
[TÊTE CLASSIFICATION] (linear layer)
    ↓
Logits/Prédictions
Dimension: (batch, seq_len, 25)