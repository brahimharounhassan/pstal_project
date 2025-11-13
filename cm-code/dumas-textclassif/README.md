## Text Classifiers for Dumas novels

The code in this folder illustrates the use of `torch` for text classification.
The task is to predict the title of Alexandre Dumas' novels from a sentence.
There are 5 possible novel titles in the dataset:
  * `La_Reine_Margot`
  * `Le_comte_de_Monte_Cristo`
  * `Le_Trois_Mousquetaires`
  * `Le_Vicomte_de_Bragelonne`
  * `Vingt_ans_apres`
  
  
## Data folder `data`

The `data` folder contains the texts used to train, tune and evaluate the 
classifiers. The texts are a sample of those used in the course _Modèles de
langage (ML)_. The corpora are in the following format:
  * UTF-8 encoded text, LF-ended lines
  * Each line contains the novel title and a sentence from that novel
  * Novel title = 1st "word" on the line, title words separated by underscores `_`
  * After the novel title, the sentence words are separated by spaces
  * Sentences have been tokenized and lowercased using the tools of ML course 
  
  
## Python code `.py`

There are three Python programs in this folder:
  * `train_textclass.py`: creates a text classification model file `model.pt`. 
  Arguments: 
  `trainfile.txt devfile.txt bow|gru|cnn word|char`
  The 3rd argument defines the type of model: 
    - `bow` = bag of words (average all input embeddings)
    - `rnn` = recurrent network combines embeddings, last RNN state predicts
    - `cnn` = convolutional network combines embeddings
  The 4th argument defines whether to use `word` or `char`acter embeddings.
  * `predict_textclass.py`: predicts the class of input text from `model.pt`
  Arguments: `testfile.txt model.pt` - `model.pt` created by `train_textclass.py`
  Input `testfile.txt` and output are in same format as `trainfile.txt` (see 
  `data` section above). First column will be replaced by predicted novel title 
  in output. 
  * `eval_textclass.py`: calculates the accuracy of prediction wrt. gold file
  Arguments: `gold-testfile.txt pred-testfile.pt` same format as `trainfile.txt`.
  Output accuracy score on `stdout`
  
  
## Bash scripts `.sh`

Scripts to automatise training and evaluation experiments.
`train_all.sh` trains all model combinations (cnn, rnn, bow) vs. (char, word)
`eval_all.sh` generates predictions and evaluates  all trained model combinations.
Results are saved in folder `pred`, with accuracies in `.acc` files.
  
## Predictions folder `pred`

Contain the predicted novel titles for the test set using all model combinations.
