# Flair Predictor Model for r/india

## Run
* Jupyter
```
jupyter-notebook
then run all cells
```
* Python directly
```
python3 model.py
```

## Folder info:
* data: datasets
* writeup: analysis of model performance

## File info:
* model.py: python file having model class and used to run training
* utils\_new.py: pythin file having helper methods to model
* model\_jupyter.ipynb: jupyter notebook for model
* clean.sh: bash script to remove checkpoints

## Model Architecture:
* 32 dimensional fasttext embeddings for each word
* 50 timesteps used for each input
* 50 hidden units in each gru cell
* 120 dimensional fully connected layer
* 240 dimensional fully connected layer
* 12 dimensional output fully connected layer

## Regularization:
* 0.6 l2 weight regularization in all layers
* dropout disabled

## Loss:
* softmax cross entropy with logits

## Optimizer:
* adam (lr = 1e-3)

## Batch size:
* 32

## Summaries:
* loss: loss per update
* loss\_avg: avg loss per epoch
* cat'n'\_acc: no of right predictions for category 'n' on validation set
* total acc: no of total right predictions on validation set

## Dataset format used:

```
[flair]
[title]
```

## Additional Notes:
* To just evaluate the model on a set, change the filenmae in FlairPredict.get\_data() and uncomment the exit() at start of training loop
* To view tensorboard summaries, run the following in this directory

```
tensorboard --logdir='./graphs/flair' --port=6006
```

* To write the Boolean results of the validation set, use write\_preds = True in self.eval\_once()

