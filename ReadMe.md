# ReadMe


## 1. Overview
This repository contains a small toy project designed to learn how to use the `Hugging Face` ecosystem. This implementation is based on the official Hugging Face tutorial, which can be found [here](https://huggingface.co/learn/nlp-course/chapter3/4).


## 2. Organization
This project contains the following python files. 
* `CustomizedTrainer.py:` our customize trainer built to train a model.
* `OurDataset.py:` a class used to load and preprocess the dataset.
* `main.py:` main script.

After running the main script, two new folders should appear.

* `logs/:` log files recording training metrics.



## 3. Usage

### 3.1 Running the code
To train and evaluate a model, simply execute the `main.py` script. Hyperparameters can be 
customized by using command-line arguments:

```commandline
main.py --hyperparameter_name hyperparameter_value
```
Alternatively, all default values can be directly modified in the code. 

Please, refer to the main.py file for the full 
list of available hyperparameters.


### 3.2 Viewing Logs


To view the training logs, ensure you have TensorBoard installed (otherwise `pip install tensorboard`
should work). Then run
```commandline
tensorboard --logdir PATH_TO_LOG_FOLDER
```
Access the dashboard via the URL displayed in your terminal (usually something that http://localhost:6006/).


