# ReadMe


## 1. Overview
This repository contains a small project developed to explore and better understand the `Hugging Face` ecosystem.

Specifically, in this project we fine-tune a [BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) model on the [symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) dataset, which contains 835 training examples and 212 validation examples.

The objective is to perform automatic classification of free-text symptom descriptions into one of 22 predefined diagnostic categories.


## 2. Project Structure

The project is organized into the following core components:

- `CustomizedTrainer.py`: Custom training class used to fine-tune the model.
- `OurDataset.py`: Class responsible for loading and preprocessing the dataset.
- `utils.py`: Collection of helper functions.
- `string_to_json.json`: JSON file mapping diagnosis (strings) to integer IDs.
- `id_to_string.json`: JSON file mapping integer IDs back to diagnosis.
- `main.py`: Main entry point of the project.

Upon execution, the following directory will also be generated:

- `logs/`: Contains training logs compatible with TensorBoard.



## 3. Usage

### 3.1 Installing requirements
Before running the code, install all required packages with:

```bash
pip install -r requirements.txt
```

### 3.2 Running the code
Execute the project by running the `main.py` script. Hyperparameters can be 
customized by using:

```commandline
main.py --hyperparameter_name hyperparameter_value
```
Alternatively, all default values can be directly modified in the code. 

The most important hyperparameter is the `--mode`, which controls the execution flow. It accepts the following values:
- `training`: fine-tune and evaluate a BiomedBERT model on the symptom_to_diagnosis dataset.
- `evaluate`: load the model pushed in the Hugging Face Hub and evaluate its performance.
- `demo`: launch an interactive demo using the model hosted on the Hugging Face Hub.

Please refer to the `main.py` file for a full 
list of configurable hyperparameters.


### 3.3 Viewing Logs

If run in `training` mode, logs will be generated and saved into the `logs/` folder. These can be read using TensorBoard
```commandline
tensorboard --logdir PATH_TO_LOG_FOLDER
```
and then access to the dashboard via the URL displayed in your terminal (usually something that http://localhost:6006/).

## 4. Performance and limitations
The model pushed on the Hugging Face Hub (the best model found during our experiments) achieves an accuracy of **96.70%** on the validation set. However, several limitations should be considered:
- Only the BiomedBERT architecture has been tested, and minimal hyperparameter tuning has been conducted. Therefore the model is likely to be suboptimal.
- The dataset is relatively small and limited to the English language, which may restrict the model’s generalization capabilities.



