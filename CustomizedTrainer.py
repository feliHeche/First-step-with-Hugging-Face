from transformers import AutoModelForSequenceClassification
from OurDataset import OurDataset
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from accelerate import Accelerator
from huggingface_hub import get_full_repo_name
import os
import gradio as gr
import re
from utils import compute_metrics



class CustomizedTrainer:
    def __init__(self, config, dataset: OurDataset):
        """
        Implementation of a NLP model using the Hugging Face ecosystem.

        :param config: configuration used to train and evaluate the model. See parse_args() in main.py for more details.
        :parma dataset: dataset used to train the model. Expected to be an instance of OurDataset.
        """
        # loading all parameters
        self.config = config

        # our customized dataset
        self.dataset = dataset

        # loading pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.checkpoint, num_labels=self.dataset.num_label)

        # optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError("Optimizer {} not implemented.".format(self.config.optimizer))
        
        # compute learning rate schduler
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.config.epochs*len(self.dataset.train_dataloader)
        )

        # if training mode, we initialize tensorboard
        if self.config.mode == "training":
            self._initialize_tensorboard()

        # accelerator used to handle different working environments (gpu vs multiple gpu vs tpu)
        self.accelerator = Accelerator()
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # settings the repo where we push the model -> only if required
        if self.config.save_model:
            os.makedirs("trained-model", exist_ok=True)
    
    def _initialize_tensorboard(self):
        """
        Initialize tensorboard. 

        Build a tensorboard file and log hyperparameters. Only called in the training mode.
        """
        
        # log folder
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # tensorboard writer
        log_dir = 'logs/' + current_time
        self.log_writer = SummaryWriter(log_dir)

        # Convert hyperparameters to formatted text
        hyperparameters = "\n".join([f"{k}: {getattr(self.config, k)}" for k in vars(self.config)])
        # Log to TensorBoard
        self.log_writer.add_text("hyperparameters", hyperparameters, global_step=0)



    def training(self):
        """
        Main training loop.
        """ 

        print('#'*50 + " Model Performance before fine-tuning " + '#'*50)
        self._eval_model()
        print('#'*138)

        for tmp in range(self.config.epochs):
            print(('#'*50 + " Training epoch {}/" + str(self.config.epochs) + ' ' + '#'*50).format(tmp+1))
            self._train_model_one_epoch(tmp=tmp*len(self.dataset.train_dataloader))
            self._eval_model(step=tmp+1)
            # local model saving
            self.model.save_pretrained("trained-model/" + self.config.model_name  + '_epoch_' + str(tmp+1))
            print('#'*119)
        
        # push the final model to the hub if required
        if self.config.push_to_hub:
            # push the model to the hub
            repo_name = get_full_repo_name(self.config.model_name)
            self.model.push_to_hub(repo_name)
            self.dataset.push_tokenizer_to_hub(repo_name=repo_name)
            print("Final model has been pushed to the hub.")


    def _train_model_one_epoch(self, tmp=0):
        """
        Train the model for one epoch.

        :param tmp: number of training steps already done.
        """
        # Used to track accuracy and loss
        trackers = {
            'predictions': [],
            'labels': [],
            'loss': [],
        }

        progress_bar = tqdm(total=len(self.dataset.train_dataloader), desc="Training")
        self.model.train()
        
        for batch in self.dataset.train_dataloader:
        
            outputs = self.model(**batch)
            loss = outputs.loss

            # compute gradient
            self.accelerator.backward(loss)

            # update the model
            self.optimizer.step()
            # update the learning rate
            self.lr_scheduler.step()
            # clear gradient before the next batch
            self.optimizer.zero_grad()

            # add measure to our trackers
            trackers['predictions'].append(torch.argmax(outputs.logits, dim=-1))
            trackers['labels'].append(batch['labels'])
            trackers['loss'].append(outputs.loss)

            # compute metrics
            computed_metrics = compute_metrics(trackers)

            # writing in tensorboard if needed
            if (progress_bar.n > 0) & (progress_bar.n % self.config.training_average_metric == 0):
                self.log_writer.add_scalar("Training: loss", computed_metrics['loss'], tmp+progress_bar.n)
                self.log_writer.add_scalar("Training: accuracy", computed_metrics['accuracy'], tmp+progress_bar.n)
                
                # reinitialization of the trackers
                trackers = {
                    'predictions': [],
                    'labels': [],
                    'loss': [],
                }

            progress_bar.update(1)

    
    def _eval_model(self, step=0):
        """
        Evaluate the model.

        :param step: number of evaluations already done.
        """
        # trackers that track predictions and labels
        trackers = {
            'predictions': [],
            'labels': [],
            'loss': [],
        }

        print('Model evaluation:')
        self.model.eval()
        for batch in self.dataset.eval_dataloader:

            # ensure no gradient is computed
            with torch.no_grad():
                outputs = self.model(**batch)
    
            predictions = torch.argmax(outputs.logits, dim=1)

            # add measure to the trackers
            trackers['predictions'].append(predictions)
            trackers['labels'].append(batch['labels'])
            trackers['loss'].append(outputs.loss)

        computed_metrics = compute_metrics(trackers=trackers)
        for (k, v) in computed_metrics.items():
            print(f"{k}: {v*100:.2f}%")

            # writing metrics in tensorboard if training mode
            if self.config.mode == "training":
                self.log_writer.add_scalar("Evaluation: " + str(k), v, step)
    

    def _load_trained_model(self):
        """
        Load the trained model.
        """
        checkpoint_name = get_full_repo_name(self.config.model_name)

        # overwiting our model with the already trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name, num_labels=self.dataset.num_label)

        # put the model on the same device as the dataset
        self.model = self.accelerator.prepare(self.model)
        print('Model has been successfully loaded.')
    

    def evaluation(self):
        """
        Load the final model and evaluate it.

        Note that the model should have been trained and pushing to the hub before calling this function.
        """
        print("We are in evaluation mode.")
        self._load_trained_model()

        # Model evaluation
        self._eval_model()

    def demo(self):
        """
        Run a demo of the model.
        """
        # loading the best model
        self._load_trained_model()

        title = 'Symptom to diagnosis'

        description = """
        ## 1. Overview
         This model performs automatic diagnosis based on test descriptions. It classifies the input text into one of 22 predefined diagnostic categories:
         | Category                 | Category         | Category                        | Category              | Category                 |
         |--------------------------|------------------|---------------------------------|-----------------------|--------------------------|
         | allergy                  | common cold      | gastroesophageal reflux disease | migraine              |  urinary tract infection |
         | arthritis                | dengue           | hypertension                    | peptic ulcer disease  | varicose veins           |
         | bronchial asthma         | diabetes         | impetigo                        | pneumonia             |
         | cervical spondylosis     | drug reaction    | jaundice                        | psoriasis             |
         | chicken pox              | fungal infection | malaria                         | typhoid               | 
        ## 2. Model and dataset  
        This model is built upon [BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract), a large language model pretrained on PubMed abstracts.  
        It was then fine-tuned using the [symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) dataset, which contains 835 training examples and 212 validation examples.

        For further details, please refer to the [GitHub repository](https://github.com/feliHeche/First-step-with-Hugging-Face) associated with this project.

        ## 3. Performance and limitations
        The model achieves an accuracy of **92.45%** on the validation set. However, several limitations should be considered:

        - Only the BiomedBERT architecture was tested, with minimal hyperparameter tuning. The current model is likely suboptimal and could benefit from further experimentation.
        - The dataset is relatively small and limited to the English language, which may restrict the model’s generalization capabilities.
        - The labels in the dataset have not been manually verified by medical professionals, raising potential concerns about label quality.


        ## 4. Model interface
        Users may enter a symptom description in the left-hand input box. The model will output the predicted diagnosis in the right-hand result panel.
        """
        

        example = [['I\'ve been having trouble sleeping because of the rash. It\'s itchy and painful. I\'ve also noticed that my nails have little dents in them, which is really alarming. And my joints are sore every day, and I don\'t know why.'],
                   ['I have a rash on my skin that is red, raised, and filled with fluid. It is painful to touch and I have a fever.'],
                   ['I\'ve been having headaches, chest discomfort, dizziness, and difficulty concentrating.']
                   ]

        demo = gr.Interface(fn=self._managing_demo_input, 
                            inputs="text", 
                            outputs="text", 
                            title=title,
                            description=description,
                            examples=example)
        
        demo.launch(share=True)
    

    def _managing_demo_input(self, text_input):
        """
        Just a simple demo for our model.
        """

        # poreprocessing the input as expected by the model
        text_input = self.dataset.tokenizer(text_input, truncation=True, return_tensors='pt')

        # putting the input text in the same device as the model
        text_input = {k: v.to(self.model.device) for k, v in text_input.items()}

        # predict label 
        output = self.model(**text_input)
        output = torch.argmax(output.logits, dim=1).item()

        # converting label into readable predicion
        text_prediction = self.dataset.id_to_string[str(output)]

        return text_prediction



