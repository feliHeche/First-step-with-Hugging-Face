from transformers import AutoModelForSequenceClassification
from OurDataset import OurDataset
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
from torch.utils.tensorboard import SummaryWriter
import datetime



class CustomizedTrainer:
    def __init__(self, config, dataset: OurDataset):
        """
        Implementation of a LLM model using the Hugging Face library.

        :param checkpoint: checkpoint used to load the model
        :parma dataset: dataset used to train the model. Expected to be an instance of OurDataset.
        """
        # loading all parameters
        self.config = config

        # loading pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.checkpoint, num_labels=2)

        # our customized dataset
        self.dataset = dataset

        # optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError("Optimizer {} not implemented.".format(self.config.optimizer))
        
        # compute learning rate schduler
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.config.epochs*len(self.dataset.train_dataloader)
        )

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


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
            print('#'*119)


    def _train_model_one_epoch(self, tmp=0):
        """
        Train the model for one epoch.
        """
        # Used to track accuracy and loss
        running_loss = 0.0
        running_correct_predictions = 0

        progress_bar = tqdm(total=len(self.dataset.train_dataloader), desc="Training")
        self.model.train()
        
        for batch in self.dataset.train_dataloader:
            # move the batch to the device
            batch = {k: v.to(self.device) for k, v in batch.items()}
        
            outputs = self.model(**batch)
            loss = outputs.loss
            # compute gradient
            loss.backward()

            # update the model
            self.optimizer.step()
            # update the learning rate
            self.lr_scheduler.step()
            # clear gradient before the next batch
            self.optimizer.zero_grad()

            # update metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            running_correct_predictions += torch.sum(predictions == batch["labels"]).item()

            # writing in tensorboard if needed
            if (progress_bar.n > 0) & (progress_bar.n % self.config.training_average_metric == 0):
                self.log_writer.add_scalar("Training: loss", running_loss, tmp+progress_bar.n)
                self.log_writer.add_scalar("Training: accuracy", running_correct_predictions/(self.config.training_average_metric*self.config.batch_size), 
                                           tmp+progress_bar.n)
                
                # settting tracker back to 0
                running_loss = 0.0
                running_correct_predictions = 0

            progress_bar.update(1)

    
    def _eval_model(self, step=0):
        """
        Evaluate the model.
        """
        metric = evaluate.load(self.config.data_name, self.config.data_type)
        print('Model evaluation:')
        self.model.eval()
        for batch in self.dataset.eval_dataloader:
            # move the batch to the device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # ensure no gradient is computed
            with torch.no_grad():
                outputs = self.model(**batch)
            
            predictions = torch.argmax(outputs.logits, dim=1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        
        computed_metrics = metric.compute()
        for (k, v) in computed_metrics.items():
            print(f"{k}: {v}")

            # writing metrics in tensorboard
            #title = "Evaluation: " + str(k)
            self.log_writer.add_scalar("Evaluation: " + str(k), v, step)




