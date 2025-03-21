from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from accelerate import Accelerator
import json


class OurDataset:
    def __init__(self, config):
        """
        Implementation of a dataset using the Hugging Face library.

        :param config: configuration used to train and evaluate the model. See parse_args() in main.py for more details.
        """
        # saving all parameters
        self.config = config

        # loading the raw dataset
        self.raw_datasets = load_dataset(self.config.dataset_checkpoint)


        # loading the tokenize associated to the checkpoint model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

        self.num_label = len(set(self.raw_datasets['train']['output_text']))


        with open("id_to_string.json", "rb") as f:
                self.id_to_string = json.load(f)
        
        with open("string_to_id.json", "rb") as f:
                self.string_to_id = json.load(f)

        # rename columns
        self.raw_datasets = self.raw_datasets.rename_column("output_text", "label_ids")

        # tokenizing the dataset using some customized tokenization function
        self.tokenized_datasets = self.raw_datasets.map(self._tokenize_function, batched=True)

        # Remove original columns that are not used by the model
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(["input_text", "label_ids"])

        # ensures everything is torch tensor
        self.tokenized_datasets.set_format("torch")

        # loading the data collator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Train and eval dataloaders construction
        self.train_dataloader = DataLoader(self.tokenized_datasets["train"], 
                                           shuffle=self.config.shuffle, 
                                           batch_size=self.config.batch_size,
                                           collate_fn=self.data_collator)
        
        self.eval_dataloader = DataLoader(self.tokenized_datasets["test"], 
                                          batch_size=self.config.batch_size, 
                                          shuffle=self.config.shuffle,
                                          collate_fn=self.data_collator)
        
        # prepare dataloard using accelaratore
        accelerator = Accelerator()
        self.train_dataloader, self.eval_dataloader = accelerator.prepare(self.train_dataloader, self.eval_dataloader)
        

    def _tokenize_function(self, example):
        """
        Tokenization function used to tokenize the dataset.

        :param example: text to tokenize.
        :return: tokenized text.
        """
        # convert label text into id
        labels = [self.string_to_id[text] for text in example['label_ids']]

        tokenized =  self.tokenizer(example["input_text"], padding="max_length", truncation=True)

        tokenized['label'] = labels

        return tokenized
    
    def push_tokenizer_to_hub(self, repo_name):
        """
        Push the tokenizer to the hub.
        """
        self.tokenizer.push_to_hub(repo_name)







