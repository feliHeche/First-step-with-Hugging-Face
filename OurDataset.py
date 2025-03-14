from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader



class OurDataset:
    def __init__(self, config):
        """
        Implementation of a dataset using the Hugging Face library.

        :param checkpoint: checkpoint used to load the tokenizer
        :param dataset_name: name of the dataset
        :param dataset_type: type of the dataset
        """
        # saving all parameters
        self.config = config

        # loading the raw dataset
        self.raw_datasets = load_dataset(self.config.data_name, self.config.data_type)

        # loading the tokenize associated to the checkpoint model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

        # loading the data collator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # tokenizing the dataset using some customized tokenization function
        self.tokenized_datasets = self.raw_datasets.map(lambda examples: self._tokenize_function(examples), batched=True)

        """
        preprocess the tokenized dataset to build the dataloaders
        """
        # remove columns that are not expected by our model
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        # model the column label as expected by our model
        self.tokenized_datasets = self.tokenized_datasets.rename_column("label", "labels")
        # ensure everything is using torch
        self.tokenized_datasets.set_format("torch")
        self.tokenized_datasets["train"].column_names

        # Train and eval dataloaders construction
        self.train_dataloader = DataLoader(self.tokenized_datasets["train"], 
                                           shuffle=self.config.shuffle, 
                                           batch_size=self.config.batch_size,
                                           collate_fn=self.data_collator)
        
        self.eval_dataloader = DataLoader(self.tokenized_datasets["validation"], 
                                          batch_size=self.config.batch_size, 
                                          shuffle=self.config.shuffle,
                                          collate_fn=self.data_collator)
        

    def _tokenize_function(self, example):
        """
        Tokenization function used to tokenize the dataset.
        """
        return self.tokenizer(example["sentence1"], example["sentence2"], truncation=True)







