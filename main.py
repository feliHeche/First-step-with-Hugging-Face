from OurDataset import OurDataset
from CustomizedTrainer import CustomizedTrainer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument('--seed_value', action='store', type=int, default=42)      

    # checkpoint
    parser.add_argument('--checkpoint', action='store', type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",   
                        help='checkpoint to load the model used in this project.')

    
    # main mode: training, evaluation or demo
    parser.add_argument('--mode', action='store', type=str, default="demo",   
                        help='determine the main mode. Possible values: training, evaluation, demo.')
    
    # dataset used to fine-tune our model
    parser.add_argument('--dataset_checkpoint', action='store', type=str, default="gretelai/symptom_to_diagnosis",   
                        help='define the dataset name used in this project.')

    
    # training parameters
    parser.add_argument('--batch_size', action='store', type=int, default=8, help='batch size used in the training.')
    parser.add_argument('--lr', action='store', type=float, default=5e-5, help='learning rate used in the training.')
    parser.add_argument('--epochs', action='store', type=int, default=3, help='number of epochs used in the training.')
    parser.add_argument('--optimizer', action='store', type=str, default="adamw", help='optimizer used in the training.')
    parser.add_argument('--shuffle', action='store', type=bool, default=True, help='shuffle the dataset or not.')

    # tensorboard parameters
    parser.add_argument('--training_average_metric', action='store', type=int, default=100, help='average metric used in the training.')

    # determine if we save the model during the training
    parser.add_argument('--save_model', action='store', type=bool, default=True, help='save the model during the training or not.')
    parser.add_argument('--model_name', action='store', type=str, default="bert-dummy-model", help='name of the model saved during the training.')

    # determine if the push to hub the final model
    parser.add_argument('--push_to_hub', action='store', type=bool, default=True, help='push the final model to the hub or not.')

    args = parser.parse_args()

    return args


def main():

    # loadding all parameters
    config = parse_args()


    # building the dataset
    dataset = OurDataset(config=config)

    # building the LLM
    model = CustomizedTrainer(config=config, dataset=dataset)

    # running the expected mode
    if config.mode == "training":
        model.training()
    elif config.mode == "evaluation":
        model.evaluation()
    elif config.mode == "demo":
        model.demo()
    else:
        raise ValueError("Configuration mode should be either training, evaluation or demo.")



if __name__=="__main__":

    main()


