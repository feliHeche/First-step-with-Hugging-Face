from OurDataset import OurDataset
from CustomizedTrainer import CustomizedTrainer
import argparse




def parse_args():
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument('--seed_value', action='store', type=int, default=42)      

    # checkpoint
    parser.add_argument('--checkpoint', action='store', type=str, default="bert-base-uncased",   
                        help='checkpin to load the model used in this project.')
    
    # dataset
    parser.add_argument('--data_name', action='store', type=str, default="glue",   
                        help='define the dataset name used in this project.')
    parser.add_argument('--data_type', action='store', type=str, default="mrpc",   
                        help='define the dataset name used in this project.')
    
    # training parameters
    parser.add_argument('--batch_size', action='store', type=int, default=8, help='batch size used in the training.')
    parser.add_argument('--lr', action='store', type=float, default=5e-5, help='learning rate used in the training.')
    parser.add_argument('--epochs', action='store', type=int, default=3, help='number of epochs used in the training.')
    parser.add_argument('--optimizer', action='store', type=str, default="adamw", help='optimizer used in the training.')
    parser.add_argument('--shuffle', action='store', type=bool, default=True, help='shuffle the dataset or not.')

    # tensorboard parameters
    parser.add_argument('--training_average_metric', action='store', type=int, default=100, help='average metric used in the training.')

    args = parser.parse_args()

    return args


def main():

    # loadding all parameters
    config = parse_args()


    # building the dataset
    dataset = OurDataset(config=config)

    # building the LLM
    model = CustomizedTrainer(config=config, dataset=dataset)

    # model training
    model.training()



if __name__=="__main__":

    main()


