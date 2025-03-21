import numpy as np
import torch

def compute_metrics(trackers):
    """
    Compute all metrics we want to keep track during training and evaluation.

    :trackers: dict containing keyword 'predictions' and 'labels'
    :return: dict containing name of the metrics as keyword and its value as value.
    """
    all_metrics = {
        'accuracy': np.nan,
        'loss': np.nan,
    }
    # cat predictions, labels and loss
    predictions = torch.cat(trackers['predictions'])
    labels = torch.cat(trackers['labels'])
    losses = torch.stack(trackers['loss'])
    
    # compute accuracy
    all_metrics['accuracy'] = (predictions == labels).sum() / len(predictions)

    # compute the mean of the loss
    all_metrics['loss'] = losses.mean()

    return all_metrics
