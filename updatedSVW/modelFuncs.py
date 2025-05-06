import tqdm
import torch 
from torch import Tensor as T

def separateData(data):
    feats = []
    labels = []
    for context, info, label in data:
        feats.append(context)
        labels.append(label)
    labels = T(labels).type(torch.LongTensor)
    return feats, labels

def accuracy(labels, probs):
    return (torch.argmax(labels, 1) == torch.argmax(probs, 1)).float().mean()

def eval(dataLoader, model):
    acc = 0
    # totalPreds = torch.empty(7)
    model.eval()
    for i, data in enumerate(dataLoader):
        preds = model(data[0])
        # torch.cat((totalPreds, preds), 0)
        acc += (torch.argmax(preds, 1) == torch.argmax(data[2], 1)).float()
        # ce = criterion(preds, labels)
    return float(acc.mean())#, totalPreds