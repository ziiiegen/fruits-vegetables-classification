import os
import random
import torch
import numpy as np
from sklearn.metrics import classification_report

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def sklearn_report(model, dataloader, device, idx2class, digits=4):
    model.eval()
    y_true, y_pred = [], []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

    labels = sorted(idx2class.keys())
    target_names = [idx2class[i] for i in labels]
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=digits, zero_division=0))