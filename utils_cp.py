import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

# Function to calculate conformal scores and prediction sets
def conformal_classification(model, calib_loader, alpha):
    model.eval()
    scores = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in calib_loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            batch_scores = 1 - probs.gather(1, y_batch.unsqueeze(1)).squeeze()
            scores.extend(batch_scores.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    
    scores = np.array(scores)
    threshold = np.quantile(scores, 1 - alpha)  # Calculate threshold for 1-alpha quantile
    return threshold

def predict_with_conformal(model, test_loader, threshold):
    model.eval()
    prediction_sets = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            prediction_set = probs >= (1 - threshold)  # Threshold to determine prediction set
            prediction_sets.append(prediction_set.cpu().numpy())
    return np.vstack(prediction_sets)