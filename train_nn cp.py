import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from models import SimpleNNModel, NNModelBatchNormalization, TransformerModel, CNNModel
from losses import focal_loss, LDAMLoss
from torch.utils.data import Subset
from utils_cp import conformal_classification, predict_with_conformal
from imbalance_tillage_data import ImbalanceTillageDataset
from utils_plots import plot_coverage_by_pred_set_size

def main(): 
    args = parser.parse_args()
    # Fix random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    path_to_data = os.path.dirname(os.path.abspath(__file__))+ '/'
    lsat_data = pd.read_csv(path_to_data + "training_data/lsat_data.csv") # Landsat data
    s1_data = pd.read_csv(path_to_data + "training_data/s1_data.csv") # Sentinel 1 data
    cdl_data = pd.read_csv(path_to_data + "training_data/cdl_df.csv") # CDL data
    # Encode crop type
    to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
    lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace(to_replace)
    lsat_data = lsat_data.set_index("pointID")
    # Map tillage classes and residue cover ranges for confusion matrix
    tillage_mapping = {"ConventionalTill": "CT", "MinimumTill": "MT", "NoTill-DirectSeed": "NT"}
    residue_mapping = {1: "Grain", 2: "Legume", 3: "Canola"}
    # Apply the mappings to your dataframe
    tillage = lsat_data["Tillage"].map(tillage_mapping)
    crop_res = lsat_data["cdl_cropType"].map(residue_mapping)    
    x_imagery = lsat_data.loc[:, "B_S0_p0":] #lost the index as point id

    # Apply normalization
    x_imagery_index=x_imagery.index
    if args.feature_scaling:
        scaler = StandardScaler()
        x_imagery = scaler.fit_transform(x_imagery)
    # Apply feature selection with PCA
    if args.feature_selection:
        pca = PCA(n_components=0.7)
        x_imagery = pca.fit_transform(x_imagery)
    if args.feature_scaling or args.feature_selection:
        x_imagery = pd.DataFrame(x_imagery, index=x_imagery_index)
    # include residueCov feature
    if args.include_residue:
        X = pd.concat([lsat_data["cdl_cropType"], lsat_data['ResidueCov'], x_imagery,],axis=1,)
        encoder_ResidueCov = LabelEncoder()
        X['ResidueCov'] = encoder_ResidueCov.fit_transform(X['ResidueCov'])
    else:
        X = pd.concat([lsat_data["cdl_cropType"], x_imagery,],axis=1,)
    X.columns = X.columns.astype(str)
    # target variable
    y_res = lsat_data["ResidueCov"]
    y_til = lsat_data["Tillage"]
    lsat_data['y_res_til'] = lsat_data['ResidueCov'] + ' ' + lsat_data['Tillage']
    y_res_til = lsat_data['y_res_til']
    lsat_data['y_res_til_group'] = lsat_data['ResidueCov'] + ' ' + lsat_data['Tillage'] + ' ' + lsat_data["cdl_cropType"].astype(str)
    y_classes = lsat_data['y_res_til_group']
    try:
        y_target = locals()[f"y_{args.target_variable}"]
        print(f"Target variable chosen: {args.target_variable}")
    except KeyError:
        print('Error: Choose a valid target variable name (e.g., til, res, res_til)')
        exit()
    if args.loss == 'LDAM':
        dataset = ImbalanceTillageDataset(X, y_target, imb_type='exp', imb_factor=0.5, rand_number=args.seed)
        X = dataset.X
        #y_target = dataset.y
        y_res_til = y_res_til.loc[y_res_til.index.intersection(X.index)] 
        y_res = y_res.loc[y_res.index.intersection(X.index)]
        y_til = y_til.loc[y_til.index.intersection(X.index)]
        criterion = LDAMLoss(cls_num_list=dataset.get_cls_num_list(), max_m=0.5, s=30)
        print("Class distribution after imbalance:"+ str(dataset.get_cls_num_list()))
        groups = X["cdl_cropType"]
        stratify_column = pd.DataFrame({"y_res": y_res, "y_til": y_til})
    elif args.loss == 'CE':
        criterion = nn.CrossEntropyLoss()
        groups = X["cdl_cropType"]
        stratify_column = pd.DataFrame({"y_res": y_res, "cdl_cropType": groups, "y_til": y_til})
    else:
        print('Error: Choose a valid loss type (e.g., CE, LDAM.)')
        exit()
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in sss.split(X, stratify_column):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]
    # Get the unique classes and their counts in the target variable
    unique_classes, class_counts = np.unique(y_res_til, return_counts=True)

    # Encode target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    # Definition of DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Definition of the model and its parameters
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)  # Number of unique classes in y_res_til
    try:
        model = globals()[args.model](input_size, num_classes)
        print(f"Model chosen: {type(model).__name__}")
    except KeyError:
        print('Choose a valid model')
        exit()
    # Definition of the loss and optimizer
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        y_train_pred = []
        y_train_true = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Track predictions for train accuracy
            _, preds = torch.max(outputs, 1)
            y_train_pred.extend(preds.numpy())
            y_train_true.extend(y_batch.numpy())

        # Calculate train accuracy for this epoch
        train_accuracy = accuracy_score(y_train_true, y_train_pred)
        
        # Evaluation
        model.eval()
        y_pred, y_true = [], []
        y_til_pred, y_til_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                y_pred.extend(preds.numpy())
                y_true.extend(y_batch.numpy())
        test_accuracy = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f} , Test Accuracy: {test_accuracy:.4f}")

    # Map predictions back to original labels
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    if args.CP:
        # Split Test Set into Calibration and Test Sets
        test_indices = list(range(len(test_dataset)))
        split_idx = int(0.5 * len(test_indices))  # Split equally into calibration and test sets
        calib_indices, test_indices = test_indices[:split_idx], test_indices[split_idx:]
        calib_loader = DataLoader(Subset(test_dataset, calib_indices), batch_size=32, shuffle=False)
        final_test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=32, shuffle=False)

        # Calibration
        alpha = args.alpha  # Miscoverage level
        threshold = conformal_classification(model, calib_loader, alpha)
        print(f"Conformal Calibration Threshold: {threshold:.4f}")

        # Prediction sets 
        prediction_sets = predict_with_conformal(model, final_test_loader, threshold)

        # Evaluation of Coverage 
        y_test_final = [y for _, y in Subset(test_dataset, test_indices)]
        correct = sum(prediction_sets[i, label] for i, label in enumerate(y_test_final))
        coverage = correct / len(y_test_final)
        print(f"Coverage: {coverage:.3f} (Target: {1 - alpha})")

        # Evaluation of Average Set Size
        pred_set_sizes = np.sum(prediction_sets, axis=1)
        avg_pred_set_size = np.mean(pred_set_sizes)
        print(f"Average Prediction Set Size: {avg_pred_set_size:.3f}")
        plot_coverage_by_pred_set_size(prediction_sets, y_test_final, args)
        print('s')
    # done
    print('Done')

parser = argparse.ArgumentParser(description='Tillage classification')
# general_params
parser.add_argument('--seed', default=42, type=int, help='randoom seed')
# data params
parser.add_argument('--feature_scaling', default=True, type=bool, help='Feature scaling') # done
parser.add_argument('--feature_selection', default=True, type=bool, help='Feature selection') # done
parser.add_argument('--include_residue', default=False, type=bool, help='include residue cover feature') # done
parser.add_argument('--target_variable', default="til", type=str, help='target variable to predict') # (e.g., til, res, res_til)
# model params
parser.add_argument('--model', default="NNModelBatchNormalization", type=str, help='neural network model') # SimpleNNModel, NNModelBatchNormalization, TransformerModel, CNNModel # done
parser.add_argument('--loss', default="CE", type=str, help='loss type') # CE, LDAM
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run') # done
# CP params
parser.add_argument('--CP', default=True, type=bool, help='target miscoverage')
parser.add_argument('--alpha', default=0.05, type=float, help='target miscoverage') #0.1, 0.05
# plot params
parser.add_argument('--save_cp_fig', default=True, type=bool, help='save CP figure')

if __name__ == '__main__':
    main()


# 3 classes vs 9 classes (1h) # 3
# PCA vs without PCA (1h) # 2
# feats vs feats+res (1h) # 2
# loss CE vs loss LDAM + plots (1h) # 2


# CP + plots (3h)
# ensembles (1h)