import numpy as np
import pandas as pd
import torch

class ImbalanceTillageDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, imb_type='exp', imb_factor=0.01, rand_number=0):
        """
        X: pandas DataFrame containing features.
        y: pandas Series containing targets (class labels).
        imb_type: Type of imbalance ('exp' or 'step').
        imb_factor: Factor to control imbalance severity (default: 0.01).
        rand_number: Random seed for reproducibility.
        """
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.cls_num = len(self.classes)
        np.random.seed(rand_number)

        # Generate number of samples per class
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.num_per_cls_dict = {cls: num for cls, num in zip(self.classes, img_num_list)}

        # Generate imbalanced data
        self.gen_imbalanced_data()

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        """
        Generate the number of samples per class based on imbalance type.
        """
        img_max = len(self.y) / cls_num  # Maximum samples per class
        img_num_per_cls = []

        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:  # Default to balanced
            img_num_per_cls.extend([int(img_max)] * cls_num)

        return img_num_per_cls

    def gen_imbalanced_data(self):
        """
        Generate imbalanced data by sampling rows from each class.
        """
        new_X = []
        new_y = []
        
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]  # Indices for the class
            np.random.shuffle(cls_indices)  # Shuffle to randomly select samples
            
            num_samples = self.num_per_cls_dict[cls]
            selected_indices = cls_indices[:num_samples]
            
            new_X.append(self.X.iloc[selected_indices])
            new_y.append(self.y.iloc[selected_indices])
        
        # Concatenate the imbalanced data
        #self.X = pd.concat(new_X).reset_index(drop=True)
        #self.y = pd.concat(new_y).reset_index(drop=True)
        self.X = pd.concat(new_X)
        self.y = pd.concat(new_y)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), torch.tensor(self.y.iloc[idx], dtype=torch.long)

    def get_cls_num_list(self):
        """
        Return a list of the number of samples per class in the dataset.
        """
        return [self.num_per_cls_dict[cls] for cls in self.classes]


# Example Usage
if __name__ == "__main__":
    # Sample data
    X = pd.DataFrame(np.random.randn(1000, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    y = pd.Series(np.random.choice([0, 1, 2, 3], size=1000))

    # Create imbalanced dataset
    dataset = ImbalanceTillageDataset(X, y, imb_type='exp', imb_factor=0.5, rand_number=42)
    print("Class distribution after imbalance:"+ str(dataset.get_cls_num_list()))

    # Check the first sample
    features, target = dataset[0]
    print("First sample features:", features)
    print("First sample target:", target)
