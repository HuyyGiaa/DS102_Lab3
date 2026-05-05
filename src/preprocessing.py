import numpy as np

def balance_data(x_train, y_train):
    normal_indices = np.where(y_train == -1)[0]
    pneumonia_indices = np.where(y_train == 1)[0]

    min_samples = len(normal_indices)
    pneumonia_indices_balanced = np.random.choice(pneumonia_indices, min_samples, replace=False)

    balanced_indices = np.concatenate([normal_indices, pneumonia_indices_balanced])
    np.random.shuffle(balanced_indices)
    
    X_train_balanced = x_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]
    
    return X_train_balanced, y_train_balanced

def preprocess(x_train, x_test):
    mean_train = x_train.mean(axis=0)
    std_train = x_train.std(axis=0)
    # divide 0
    std_train[std_train == 0] = 1e-8
    
    x_train = (x_train - mean_train) / std_train
    x_test = (x_test - mean_train) / std_train
    
    return x_train, x_test