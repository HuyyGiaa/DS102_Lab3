import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SVM:
    def __init__(self, C: float = 1.0, lr: float = 0.01, n_epochs: int = 1000):
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.losses = []
        self.w = None
        self.b = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_epochs):
            y_pred_full = self.predict(X)
            loss = self.loss_fn(y, y_pred_full)
            self.losses.append(loss)

            #SGD
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i] 
                
                # Tính giá trị dự đoán cho 1 sample
                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    dw = self.w
                    db = 0
                else:
                    dw = self.w - self.C * (y_i * x_i)
                    db = -self.C * y_i 

                # Cập nhật Gradient
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X: np.ndarray):
        return np.dot(X, self.w) + self.b

    def loss_fn(self, y: np.ndarray, y_pred: np.ndarray):
        l2_reg = 0.5 * np.dot(self.w, self.w)
        hinge_loss = self.C * np.maximum(0, 1 - y * y_pred).sum()
        return l2_reg + hinge_loss
    
    def metrics(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        y_pred_classes = np.where(y_pred >= 0, 1, -1)
        
        accuracy = accuracy_score(y, y_pred_classes)
        precision = precision_score(y, y_pred_classes)
        recall = recall_score(y, y_pred_classes)
        f1 = f1_score(y, y_pred_classes)
        
        return accuracy, precision, recall, f1