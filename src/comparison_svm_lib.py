from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
from src.load_data import collect_data
from src.preprocessing import preprocess, balance_data
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd

def train_svm_sklearn(x_train, y_train, n_epochs=50):
    model = SGDClassifier(
        loss="hinge",
        random_state=42,
        learning_rate='constant',
        max_iter=n_epochs,
        eta0=0.0001,
        alpha=0.0001
    )
    
    model.fit(x_train, y_train)
        
    return model


def evaluate(model, x_test, y_test):
    y_predict = model.predict(x_test)
    
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    
    return precision, recall, f1
    
def plot_comparison(results_svm_lib, results_svm_scratch):
    metrics = ["Precision", "Recall", "F1"]
    
    df = pd.DataFrame({
        "Metric": metrics * 2, 
        "Score": np.concatenate((results_svm_lib, results_svm_scratch)),
        "Model": ["SVM Library"] * 3 + ["SVM Scratch"] * 3
    })
    
    plt.figure(figsize=(10, 6)) 
    
    ax = sns.barplot(x="Metric", y="Score", hue="Model", data=df)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
        
    plt.title("Comparison: SVM Library vs SVM from Scratch")
    
    plt.ylim(0, 1.1) 
    plt.savefig('output/Comparision_2_models.png', dpi=300)
    plt.show()