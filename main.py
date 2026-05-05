from src.load_data import collect_data
from src.preprocessing import preprocess, balance_data
from src.SVM import SVM
import numpy as np
import matplotlib.pyplot as plt
from src.comparison_svm_lib import train_svm_sklearn, evaluate, plot_comparison

def plot_loss(losses):
    plt.figure(figsize=(10, 4))
    plt.plot(losses, color='blue', linewidth=1.5)
    plt.title("Training Loss per ecpoch")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/loss.png', dpi=300)
    plt.show()
    

if __name__ == "__main__":
    x_train, y_train = collect_data("train")
    x_test, y_test = collect_data("test")

    model_from_scratch = SVM(
        lr=0.0001, 
        n_epochs=50
    )

    x_train, y_train = balance_data(x_train, y_train)
    x_train, x_test = preprocess(x_train, x_test)
    
    y_train = np.where(y_train <= 0, -1, 1)
    y_test = np.where(y_test <= 0, -1, 1)
    #Label -1 vs 1
    model_from_scratch.fit(x_train, y_train)
    model_from_scratch.predict(x_test)

    precision_from_scratch, recall_from_scratch, f1_from_sractch = model_from_scratch.metrics(x_train, y_train)
    print(f"Precision: {precision_from_scratch}")
    print(f"Recall: {recall_from_scratch}")
    print(f"F1 score: {f1_from_sractch}")
    results_from_scratch = [precision_from_scratch, recall_from_scratch, f1_from_sractch]
    
    plot_loss(model_from_scratch.losses)
    
    # model from sklearn
    model_sklearn = train_svm_sklearn(x_train, y_train)
    metrics_sklearn = evaluate(model_sklearn, x_test, y_test)
    results_from_sklearn = list(metrics_sklearn)
    
    plot_comparison(results_from_sklearn, results_from_scratch)
    