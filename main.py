from load_data import collect_data
from SVM import SVM
import numpy as np

x_train, y_train = collect_data("train")
x_test, y_test = collect_data("test")

indices = np.arange(x_train.shape[0]) 
np.random.shuffle(indices)

normal_indices = np.where(y_train == 1)[0]
pneumonia_indices = np.where(y_train == -1)[0]

# 2. Lấy số lượng bằng với nhóm thiểu số (1341)
min_samples = len(normal_indices)
# Chọn ngẫu nhiên 1341 ảnh viêm phổi từ 3875 ảnh
pneumonia_indices_balanced = np.random.choice(pneumonia_indices, min_samples, replace=False)

# 3. Gộp lại và xáo trộn (Shuffle)
balanced_indices = np.concatenate([normal_indices, pneumonia_indices_balanced])
np.random.shuffle(balanced_indices)

# 4. Gán lại tập Train đã cân bằng
X_train_balanced = x_train[balanced_indices]
y_train_balanced = y_train[balanced_indices]

model = SVM(
    lr=0.001, 
    n_epochs=1000
)

model.fit(X_train_balanced, y_train_balanced)
model.predict(x_test)

accuracy, precision, recall, f1 = model.metrics(X_train_balanced, y_train_balanced)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")