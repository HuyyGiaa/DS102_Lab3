# Chest X-Ray Pneumonia Detection: SVM from Scratch

This repository contains a complete Machine Learning pipeline for classifying Chest X-Ray images into **Normal** and **Pneumonia** categories. The core feature of this project is a custom implementation of a **Support Vector Machine (SVM)** algorithm built entirely from scratch using NumPy, optimized via Stochastic Gradient Descent (SGD).

## 🚀 Features

- **Custom SVM Implementation:** Built from scratch utilizing Hinge Loss, L2 Regularization, and SGD without relying on external ML libraries for the core logic.
- **Image Preprocessing:** Includes automated grayscale conversion, image resizing (128x128), and feature standardization (pixel-wise Mean and Standard Deviation scaling).
- **Data Balancing:** Implements an undersampling technique to handle class imbalance between Normal and Pneumonia datasets.
- **Library Comparison:** Features a direct, fair performance comparison module against `scikit-learn`'s `SGDClassifier`.
- **Modular Architecture:** Clean, production-ready project structure separating data loading, preprocessing, model training, and evaluation.

## 📂 Project Structure

```text
DS102_LAB3/
├── archive/              # Raw dataset (Ignored by Git)
├── notebooks/            # Jupyter notebooks for mathematical proofs (SMO, Lagrange, etc.)
├── output/               # Generated assets (Training loss curves, comparison bar charts)
├── src/                  # Core modules
│   ├── load_data.py            # Image reading and matrix conversion (OpenCV)
│   ├── preprocessing.py        # Standardization and data balancing
│   ├── SVM.py                  # The custom SVM class (from scratch)
│   └── comparison_svm_lib.py   # Scikit-learn model and evaluation scripts
├── main.py               # The main pipeline orchestrator
├── requirements.txt      # Python dependencies
└── .gitignore            # Git configuration

📊 Dataset

The model is trained on the standard Chest X-Ray Images (Pneumonia) dataset (typically sourced from Kaggle).

    Ensure the dataset is extracted and placed inside the archive/chest_xray/ directory.

    The expected subdirectories are train/, test/, and val/, each containing NORMAL and PNEUMONIA image folders.

⚙️ Installation & Setup

    Clone the repository:
    Bash

    git clone <https://github.com/HuyyGiaa/DS102_Lab3>
    cd DS102_LAB3


2. **Install the required dependencies:**
   It is recommended to use a virtual environment (like Conda or venv). Run the following command to install all necessary packages:
   ```bash
   pip install -r requirements.txt
   

🏃‍♂️ Usage

To run the entire pipeline—from loading data to training both models and generating the comparison charts—simply execute the main.py file:
Bash

python main.py

What happens when you run this?

    Images are loaded, resized, and scaled to [0, 1].

    The dataset is balanced using random undersampling.

    Features are standardized using column-wise Mean and Std.

    The custom SVM trains for 50 epochs, tracking the loss function.

    The sklearn SVM trains for 50 epochs with a constant learning rate for a fair comparison.

    The terminal outputs evaluation metrics (Precision, Recall, F1-Score).

    Visualizations (loss.png and Comparision_2_models.png) are saved directly to the output/ folder.

📈 Results

Our custom-built SVM demonstrates outstanding performance, successfully competing with the standard scikit-learn library.

    Precision: ~ 0.98

    Recall: ~ 0.96

    F1-Score: ~ 0.97

(Check the output/ folder for the detailed training loss curves and the side-by-side metric comparison bar chart).