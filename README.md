# fmnisit-_using_ANN
# Fashion‑MNIST Classification Using Artificial Neural Network (ANN)

A clean and professional implementation of a **fully connected Artificial Neural Network (ANN)** to classify images from the **Fashion‑MNIST (FMNIST)** dataset. This project demonstrates an end‑to‑end deep learning workflow including data preprocessing, model design, training, evaluation, and result visualization.

---

## 📌 Project Overview

Fashion‑MNIST is a widely used benchmark dataset consisting of **28×28 grayscale images** of fashion products across **10 categories** such as T‑shirts, trousers, shoes, and bags. Compared to the classic MNIST digits dataset, FMNIST presents a more realistic and challenging image classification problem.

This repository focuses on solving the FMNIST classification task using a **feedforward ANN**, making it ideal for:

* Understanding neural network fundamentals
* Learning image preprocessing for deep learning
* Academic mini‑projects and beginner ML portfolios

---

## 🧠 Dataset Details

* **Dataset**: Fashion‑MNIST
* **Training samples**: 60,000
* **Test samples**: 10,000
* **Image size**: 28 × 28 (grayscale)
* **Number of classes**: 10

Each image is flattened into a **784‑dimensional feature vector** before being passed to the ANN.

---

## 🛠️ Tech Stack

* **Programming Language**: Python 3
* **Libraries & Frameworks**:

  * TensorFlow / Keras
  * NumPy
  * Matplotlib
  * Scikit‑learn
* **Development Environment**: Jupyter Notebook

---

## 🏗️ Model Architecture

The ANN model consists of multiple fully connected layers:

* **Input Layer**: 784 neurons (flattened image)
* **Hidden Layer 1**: Dense layer with ReLU activation
* **Hidden Layer 2**: Dense layer with ReLU activation
* **Output Layer**: 10 neurons with Softmax activation

**Loss Function**: Categorical Cross‑Entropy
**Optimizer**: Adam
**Evaluation Metric**: Accuracy

---

## ⚙️ Workflow

1. Load the Fashion‑MNIST dataset
2. Normalize pixel values to the range [0, 1]
3. Flatten images into 1‑D vectors
4. One‑hot encode class labels
5. Build and compile the ANN model
6. Train the model on training data
7. Evaluate performance on test data
8. Visualize accuracy, loss, and predictions

---

## 📊 Results

The ANN achieves strong baseline performance on the Fashion‑MNIST dataset, typically reaching **high classification accuracy** with proper tuning of epochs and hidden layers.

This confirms the effectiveness of feedforward neural networks for structured image‑based classification tasks.

---

## 📂 Repository Structure

```
FMNIST_Using_ANN/
│
├── fmnisit__using_ANN.ipynb   # Main notebook with full implementation
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies (optional)
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/FMNSIT_Using_ANN.git
cd FMNIST_Using_ANN
```

### 2️⃣ Install Dependencies

```bash
pip install numpy tensorflow matplotlib scikit-learn
```

### 3️⃣ Run the Notebook

Open `fmnisit__using_ANN.ipynb` in Jupyter Notebook or JupyterLab and execute the cells sequentially.

---

## 🔮 Future Improvements

* Add **Dropout** layers to reduce overfitting
* Perform **hyperparameter tuning**
* Compare ANN performance with **CNN models**
* Add confusion matrix and class‑wise metrics

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit a pull request for enhancements or bug fixes.

---

## 📄 License

This project is open‑source and intended for **educational and academic use**.

---

## ⭐ Acknowledgements

* Zalando Research for the Fashion‑MNIST dataset
* TensorFlow and Keras documentation

---

If you find this project useful, please ⭐ star the repository!
