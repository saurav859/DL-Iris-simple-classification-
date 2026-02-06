🌸 Iris Flower Classification using Deep Learning

This project implements a deep learning-based multi-class classification model to predict the species of an Iris flower using four numerical features: sepal length, sepal width, petal length, and petal width. A feedforward neural network (MLP) is trained on the classic Iris dataset to classify samples into Setosa, Versicolor, and Virginica.

Problem Type: Supervised Learning, Multi-class Classification

Inputs: 4 continuous features

Outputs: 3-class categorical label

Model: Fully Connected Neural Network (MLP)

🗂️ Dataset

Source: Iris dataset (UCI / scikit-learn)

Samples: 150

Features: 4 numeric

Classes: 3 (balanced dataset)

⚙️ Tech Stack

Python

NumPy, Pandas

Scikit-learn (preprocessing, split, metrics)

TensorFlow/Keras (or PyTorch)

Matplotlib/Seaborn (visualization)

🏗️ Pipeline

Load dataset and perform basic EDA

Preprocess data (train-test split, feature scaling, one-hot encoding)

Build MLP model (Dense layers with ReLU + Softmax output)

Compile model (Adam optimizer, Categorical Crossentropy loss)

Train on training set and validate on test set

Evaluate using accuracy, confusion matrix, and classification report

Predict class labels for unseen samples

🧱 Model Architecture (Example)
Input (4 features)
 → Dense(16, ReLU)
 → Dense(8, ReLU)
 → Dense(3, Softmax)

📊 Results

The model achieves ~95–100% test accuracy (depending on split and hyperparameters). Most confusion occurs between Versicolor and Virginica, which is expected due to feature overlap. Overall, the network learns a robust non-linear decision boundary for all three classes.
