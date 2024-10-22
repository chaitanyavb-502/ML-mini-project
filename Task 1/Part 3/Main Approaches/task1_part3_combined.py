import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load datasets
train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
train_seq_X = train_seq_df['input_str'].tolist()
train_seq_Y = train_seq_df['label'].tolist()

# Preprocessing: Convert input strings to sequences of integers
max_length = 50
train_seq_X = [[int(digit) for digit in seq] for seq in train_seq_X]
train_seq_X = np.array(train_seq_X)

# Convert labels to integers
label_encoder = LabelEncoder()
train_seq_Y = label_encoder.fit_transform(train_seq_Y)
train_seq_Y = np.array(train_seq_Y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_seq_X, train_seq_Y, test_size=0.2, random_state=42)

# Define CNN model architecture with reduced parameters
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Embedding(input_dim=10, output_dim=16, input_length=input_shape[1]))  # Reduced embedding dimension
    model.add(Conv1D(16, kernel_size=3, activation='relu'))  # Reduced number of filters
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))  # Reduced dense layer size
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train models and record validation accuracy
def train_model_on_subsets(model_func, model_type, X_train, y_train, X_val, y_val):
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracies = []
    
    for p in percentages:
        subset_size = int(len(X_train) * p)
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]

        if model_type == "CNN":
            # Build and train the CNN model
            model = model_func(X_train_subset.shape)
            model.fit(X_train_subset, y_train_subset, epochs=10, batch_size=64, verbose=0)
            y_val_pred = (model.predict(X_val) > 0.5).astype("int32")
            val_accuracy = accuracy_score(y_val, y_val_pred)
        else:
            model = model_func()
            model.fit(X_train_subset, y_train_subset)
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)

        accuracies.append(val_accuracy * 100)

    return percentages, accuracies

# Define models
models = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "kNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": lambda: DecisionTreeClassifier(),
    "Soft Margin SVM": lambda: SVC(kernel='linear', C=1),
    "CNN": build_cnn_model
}

results = {}

for model_name, model_func in models.items():
    percentages, accuracies = train_model_on_subsets(model_func, model_name, X_train, y_train, X_val, y_val)
    results[model_name] = (percentages, accuracies)

# Plotting the results
plt.figure(figsize=(12, 8))
for model_name, (percentages, accuracies) in results.items():
    plt.plot([p * 100 for p in percentages], accuracies, marker='o', label=model_name)

plt.title("Validation Accuracy vs Training Data Percentage")
plt.xlabel("Training Data Percentage")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.ylim(0, 100)  # Set y-axis limits for clarity
plt.legend(loc="lower right")
plt.show()
