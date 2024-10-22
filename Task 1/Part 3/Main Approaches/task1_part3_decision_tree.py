import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

# Function to train Decision Tree and record validation accuracy
def train_decision_tree_on_subsets(X_train, y_train, X_val, y_val):
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracies = []
    
    for p in percentages:
        subset_size = int(len(X_train) * p)
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]

        # Train Decision Tree model
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train_subset, y_train_subset)
        y_val_pred = dt_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        accuracies.append(val_accuracy * 100)

    return percentages, accuracies

# Train and evaluate Decision Tree
percentages, accuracies = train_decision_tree_on_subsets(X_train, y_train, X_val, y_val)

# Plotting the results
plt.figure(figsize=(12, 8))
plt.plot([p * 100 for p in percentages], accuracies, marker='o', label="Decision Tree")

plt.title("Decision Tree Validation Accuracy vs Training Data Percentage")
plt.xlabel("Training Data Percentage")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.ylim(0, 100)  # Set y-axis limits for clarity
plt.legend(loc="lower right")
plt.show()
