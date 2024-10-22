import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load datasets
train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']
valid_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
valid_feat_X = valid_feat['features']
valid_feat_Y = valid_feat['label']

test_feat = np.load("datasets/test/test_feature.npz", allow_pickle=True)
test_feat_X = test_feat['features']

percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Function to train SVM and compute accuracy
def evaluate_svm(train_feat_X_method, valid_feat_X_method):
    accuracies = []
    for percentage in percentages:
        n_samples = int(len(train_feat_X_method) * percentage)
        x_train_subset = train_feat_X_method[:n_samples]
        y_train_subset = train_feat_Y[:n_samples]
        
        # Train the SVM model
        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(x_train_subset, y_train_subset)

        # Predict on the validation set
        y_pred = svm_classifier.predict(valid_feat_X_method)

        # Calculate accuracy
        accuracy = accuracy_score(valid_feat_Y, y_pred)
        accuracies.append(accuracy * 100)
    
    return accuracies

# Concatenation of Embeddings
train_feat_X_concat = train_feat_X.reshape(train_feat_X.shape[0], -1)
valid_feat_X_concat = valid_feat_X.reshape(valid_feat_X.shape[0], -1)

# Evaluate SVM
accuracies_svm = evaluate_svm(train_feat_X_concat, valid_feat_X_concat)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot([p * 100 for p in percentages], accuracies_svm, label="SVM", marker='o')

plt.xlabel("Percentage of Training Data")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy for Different Percentage of Training Data")
plt.legend()
plt.grid(True)
plt.show()
