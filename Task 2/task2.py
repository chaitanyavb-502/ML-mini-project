import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# Load Datasets
# Training Datasets
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()

train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']

train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
train_seq_X = train_seq_df['input_str'].tolist()
train_seq_Y = train_seq_df['label'].tolist()

# Validation Datasets
valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
valid_emoticon_X = valid_emoticon_df['input_emoticon'].tolist()
valid_emoticon_Y = valid_emoticon_df['label'].tolist()

valid_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
valid_feat_X = valid_feat['features']
valid_feat_Y = valid_feat['label']

valid_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv") 
valid_seq_X = valid_seq_df['input_str'].tolist() 
valid_seq_Y = valid_seq_df['label'].tolist() 

# Convert labels to integers
label_encoder = LabelEncoder()
train_emoticon_Y = label_encoder.fit_transform(train_emoticon_Y)
train_feat_Y = label_encoder.fit_transform(train_feat_Y)
train_seq_Y = label_encoder.fit_transform(train_seq_Y)

valid_emoticon_Y = label_encoder.transform(valid_emoticon_Y)
valid_feat_Y = label_encoder.transform(valid_feat_Y)
valid_seq_Y = label_encoder.transform(valid_seq_Y)

# Prepare the third dataset: digit encoding
max_length = 50
train_seq_X_encoded = [[int(digit) for digit in seq] for seq in train_seq_X]
train_seq_X_encoded = np.array(train_seq_X_encoded)

valid_seq_X_encoded = [[int(digit) for digit in seq] for seq in valid_seq_X]
valid_seq_X_encoded = np.array(valid_seq_X_encoded)

# Feature Transformation
# 1. One-Hot Encoding for Emoticons
encoder = OneHotEncoder(sparse_output=False, handle_unknown = 'ignore')
train_emoticon_X_encoded = encoder.fit_transform(np.array(train_emoticon_X).reshape(-1, 1))
valid_emoticon_X_encoded = encoder.transform(np.array(valid_emoticon_X).reshape(-1, 1))


# 2. Reshape train_feat_X to 2D if it has more than 2 dimensions
if train_feat_X.ndim > 2:
    train_feat_X_reshaped = train_feat_X.reshape(train_feat_X.shape[0], -1)  # Flatten the feature matrix
else:
    train_feat_X_reshaped = train_feat_X

if valid_feat_X.ndim > 2:
    valid_feat_X_reshaped = valid_feat_X.reshape(valid_feat_X.shape[0], -1)  # Flatten the feature matrix
else:
    valid_feat_X_reshaped = valid_feat_X

# 3. Apply PCA to reduce deep features to 7,000 features
pca = PCA(n_components=7000)  # Adjust number of components as needed
train_feat_X_reduced = pca.fit_transform(train_feat_X_reshaped)
valid_feat_X_reduced = pca.transform(valid_feat_X_reshaped)

# 4. Concatenate Deep Features (after PCA) with One-Hot Encoded Features
combined_train_features = np.concatenate((train_feat_X_reduced, train_emoticon_X_encoded), axis=1)
combined_valid_features = np.concatenate((valid_feat_X_reduced, valid_emoticon_X_encoded), axis=1)

# 5. Concatenate the Combined Features with Digit Encoded Features
final_train_features = np.concatenate((combined_train_features, train_seq_X_encoded), axis=1)
final_valid_features = np.concatenate((combined_valid_features, valid_seq_X_encoded), axis=1)

# Prepare labels
final_train_labels = train_seq_Y  # Labels for training
final_valid_labels = valid_seq_Y  # Labels for validation

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Soft Margin SVM": SVC(kernel='linear', C=1),
    "Decision Tree": DecisionTreeClassifier()
}

# Store results for plotting
percentages = [0.2, 0.4, 0.6, 0.8, 1.0]  # Training percentages
results = {model_name: [] for model_name in models.keys()}  # Initialize results

# Train on subsets of the training data
for p in percentages:
    subset_size = int(len(final_train_features) * p)
    X_train_subset = final_train_features[:subset_size]
    y_train_subset = final_train_labels[:subset_size]

    for model_name, model in models.items():
        print(f"Training {model_name} on {p * 100:.0f}% of the training data...")
        model.fit(X_train_subset, y_train_subset)
        
        # Evaluate on the validation set
        y_val_pred = model.predict(final_valid_features)  # Use the validation features
        val_accuracy = accuracy_score(final_valid_labels, y_val_pred)

        results[model_name].append(val_accuracy*100)
        print(f"{model_name} Validation Accuracy: {val_accuracy:.4f} on {p * 100:.0f}% training data")

# Plotting the results
plt.figure(figsize=(10, 6))
for model_name, accuracies in results.items():
    plt.plot([p * 100 for p in percentages], accuracies, marker='o', label=model_name)

plt.title("Validation Accuracy of Different Models vs Training Data Percentage")
plt.xlabel("Training Data Percentage")
plt.ylabel("Validation Accuracy")
plt.xticks([20, 40, 60, 80, 100])
plt.ylim(0, 100)  # Set y-axis limits for clarity
plt.grid()
plt.legend()
plt.show()