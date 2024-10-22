import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon']
train_emoticon_Y = train_emoticon_df['label']


valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
valid_emoticon_X = valid_emoticon_df['input_emoticon']
valid_emoticon_Y = valid_emoticon_df['label']


unique_emojis = set()
for i in train_emoticon_X:
    unique_emojis.update(list(i))
unique_emojis = list(unique_emojis)

# All the unique emojis collected from each string of emoticons. 
# Next part is creating emoji-to-index dictionary

emoji_to_index = {emoji: idx for idx, emoji in enumerate(unique_emojis)}

# One-hot encoding function for a single emoji
def one_hot(emoji):
    encoding = np.zeros(len(unique_emojis), dtype=int)
    if emoji in emoji_to_index:
        index = emoji_to_index[emoji]
        encoding[index] = 1
    return encoding

# We finally need one-hot encoding function for a sequence of 13 emojis
def one_hot_encode_and_concatenate(emoji_sequence):
    concatenated_vector = np.zeros(13 * len(unique_emojis), dtype=int) 
    # 13 one-hot vectors concatenated
    for idx, emoji in enumerate(emoji_sequence[:13]):
        one_hot_vector = one_hot(emoji)  # One-hot encode each emoji
        concatenated_vector[idx * len(unique_emojis):(idx + 1) * len(unique_emojis)] = one_hot_vector  # Concatenate
    return concatenated_vector
    
train_X = train_emoticon_X.apply(one_hot_encode_and_concatenate)

# Convert the result into a 2D numpy array for training (batch_size x 13 * num_unique_emojis)
train_X = np.stack(train_X)

valid_X = valid_emoticon_X.apply(one_hot_encode_and_concatenate)
valid_X = np.stack(valid_X)


percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Function to train Soft SVM and compute accuracy for each method
def evaluate_svm(train_feat_X_method, valid_feat_X_method, method_name):
    accuracies = []
    for percentage in percentages:
        n_samples = int(len(train_feat_X_method) * percentage)
        
        # Subset the training data
        x_train_subset = train_feat_X_method[:n_samples]
        y_train_subset = train_emoticon_Y[:n_samples]

        # Train the Soft SVM model
        svm_classifier = SVC(kernel='linear', random_state=42)  # Soft SVM with linear kernel
        svm_classifier.fit(x_train_subset, y_train_subset)

        # Predict on the validation set
        y_pred = svm_classifier.predict(valid_feat_X_method)

        # Calculate accuracy
        accuracy = accuracy_score(valid_emoticon_Y, y_pred)
        accuracies.append(accuracy)
    
    return accuracies


accuracies_concat = evaluate_svm(train_X, valid_X, "Concatenation")

plt.figure(figsize=(10, 6))
plt.plot([p * 100 for p in percentages], accuracies_concat, label="Concatenation", marker='o')

plt.xlabel("Percentage of Training Data")
plt.ylabel("Accuracy")
plt.title("Soft SVM Accuracy for Different Feature Representation Methods")
plt.legend()
plt.grid(True)
plt.show()
