import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Load training data
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon']
train_emoticon_Y = train_emoticon_df['label']

# Load test data
test_emoticon_df = pd.read_csv("datasets/test/test_emoticon.csv")
test_emoticon_X = test_emoticon_df['input_emoticon']

# Collect unique emojis from training data
unique_emojis = set()
for i in train_emoticon_X:
    unique_emojis.update(list(i))
unique_emojis = list(unique_emojis)

# Create emoji-to-index dictionary
emoji_to_index = {emoji: idx for idx, emoji in enumerate(unique_emojis)}

# One-hot encoding function for a single emoji
def one_hot(emoji):
    encoding = np.zeros(len(unique_emojis), dtype=int)
    if emoji in emoji_to_index:
        index = emoji_to_index[emoji]
        encoding[index] = 1
    return encoding

# One-hot encoding for a sequence of 13 emojis
def one_hot_encode_and_concatenate(emoji_sequence):
    concatenated_vector = np.zeros(13 * len(unique_emojis), dtype=int) 
    for idx, emoji in enumerate(emoji_sequence[:13]):
        one_hot_vector = one_hot(emoji)
        concatenated_vector[idx * len(unique_emojis):(idx + 1) * len(unique_emojis)] = one_hot_vector
    return concatenated_vector

# Encode training and test data
train_X = np.stack(train_emoticon_X.apply(one_hot_encode_and_concatenate))
test_X = np.stack(test_emoticon_X.apply(one_hot_encode_and_concatenate))

# Train Soft SVM on the full training set
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(train_X, train_emoticon_Y)

# Predict on the test set
test_predictions = svm_classifier.predict(test_X)

output_df = pd.DataFrame({'predicted_label': test_predictions})
output_df.to_csv("test_predictions.csv", index=False)
