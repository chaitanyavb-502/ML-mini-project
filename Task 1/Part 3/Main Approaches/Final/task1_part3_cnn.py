import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Load training datasets
train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
train_seq_X = train_seq_df['input_str'].tolist()
train_seq_Y = train_seq_df['label'].tolist()

# Preprocessing: Convert input strings to sequences of integers
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

# Train the CNN model
model = build_cnn_model(X_train.shape)
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)

# Load test dataset
test_seq_df = pd.read_csv("datasets/test/test_text_seq.csv")
test_seq_X = test_seq_df['input_str'].tolist()
test_seq_X = [[int(digit) for digit in seq] for seq in test_seq_X]
test_seq_X = np.array(test_seq_X)

# Predict on the test set
test_predictions = (model.predict(test_seq_X) > 0.5).astype("int32")
test_predictions = test_predictions.flatten()  # Flatten the predictions

# Convert the predicted labels back to original label encoding
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Save the predictions to a CSV file
output_df = pd.DataFrame({"input_str": test_seq_df['input_str'], "predicted_label": test_predictions_labels})
output_df.to_csv("test_predictions_cnn.csv", index=False)
