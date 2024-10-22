import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load datasets
train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']
valid_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
valid_feat_X = valid_feat['features']
valid_feat_Y = valid_feat['label']

test_feat = np.load("datasets/test/test_feature.npz", allow_pickle=True)
test_feat_X = test_feat['features']

# Use all of the training data to train the model
train_feat_X_concat = train_feat_X.reshape(train_feat_X.shape[0], -1)

# Train Logistic Regression on the entire training dataset
LR_classifier = LogisticRegression(random_state=42)
LR_classifier.fit(train_feat_X_concat, train_feat_Y)

# Predict on the test set
test_feat_X_concat = test_feat_X.reshape(test_feat_X.shape[0], -1)
test_predictions = LR_classifier.predict(test_feat_X_concat)

# Save the test predictions to a CSV file
test_predictions_df = pd.DataFrame({
    'Predicted_Label': test_predictions
})

# Specify the path where you want to save the file
test_predictions_df.to_csv('logistic_regression_test_predictions.csv', index=False)

print("Predicted labels saved to logistic_regression_test_predictions.csv.")
