# House Prices Prediction using TensorFlow Decision Forests

import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Handle missing values
train_df = train_df.fillna(train_df.median(numeric_only=True))
train_df = train_df.fillna("Unknown")

test_df = test_df.fillna(test_df.median(numeric_only=True))
test_df = test_df.fillna("Unknown")

# Split training data
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

label = "SalePrice"

# Convert to TF datasets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label)
val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(val_df, label=label)

# Build model
model = tfdf.keras.RandomForestModel(
    task=tfdf.keras.Task.REGRESSION
)

# Train model
model.fit(train_ds)

# Evaluate model
model.evaluate(val_ds)

# Predict on test data
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)
predictions = model.predict(test_ds)

# Create submission file
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": predictions.flatten()
})

submission.to_csv("submission.csv", index=False)

print("Training complete. Submission file saved as submission.csv")
