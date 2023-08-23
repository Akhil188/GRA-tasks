import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data.csv")
labels = pd.read_csv("label.csv")

# Split data into input features (X) and labels (y)

X = data.drop("unix Timestamp", axis=1).values
y = labels.values

# Reshape input data into the desired format (55, 2500, 2)
X = X.reshape(55, 2500, 2)  # 55 instances, each with a shape of (2500, 2)

# Convert labels to one-hot encoding
num_classes = 3  # Number of unique labels (1, 2, 3)
y = tf.keras.utils.to_categorical(y - 1, num_classes=num_classes) 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(2500, 2)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax') 
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with random batching
batch_size = 32
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i:i + batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        model.train_on_batch(X_batch, y_batch)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save model predictions as "output.txt"
predictions = model.predict(X_test)
np.savetxt("output.txt", predictions)
