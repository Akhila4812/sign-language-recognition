import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.tensorflow

# Set random seed for consistent results
tf.random.set_seed(42)
np.random.seed(42)

# Load the dataset
train_df = pd.read_csv('C:/Users/akhil/sign_language/data/sign_mnist_train.csv')
test_df = pd.read_csv('C:/Users/akhil/sign_language/data/sign_mnist_test.csv')

# Filter for labels 0-8 (letters A-I)
train_df = train_df[train_df['label'].between(0, 8)]
test_df = test_df[test_df['label'].between(0, 8)]

# Separate images (pixels) and labels
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Normalize pixel values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape images to 28x28 pixels with 1 color channel (grayscale)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot format (9 classes: 0-8 for A-I)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=9)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=9)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(9, activation='softmax')  # 9 classes (A-I)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Start MLflow to track the training
with mlflow.start_run():
    # Train the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

    # Test the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Log results to MLflow
    mlflow.log_param('epochs', 5)
    mlflow.log_param('batch_size', 32)
    mlflow.log_metric('test_loss', test_loss)
    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.tensorflow.log_model(model, 'cnn_model')

    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

# Save the model
model.save('C:/Users/akhil/sign_language/cnn_model.h5')
