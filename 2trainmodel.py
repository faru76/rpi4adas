import os
import cv2
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for matplotlib (non-interactive)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0 = all messages, 3 = only error messages)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load and preprocess data
def load_data(data_path):
    images = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(data_path, filename))
            img = cv2.resize(img, (64, 64))  # Resize image to 64x64 pixels
            images.append(img)
            label = filename.split('_')[-2]  # Extract label from filename (assuming filename format)
            labels.append(label)
    return np.array(images), np.array(labels)

# Manually define the label to index mapping
label_to_index = {
    "forward": 0,
    "left": 1,
    "right": 2,
}

# Convert labels to one-hot encoding using the predefined mapping
def one_hot_encode(labels, label_to_index):
    one_hot_labels = np.array([label_to_index[label] for label in labels])
    one_hot_labels = tf.keras.utils.to_categorical(one_hot_labels, num_classes=len(label_to_index))
    return one_hot_labels

data_path = "training_data"
images, labels = load_data(data_path)
labels = one_hot_encode(labels, label_to_index)

# Print the label to index mapping
print("Label to Index Mapping:")
for label, index in label_to_index.items():
    print(f"{label}: {index}")

# Normalize images to [0, 1] range
images = images / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define a Convolutional Neural Network (CNN) model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_to_index), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Create callbacks for early stopping and model checkpointing
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore weights from the epoch with the best value of the monitored quantity
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',   # Filepath to save the best model
    monitor='val_loss',    # Monitor validation loss
    save_best_only=True,   # Save only the best model instead of every epoch
    mode='min'             # Minimize validation loss
)

# Train the model
history = model.fit(
    X_train, y_train,               # Training data
    epochs=50,                      # Number of epochs to train the model
    validation_data=(X_test, y_test),  # Validation data
    callbacks=[early_stopping, checkpoint]  # Callbacks for early stopping and model checkpointing
)

# Save the final trained model
model.save("final_model.keras")

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Print parameters of the final epoch where training stopped
final_epoch = len(history.history['loss']) - 1
print(f"\nFinal epoch (epoch {final_epoch+1}):")
print(f" - loss: {history.history['loss'][final_epoch]:.4f}")
print(f" - accuracy: {history.history['accuracy'][final_epoch]:.4f}")
print(f" - val_loss: {history.history['val_loss'][final_epoch]:.4f}")
print(f" - val_accuracy: {history.history['val_accuracy'][final_epoch]:.4f}")

# Generate confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_to_index.keys(), yticklabels=label_to_index.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Print classification report
class_report = classification_report(y_true_classes, y_pred_classes, target_names=label_to_index.keys())
print("\nClassification Report:\n", class_report)

# Plot training history: Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.savefig('accuracy_history.png')

# Plot training history: Loss
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.savefig('loss_history.png')
