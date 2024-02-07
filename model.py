import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Plot training history and save as PNG
def plot_training_history(history, filename):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


file_path = "./dataset/projects_data.csv"
df = pd.read_csv(file_path)

# Identify and exclude non-numeric columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
X_numeric = df.drop(columns=non_numeric_cols) 

# Alternatively, you can explicitly select numeric columns
# X_numeric = df.select_dtypes(include=['number'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_numeric, df["IsEGovernance"], test_size=0.2, random_state=42)
# Tokenize and pad the input text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)  # Assuming X_train is a column with text data

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

joblib.dump(tokenizer, './models/tokenizer.joblib') 
joblib.dump(scaler, './models/scaler.joblib')

X_test = scaler.transform(X_test)

# Build the neural network model 
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train the model (same as before)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


plot_training_history(history, './Model_metrics/GovernanceClassifier/training_history.png')

# Evaluate the model (same as before)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("./models/saved_model.h5")


# Plot confusion matrix and save as PNG
def plot_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename)
    plt.show()
    plt.close()

# Make predictions on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Define classes (assuming binary classification)
classes = ['Not e-Governance', 'e-Governance']

plot_confusion_matrix(y_test, y_pred, classes, './Model_metrics/GovernanceClassifier/confusion_matrix.png')