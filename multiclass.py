import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("./dataset/multclass_dataset.csv")

# Tokenize and pad text data
max_words = 1000
max_len = 20
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)


# One-hot encode labels
mlb = MultiLabelBinarizer()
# Flatten the list of lists before applying MultiLabelBinarizer
labels_flat = [label for sublist in df['label'].apply(lambda x: [x.split('_')]) for label in sublist]
labels_encoded = mlb.fit_transform(labels_flat)

print(mlb.classes_)

np.save('./models/multiclass/mlb_classes.npy', mlb.classes_)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_encoded, test_size=0.2, random_state=42)


model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(LSTM(64))
model.add(Dense(len(mlb.classes_), activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# history = model.fit(X_train, y_train, epochs=200, batch_size=2, validation_data=(X_test, y_test))
# Save the trained model

checkpoint = ModelCheckpoint(
    './models/multiclass/multiclass.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model with the ModelCheckpoint callback
history = model.fit(
    X_train, y_train,
    epochs=200, batch_size=2,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

# Plotting the accuracy and loss graphs
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save the plot as a PNG file
plt.savefig('./Model_metrics/multiclass/training_plot.png')


# Predict the labels for the test set
y_pred = model.predict(X_test)

# Binarize the predictions using a threshold (e.g., 0.5)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate the multilabel confusion matrix
conf_matrix = multilabel_confusion_matrix(y_test, y_pred_binary)

# Plot the confusion matrix
fig, axes = plt.subplots(len(mlb.classes_), 1, figsize=(10, 3*len(mlb.classes_)))

for i, (label, matrix) in enumerate(zip(mlb.classes_, conf_matrix)):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    disp.plot(ax=axes[i], cmap='viridis', values_format='d')
    axes[i].set_title(f'Confusion Matrix for Class: {label}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('True')

plt.tight_layout()
plt.savefig('./Model_metrics/multiclass/confusion_metrics.png')
