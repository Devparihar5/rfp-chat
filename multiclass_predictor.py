import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def load_multiclass_model(model_path, tokenizer_path, mlb_path):
    df = pd.read_csv("./dataset/multclass_dataset.csv")

    max_words = 1000
    loaded_tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    loaded_tokenizer.fit_on_texts(df['text'])

    # Load the saved model
    loaded_model = load_model(model_path)

    # Load the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.classes_ = np.load(mlb_path, allow_pickle=True)

    return loaded_model, loaded_tokenizer, mlb

def predict_multiclass_labels(input_text, model, tokenizer, mlb):
    max_len = 20
    # Convert input text to lowercase
    input_text_lower = input_text.lower()

    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([input_text_lower])
    padded_input = pad_sequences(input_sequence, maxlen=max_len)

    # Predict labels
    predicted_labels = mlb.classes_[np.where(model.predict(padded_input) > 0.5)[1]]

    return predicted_labels

# Example usage:
# loaded_model, loaded_tokenizer, mlb = load_multiclass_model("./models/multiclass/multiclass.h5", "./models/multiclass/tokenizer_config.json", "./models/multiclass/mlb_classes.npy")
# input_text = "Your input text here."
# predicted_labels = predict_multiclass_labels(input_text, loaded_model, loaded_tokenizer, mlb)
# print("Predicted Labels:", predicted_labels)
