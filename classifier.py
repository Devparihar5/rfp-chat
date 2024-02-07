import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

class EGovernanceClassifier:
    def __init__(self, model_path='./models/saved_model.h5', tokenizer_path='./models/tokenizer.joblib', scaler_path='./models/scaler.joblib', threshold=0.3):
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold

    def preprocess_text(self, input_text):
        input_sequence = self.tokenizer.texts_to_sequences([input_text])
        input_sequence = pad_sequences(input_sequence, maxlen=1)
        clean_text = self.scaler.transform(input_sequence)
        return clean_text

    def classify_input(self, input_text):
        clean_text = self.preprocess_text(input_text)

        # Predict probabilities using the trained model
        probabilities = self.model.predict(clean_text)
        # print(probabilities)
        # Convert probabilities to binary predictions based on the threshold
        # print((probabilities > self.threshold))
        binary_predictions = (probabilities > self.threshold).astype(int)

        # If you have a single prediction, you can access it like this
        single_prediction = binary_predictions[0]
        # print(single_prediction)
        # Return the result
        if single_prediction == 1:
            return "RFP is E-Governance"
        else:
            return "RFP is NON E-Governance"
        
        
# predictor = EGovernanceClassifier()
# response_answer = """The primary goal of the project, as per the document, is to provide contracted services which
# include development, operation, maintenance, and associated services for a specified site, as per the
# agreement. The agreement also includes requirements such as schedules, technical data, performance
# characteristics, and standards (Indian and International). The document also mentions "Service Down
# Time" which refers to the period when specified services or network segments are not available."""
# title = predictor.classify_input(response_answer)
# print(title)


