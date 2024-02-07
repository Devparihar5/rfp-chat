# rfp-chat

#### Usage

1. **Installation:**


    Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. **Setup:**

    - Ensure you have a Hugging Face API token. You can obtain it from the [Hugging Face website](https://huggingface.co/join).
    - Set your Hugging Face API token as an environment variable:
    ```bash
    HUGGINGFACEHUB_API_TOKEN="your_hugging_face_api"
    ```

3. **Running the Code:**

    - Run the `main.py` script with the path to the PDF document as an argument:
    - to run with new rfp's run the following command
   ```bash
    python chat.py <pdf_file_path>
    ```

      or

    - to run with last processed rfp run the following command
   ```bash
    python chat.py
    ```

5. **Outputs:**

    - The script will generate a PDF report containing detailed responses to predefined questions extracted from the document.
    - Additionally, the script will produce a CSV file containing the extracted information along with their classifications.

#### Components:

1. **Document Loading and Processing:**
    - The `PyPDFLoader` is used to load PDF documents.
    - Text splitting is done using the `RecursiveCharacterTextSplitter`.

2. **Vector Database:**
    - `Chroma` is utilized to create and load vector databases for efficient document retrieval.

3. **Language Models and Embeddings:**
    - `HuggingFaceEmbeddings` are used for text embeddings.
    - Language models (LLMs) from Hugging Face Hub are employed for conversational retrieval chains.

4. **Model Initialization:**
    - The `initialize_llmchain` function initializes the conversational retrieval chain using the specified LLM model.

5. **Classification:**
    - A multiclass classifier is trained to classify responses into relevant categories using a pre-trained model.
    - The `EGovernanceClassifier` is utilized for classification.

6. **Report Generation:**
    - The `save_responses_as_pdf` function generates a PDF report with formatted responses.

#### Files:

- `main.py`: Main script to run the E-Governance document analysis.
- `classifier.py`: Contains the implementation of the EGovernanceClassifier for classification.
- `multiclass_predictor.py`: Implements functions for loading and predicting with the multiclass classification model.
- `questions.json`: JSON file containing predefined questions for document analysis.


#### Notes:

- Ensure you have the necessary permissions to access and utilize the Hugging Face API.
- Make sure to provide the correct path to the PDF document while running the script.
- Adjust configurations such as model paths, chunk sizes, and overlap as per your requirements.
