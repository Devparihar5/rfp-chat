from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph,PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from classifier import EGovernanceClassifier
from multiclass_predictor import load_multiclass_model, predict_multiclass_labels
import os
import json
import sys
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# Create an empty DataFrame or load existing CSV if available
columns = ['question', 'answer', 'source', 'source_page', 'labels']
result_df  = pd.DataFrame(columns=columns)

default_persist_directory = './chromadb/'

# Load PDF document and create doc splits
def load_doc(file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    print(f"Document Successfully loaded......")
    # print(doc_splits)
    return doc_splits

# Create vector database
def create_db(splits):
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=default_persist_directory
    )
    
    return vectordb

# Load vector database
def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        persist_directory=default_persist_directory,
        embedding_function=embedding)
    print(f"Vectordb Successfully loaded......")
    return vectordb

# Initialize the llm-model
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
    print(f"Initializing llmchain for {llm_model}......")
    print("Loading Hugging Face model...")
    llm = HuggingFaceHub(
        repo_id=llm_model,
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k}
    )
    print("Creating ConversationBufferMemory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    print("Creating vector retriever from the vector database...")
    retriever = vector_db.as_retriever()

    print("Creating ConversationalRetrievalChain from Hugging Face model and vector retriever...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
    )
    print("Successfully Initialized llmchain...")
    return qa_chain

# Initialize database
def initialize_database(file_path, chunk_size, chunk_overlap):
    print(f"Loading document from {file_path}...")
    doc_splits = load_doc(file_path, chunk_size, chunk_overlap)
    
    print("Creating vector database...")
    vector_db = create_db(doc_splits)
    print("Vector database created successfully...")
    return vector_db

def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")

    return formatted_chat_history

# Save response in .txt file
def save_responses_as_pdf(title, pdf_filename, responses,aadhaar_status):
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []
    elements.append(Paragraph(f'{title}', getSampleStyleSheet()['Heading1']))
    
    if aadhaar_status!=None:
        elements.append(Paragraph(f'{aadhaar_status}', getSampleStyleSheet()['Heading1']))

    for response in responses:
        message,response_answer, response_source, response_page = response
        # Add question, answer, source, and page number
        elements.append(Paragraph(f'Question: {message}', getSampleStyleSheet()['BodyText']))
        elements.append(Paragraph(f'Answer: {response_answer}', getSampleStyleSheet()['BodyText']))
        elements.append(Paragraph(f'Source: {response_source}', getSampleStyleSheet()['BodyText']))
        elements.append(Paragraph(f'Page Number: {response_page}', getSampleStyleSheet()['BodyText']))

        # Add a page break after each question
        elements.append(PageBreak())

    doc.build(elements)

# Get response from the model
def get_response(message,formatted_chat_history):
    # Generate response using QA chain
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()

    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    return response_answer,response_source1,response_source1_page

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) > 2:
        print("Usage: python chat.py [<pdf_file_path>]")
        sys.exit(1)

    document_path = None
    if len(sys.argv) == 2:
        pdf_file_path = sys.argv[1]
    

       
    # Read the questions list from the JSON file
    with open('questions.json', 'r') as json_file:
        loaded_questions = json.load(json_file)
    
    
    model_path = "./models/multiclass/multiclass.h5"
    tokenizer_path = "./models/multiclass/tokenizer_config.json"
    mlb_path = "./models/multiclass/mlb_classes.npy"

    loaded_model, loaded_tokenizer, mlb = load_multiclass_model(model_path, tokenizer_path, mlb_path)

    # Model Configuration
    chunk_size = 600
    chunk_overlap = 40

    # llm_model  = "mistralai/Mistral-7B-Instruct-v0.1"
    llm_model  = "mistralai/Mistral-7B-Instruct-v0.2"

    temperature = 0.7
    max_tokens = 1024
    top_k_samples = 3

    # train a new model 

    if document_path==None:
        vector_db = load_db()
    else:
        vector_db = initialize_database(document_path, chunk_size, chunk_overlap)
        

    qa_chain = initialize_llmchain(llm_model, temperature, max_tokens, top_k_samples, vector_db)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    # create the TXT filename by appending ".txt" to the PDF base name
    file_name = f"./output/response_document.pdf"
    

    responses = []
    history = []

    message = "Generate a detailed overview for the project mentioned in the document. Summarize the project's goals, objectives, and key features. Indicate whether the project is associated with capacity building. If there is any reference to a comprehensive capacity building toolkit issued by SUDA, include relevant details. Additionally, mention if the project overview is explicitly stated or inferred. Ensure the output is clear and organized."

    formatted_chat_history = format_chat_history(message, history)

    response_answer,response_source1,response_source1_page = get_response(message,formatted_chat_history)
    # responses.append((message,response_answer,response_source1, response_source1_page))
    # print(response_answer)
    
    predictor = EGovernanceClassifier()

    # Get the classification result
    title = predictor.classify_input(response_answer)
    if "aadhaar" in response_answer.lower():
        aadhaar_status = "Aadhaar-Related RFP"
        title = "RFP is E-Governance"
    else:
        aadhaar_status = "NoN-Aadhaar RFP."    
    if title=="RFP is NON E-Governance":
        aadhaar_status = None
  
    for message in loaded_questions:

        formatted_chat_history = format_chat_history(message, history)

        # get the response from the llm 
        response_answer,response_source1,response_source1_page = get_response(message,formatted_chat_history)
        input_text =response_answer
        
        predicted_labels = predict_multiclass_labels(input_text, loaded_model, loaded_tokenizer, mlb)
        if len(predicted_labels) > 0:
            result_row = {
            'question': message,
            'answer': response_answer,
            'source': response_source1,
            'source_page': response_source1_page,
            'labels': ', '.join(predicted_labels) if len(predicted_labels) > 0 else None
            }
            # Create a temporary DataFrame with the current result_row
            temp_df = pd.DataFrame([result_row])

            # Concatenate the temporary DataFrame to the result_df
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

        history = history + [(message, response_source1)]

        responses.append((message,response_answer,response_source1, response_source1_page))
        print(f"{message} sucessfully processed!!")

    # print(responses)
    save_responses_as_pdf(title,file_name, responses,aadhaar_status)
    result_df.to_csv('./output/results_with_classifications.csv', index=False)
    print(f"Output for all the details related to RFP as {file_name}!!!")
