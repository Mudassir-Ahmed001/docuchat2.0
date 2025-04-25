import streamlit as st
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
import shutil
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import zipfile
import base64
import requests
import os
import torch
from streamlit_option_menu import option_menu
import os
from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import nltk
nltk.download('punkt')
import json
#os.environ["PYTHON_SQLITE_LIBRARY"] = "pysqlite3"
import sys




#GROQ_API_KEY = "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"
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY




st.set_page_config(layout="wide")

# Initialize session state with default values
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.chat_history = []
    st.session_state.documents_processed = False
    st.session_state.processing_complete = False
    st.session_state.files_uploaded = False
    st.session_state.vectorstore_path = None
    st.session_state.first_query = True
    st.session_state.flash_cards = []
    st.session_state.pipeline = None
    st.session_state.mcqs = None
    st.session_state.user_answers = {}
    st.session_state.show_results = False
    st.session_state.correct_answers = {}



# Initialize user_name if not already set
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# Firebase initialization
@st.cache_resource
def initialize_firebase():
    if not firebase_admin._apps:
        firebase_creds_str = st.secrets["FIREBASE_CREDENTIALS"]
        firebase_creds = json.loads(firebase_creds_str)
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'document-chatbot-generat-bf086.appspot.com'
        })
    return firestore.client(), storage.bucket()

try:
    db, bucket = initialize_firebase()
except Exception as e:
    st.error(f"Failed to initialize Firebase: {e}")

# Create necessary directories
local_directory = "data"
output_directory = "output"
os.makedirs(local_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)


def create_download_link(file_path, link_text):
    """Create a download link for a file"""
    with open(file_path, 'rb') as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    filename = os.path.basename(file_path)
    mime_type = 'application/zip' if file_path.endswith('.zip') else 'application/octet-stream'
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def clean_directory(directory):
    """Safely clean a directory by removing all files and subdirectories"""
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                st.warning(f"Failed to remove {item_path}: {e}")

def load_document(file_path):
    """Load document based on file extension with improved PDF handling"""
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.pdf':
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            return loader.load()

        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
            return loader.load()

        #elif file_extension == '.csv':
        #    df = pd.read_csv(file_path)
        #    text_content = []
        #    for index, row in df.iterrows():
        #        row_text = f"Row {index + 1}:\n"
        #        for column in df.columns:
        #            row_text += f"{column}: {row[column]}\n"
        #        text_content.append({"page_content": row_text, "metadata": {"source": file_path}})
        #    return text_content

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

def save_vectorstore():
    """Save the vectorstore to disk"""
    if st.session_state.vectorstore:
        vectorstore_path = os.path.join(output_directory, "vectorstore")
        os.makedirs(vectorstore_path, exist_ok=True)
        st.session_state.vectorstore.save_local(vectorstore_path)
        st.session_state.vectorstore_path = vectorstore_path
        return vectorstore_path
    return None

def create_standalone_chatbot():
    """Create a standalone chatbot script"""
    standalone_code = '''
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Configure your Groq API key
GROQ_API_KEY = "Your-own-API-KEY-here"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embedding = HuggingFaceEmbeddings()


vectorstore_path = "vectorstore"
vectorstore = FAISS.load_local(
    vectorstore_path,
    embedding,
    allow_dangerous_deserialization=True
)


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)


def user_query(query):
    """Process a user query and return the answer"""
    response = qa_chain.invoke({"query": query})
    return response["result"]


def main():
    print("Document Chatbot initialized. Type 'quit' to exit.")
    while True:
        query = input("\nASK YOUR QUESTION: ")
        if query.lower() == 'quit':
            break

        try:
            answer = user_query(query)
            print("\nANSWER:")
            print(answer)
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    main()

'''

    script_path = os.path.join(output_directory, "chatbot.py")
    with open(script_path, "w") as f:
        f.write(standalone_code)
    return script_path

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = '''
langchain-community==0.2.15
langchain-chroma==0.1.3
langchain-text-splitters==0.2.2
langchain-huggingface==0.0.3
langchain-groq==0.1.9
unstructured==0.15.0
unstructured[pdf]==0.15.0
faiss-cpu
nltk==3.8.1

'''
    requirements_path = os.path.join(output_directory, "requirements.txt")
    with open(requirements_path, "w") as f:
        f.write(requirements.strip())
    return requirements_path

def create_downloadable_package():
    """Create a downloadable zip package with all necessary files"""
    try:
        # Save vectorstore
        vectorstore_path = save_vectorstore()

        # Create standalone chatbot script
        script_path = create_standalone_chatbot()

        # Create requirements file
        requirements_path = create_requirements_file()

        # Create README
        readme_content = '''
# Standalone Document Chatbot

## Setup Instructions
1. Extract all files to a directory
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. add your own groq API KEY in the code.

'''
        readme_path = os.path.join(output_directory, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content.strip())

        # Create zip file
        zip_path = os.path.join(output_directory, "chatbot_package.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(script_path, os.path.basename(script_path))
            zipf.write(requirements_path, os.path.basename(requirements_path))
            zipf.write(readme_path, os.path.basename(readme_path))

            # Add vectorstore files
            for root, _, files in os.walk(vectorstore_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join("vectorstore", os.path.relpath(file_path, vectorstore_path))
                    zipf.write(file_path, arcname)

        return zip_path
    except Exception as e:
        st.error(f"Error creating downloadable package: {e}")
        return None

def process_documents():
    with st.spinner("Processing documents..."):
        try:
            documents = []

            # Process each file in the directory
            for filename in os.listdir(local_directory):
                file_path = os.path.join(local_directory, filename)
                if os.path.isfile(file_path):  # Only process files
                    doc_content = load_document(file_path)
                    if doc_content:
                        documents.extend(doc_content)

            if not documents:
                st.error("No content could be extracted from the documents.")
                return False

            # Split documents
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            text_chunks = text_splitter.split_documents(documents)

            if not text_chunks:
                st.error("No text chunks were created from the documents.")
                return False

            # Create embeddings
            embedding = HuggingFaceEmbeddings()

            # Create vectorstore
            vectorstore = FAISS.from_documents(text_chunks, embedding)
            st.session_state.vectorstore = vectorstore

            # Initialize LLM
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )

            st.session_state.qa_chain = qa_chain
            st.session_state.documents_processed = True
            st.session_state.processing_complete = True

            return True

        except Exception as e:
            st.error(f"Error during document processing: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return False

def upload_to_firebase(file_path, file_name, user_name):
    """Upload file to Firebase storage in user-specific folders"""
    try:
        # Ensure bucket is initialized
        if not bucket:
            st.error("bucket is not initialized.")
            return None
        
        # Create a blob for the file
        blob = bucket.blob(f"{user_name}/uploaded_files/{file_name}")
        
        # Upload the file
        blob.upload_from_filename(file_path)
        
        # Generate public URL
        file_url = blob.public_url
        #st.info(f"Uploaded file URL: {file_url}")
        return file_url
    except Exception as e:
        st.error(f"Error uploading file to Firebase: {e}")
        return None


def generate_flash_cards_from_documents(topic, count):
    """Generate flash cards from documents with clear question/answer separation"""
    if not st.session_state.vectorstore:
        st.error("No documents processed yet.")
        return []

    prompt = f"""Generate {count} flash cards about '{topic}' based on the uploaded documents.
    For each card, clearly format as:
    Q: [Question]
    A: [Answer]

    Make each question clear and concise, and each answer informative but brief."""

    try:
        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        response = qa_chain.invoke({"query": prompt})
        cards_text = response["result"]

        # Parse the response into flash cards
        flash_cards = []
        current_card = {}

        for line in cards_text.split('\n'):
            line = line.strip()
            if line.startswith('Q:'):
                if current_card.get('question'):  # Save previous card
                    flash_cards.append(current_card)
                    current_card = {}
                current_card['question'] = line[2:].strip()
            elif line.startswith('A:') and current_card.get('question'):
                current_card['answer'] = line[2:].strip()
                flash_cards.append(current_card)
                current_card = {}

        # Add any remaining card
        if current_card.get('question'):
            flash_cards.append(current_card)

        return flash_cards
    except Exception as e:
        st.error(f"Error generating flash cards: {str(e)}")
        return []

def generate_flash_card_html(question, answer):
    return f"""
    <div class="card">
        <div class="card-inner">
            <div class="card-front">
                <div class="card-content">
                    <p style="font-weight: bold; font-size: 16px;">{question}</p>
                </div>
            </div>
            <div class="card-back">
                <div class="card-content">
                    <p style="font-size: 16px;">{answer}</p>
                </div>
            </div>
        </div>
    </div>
    """





def add_css_for_flash_cards():
    st.markdown("""
    <style>
        .flash-card-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            padding: 15px;
        }

        .card {
            background-color: transparent;
            width: 300px;
            height: 200px;
            perspective: 1000px;
            margin: 10px;
        }

        .card-inner {
            position: relative;
            width: 100%;
            height: 100%;
            text-align: center;
            transition: transform 0.8s;
            transform-style: preserve-3d;
            cursor: pointer;
        }

        .card:hover .card-inner {
            transform: rotateY(180deg);
        }

        .card-front, .card-back {
            position: absolute;
            width: 100%;
            height: 100%;
            -webkit-backface-visibility: hidden;
            backface-visibility: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            padding: 15px;
            box-sizing: border-box;
        }

        .card-front {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #2c3e50;
        }

        .card-back {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: rotateY(180deg);
        }

        .card-content {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            line-height: 1.4;
        }
    </style>
    """, unsafe_allow_html=True)









def generate_mcqs(topic=None, num_questions=5, difficulty="Medium"):
    """Generate MCQs using the LLM."""
    if not st.session_state.vectorstore or not st.session_state.qa_chain:
        st.error("Please upload and process documents first.")
        return None

    # Prompt the LLM to generate MCQs with difficulty
    query = (
        f"Generate {num_questions} multiple-choice questions with 4 options each based on the topic '{topic}'. "
        f"Make the questions '{difficulty}' in difficulty. "
        "Provide the correct answer for each question in the format 'Correct Answer for Q<number>: <single-character>' "
        "(e.g., 'Correct Answer for Q1: A'). Ensure the answer is only 'A', 'B', 'C', or 'D'. "
        "Do not include any other characters or text in the correct answer."
    )

    try:
        response = st.session_state.qa_chain.invoke({"query": query})
        raw_mcqs = response.get("result", "")
        return parse_mcqs(raw_mcqs, num_questions)
    except Exception as e:
        st.error(f"Error generating MCQs: {e}")
        return None


# Parse MCQs
def parse_mcqs(raw_mcqs, num_questions):
    """Parse raw MCQs into structured format with full-text correct answers."""
    mcqs = []
    correct_answers = {}

    # Split response into question blocks and answer section
    parts = raw_mcqs.strip().split("\n\n")
    answer_section = [part for part in parts if "Correct Answer for Q" in part]

    # Parse MCQs from the main part of the response
    count = 0
    for block in parts:
        if count >= num_questions:  # Stop parsing if we've reached the required number of MCQs
            break
        lines = block.strip().split("\n")
        if len(lines) >= 5:  # Ensure enough lines for a question and 4 options
            question = lines[0].strip()
            options = [line.strip() for line in lines[1:5]]
            options.append("None")  # Add "None" option by default
            mcqs.append({"question": question, "options": options})
            count += 1

    # Extract correct answers and map them to full option strings
    if answer_section:
        answer_lines = answer_section[-1].strip().split("\n")  # Take the last block for correct answers
        for line in answer_lines:
            if line.startswith("Correct Answer for Q"):
                try:
                    question_num, answer_letter = line.split(":")
                    question_num = int(question_num.split("Q")[1].strip()) - 1
                    # Map letter to full option text
                    correct_answers[question_num] = mcqs[question_num]["options"][
                        ord(answer_letter.strip()) - ord('A')
                    ]
                except (ValueError, IndexError):
                    continue  # Skip lines that don't match the expected format

    # Save parsed data to session state
    st.session_state.correct_answers = correct_answers
    return mcqs

# Evaluate MCQs
def evaluate_mcqs(mcqs, user_answers):
    """Evaluate the user's MCQ answers."""
    score = 0
    total = len(mcqs)
    wrong_questions = []

    for i, mcq in enumerate(mcqs):
        correct_answer = st.session_state.correct_answers.get(i, "None")
        user_answer = user_answers.get(i, "None")

        if user_answer.strip() == correct_answer.strip():  # Compare full-text answers
            score += 1
        else:
            wrong_questions.append({
                "question": mcq["question"],
                "user_answer": user_answer,  # Show raw user answer
                "correct_answer": correct_answer,
                "options": mcq["options"]
            })

    percentage = (score / total) * 100 if total > 0 else 0
    return score, total, percentage, wrong_questions

# Suggestions for improvement
def generate_suggestions_from_wrong_answers(wrong_questions):
    """Provide improvement suggestions based on wrong answers."""
    if not wrong_questions:
        return "Great job! You answered all questions correctly. Keep up the good work!"

    questions_text = "\n".join([q["question"] for q in wrong_questions])
    prompt = (
        f"Based on the following incorrectly answered questions:\n\n{questions_text}\n\n"
        f"Provide a 100-word suggestion for the user to improve their knowledge in these areas."
    )

    try:
        response = st.session_state.qa_chain.invoke({"query": prompt})
        suggestions = response.get("result", "No suggestions generated. Please try again.")
        return suggestions.strip()
    except Exception as e:
        return "There was an error generating suggestions. Please try again."

# Main UI

if not st.session_state.user_name:
 st.session_state.user_name = st.text_input("Please enter your name:")
 if st.button("Submit"):
     if not st.session_state.user_name:
         st.warning("Please enter your name")
     else:
         clean_directory(local_directory)  # Clear directory upon login
         st.session_state.initialized = True
else:
 #selected_page = st.radio("Navigate to:", ('Home Page', 'Chat Bot', 'Flash Cards', 'Exam mode', 'AI Generated Video'), horizontal=True)
 selected_page = option_menu(
    menu_title=None,  # No title
    options=["Home", "Chat Bot", "Flash Cards", "Exam mode", "AI Generated Video"],  # Menu options
    icons=["house-fill", "chat-dots-fill", "card-list", "clipboard-data", "play-circle"],  # Updated icons
    menu_icon="app-indicator",  # Main menu icon
    default_index=0,  # Default selected option
    orientation="horizontal",  # Horizontal menu
    styles={
        "container": {
            "padding": "0",  # No extra padding
            "width": "100%",  # Span the full width of the viewport
            "background-color": "#212121",  # Background color for the menu
        },
        "icon": {"color": "white", "font-size": "20px"},  # Styling for icons
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "color": "white",
            "--hover-color": "#333333",  # Subtle hover effect
        },
        "nav-link-selected": {
            "background-color": "#ff4b4b",  # Highlight for selected item
            "color": "white",
            "font-weight": "bold",
        },
    },
 )



 clean_directory(local_directory)

 # Shared Sidebar for Chat Bot, Flash Cards, AI Generated Video
 if selected_page in ['Chat Bot', 'Flash Cards', 'AI Generated Video', 'Exam mode']:
     st.sidebar.title(f"Welcome, {st.session_state.user_name}!")
     st.sidebar.title("File Upload")

     uploaded_files = st.sidebar.file_uploader(
         "Choose files",
         type=['pdf', 'docx'],
         accept_multiple_files=True
     )

     if uploaded_files and not st.session_state.documents_processed:
        try:
            clean_directory(local_directory)
            for file in uploaded_files:
                file_path = os.path.join(local_directory, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Call the upload_to_firebase function
                file_url = upload_to_firebase(file_path, file.name, st.session_state.user_name)
                
                # Provide feedback to the user
                if file_url:
                    st.sidebar.success(f"Uploaded {file.name} successfully!")
                else:
                    st.sidebar.error(f"Failed to upload {file.name}")

            st.sidebar.success("All files uploaded and processed locally!")
            st.session_state.files_uploaded = True
        except Exception as e:
            st.sidebar.error(f"Error uploading files: {e}")


     if st.sidebar.button("Process Documents", disabled=not st.session_state.files_uploaded):
         if process_documents():
             st.session_state.documents_processed = True
             st.success("Documents processed successfully!")
      # Debug information
     if selected_page != 'Home Page' and st.sidebar.checkbox("Show Debug Info"):
          st.sidebar.write("### Debug Information")
          st.sidebar.write("Files uploaded:", st.session_state.files_uploaded)
          st.sidebar.write("Documents processed:", st.session_state.documents_processed)
          st.sidebar.write("Processing complete:", st.session_state.processing_complete)
          st.sidebar.write("Vectorstore initialized:", st.session_state.vectorstore is not None)
          st.sidebar.write("QA Chain initialized:", st.session_state.qa_chain is not None)
          if os.path.exists(local_directory):
              st.sidebar.write("Files in local directory:", os.listdir(local_directory))


 if selected_page == 'Home':
      st.title("Welcome to Exceptio")
      st.subheader("Your AI-Powered Document Companion")

      # Overview Section
      st.markdown("""
      ---
      ### What is Exceptio?
      Exceptio is an innovative AI-powered platform designed to help you interact with your documents intelligently.
      With features like document-based Q&A, flashcard generation, exam preparation, and AI-driven insights,
      Exceptio transforms static documents into dynamic, interactive experiences.
      """)

      # Features Section
      st.markdown("""
      ---
      ### Key Features
      **1. Document Chatbot:**
      Ask questions directly from your uploaded documents and receive concise, accurate answers. Ideal for quick lookups or deep dives into document content.

      **2. Flashcards Generator:**
      Automatically generate flashcards to aid in studying and revision, perfect for students and professionals preparing for exams or presentations.

      **3. Exam Mode:**
      Create multiple-choice quizzes (MCQs) based on your document content with varying levels of difficulty to test your knowledge.

      **4. AI-Generated Videos (Coming Soon):**
      Get AI-powered video summaries based on your documents for visual learners.

      **5. Secure File Management:**
      Upload, process, and store your files securely using Firebase technology.
      """)

      # How to Use Section
      st.markdown("""
      ---
      ### How to Use Exceptio
      1. **Log In:** Enter your name to begin using the application.
      2. **Upload Documents:**
        - Use the sidebar to upload PDF or Word documents.
        - Ensure your documents are not password-protected for processing.
      3. **Process Documents:** Click "Process Documents" in the sidebar to analyze and index the content for interaction.
      4. **Explore Features:** Navigate through the top menu to:
        - **Chat Bot:** Ask questions about your document.
        - **Flash Cards:** Generate and review flashcards.
        - **Exam Mode:** Test your understanding with quizzes.
        - **AI-Generated Video:** Stay tuned for upcoming features!
      """)

      # Tips Section
      st.markdown("""
      ---
      ### Tips for Best Results
      - **Document Quality:** Upload clean, well-structured documents to improve AI accuracy.
      - **Document Size:** Split large documents into smaller sections for better processing and faster results.
      - **Questions:** Be specific when asking questions in the chatbot for more accurate answers.
      """)

      # Support Section
      st.markdown("""
      ---
      ### Need Help?
      If you encounter any issues or have questions, feel free to contact us at:
      **Support Email:** ai.exceptio@gmail.com
      **Documentation:** [Visit Documentation](#)
      """)

      # Closing Message
      st.markdown("""
      ---
      Explore Exceptio today and revolutionize the way you interact with your documents!
      """)

 elif selected_page == 'Chat Bot':

    chat_col, download_col = st.columns([2, 1])

    with chat_col:
        st.title("Chat Bot")

        # Add custom CSS for styling
        st.markdown(
            """
            <style>
                /* Sticky Top Menu Fix */
                .block-container {
                    padding-top: 70px; /* Adjust for menu height */
                }

                /* Chat Container Styles */
                .chat-container {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    max-width: 1000px;
                    margin: auto;
                    padding: 10px;
                }
                .chat-message {
                    display: flex;
                    align-items: flex-start;
                    gap: 10px;
                }
                .chat-message.user {
                    flex-direction: row-reverse;
                }
                .chat-bubble {
                    padding: 15px 20px;
                    border-radius: 20px;
                    max-width: 80%;
                    word-wrap: break-word;
                    font-size: 16px;
                    line-height: 1.5;
                }
                .chat-bubble.user {
                    background-color: #0078d4;
                    color: white;
                }
                .chat-bubble.assistant {
                    background-color: #e8e8e8;
                    color: #333;
                }
                .chat-avatar {
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    background-color: #444;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    color: white;
                }
                .chat-input-container {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                    padding-top: 20px;
                }
                textarea {
                    flex: 1;
                    padding: 10px;
                    font-size: 16px;
                    border: 1px solid #ccc;
                    border-radius: 10px;
                    resize: none;
                }
                .send-button {
                    background-color: #0078d4;
                    color: white;
                    padding: 10px 20px;
                    font-size: 16px;
                    border: none;
                    border-radius: 10px;
                    cursor: pointer;
                }
                .send-button:hover {
                    background-color: #005fa3;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


        # Main Chat Interface Container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        if st.session_state.documents_processed:
            # Display chat history
            for message in st.session_state.chat_history:
                if message.get("type") == "user":
                    st.markdown(
                        f"""
                        <div class="chat-message user">
                            <div class="chat-avatar" style="background-color: #0078d4;">👤</div>
                            <div class="chat-bubble user">{message.get("content")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-message assistant">
                            <div class="chat-avatar" style="background-color: #4caf50;">🤖</div>
                            <div class="chat-bubble assistant">{message.get("content")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            # End of chat history

            # Chat input section
            st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
            query = st.text_area(
                "",
                placeholder="Type your message here...",
                height=70
            )
            if st.button("Send", key="send_button", help="Send your message"):
                if query.strip():
                    st.session_state.chat_history.append({"type": "user", "content": query})
                    with st.spinner("Processing..."):
                        try:
                            response = st.session_state.qa_chain.invoke({"query": query})
                            answer = response.get("result", "No response available.")
                            st.session_state.chat_history.append(
                                {"type": "assistant", "content": answer}
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing query: {e}")
                else:
                    st.warning("Please enter a message before sending.")
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.warning("Please upload and process documents from sidebar.")

        st.markdown('</div>', unsafe_allow_html=True)



    with download_col:
        st.header("Download Chatbot")

        if st.session_state.documents_processed:
            if st.button("Generate Download Package"):
                with st.spinner("Creating downloadable package..."):
                    try:
                        zip_path = create_downloadable_package()
                        if zip_path and os.path.exists(zip_path):
                            # Create download link
                            download_link = create_download_link(
                                zip_path,
                                "Download Chatbot Package"
                            )
                            st.markdown(download_link, unsafe_allow_html=True)

                            st.success("""
                            Package created successfully! The download package includes:
                            - Standalone chatbot script
                            - Processed document embeddings
                            - Requirements file
                            - Setup instructions
                            """)

                            # Upload package to Firebase
                            package_url = upload_to_firebase(
                                zip_path,
                                f"chatbot_package_{st.session_state.user_name}.zip",
                                st.session_state.user_name
                            )
                            #if package_url:
                                #st.info(f"Download package uploaded to Firebase. Public URL: {package_url}")
                        else:
                            st.error("Failed to create download package.")
                    except Exception as e:
                        st.error(f"Error creating download package: {e}")

 elif selected_page == 'Flash Cards':
    st.title("Flash Cards")
    add_css_for_flash_cards()

    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Enter a topic for generating flash cards:")
    with col2:
        card_count = st.slider("Number of flash cards to generate:", 1, 20, 10)

    if st.button("Generate Flash Cards", disabled=not topic.strip()):
        with st.spinner("Generating flash cards..."):
            st.session_state.flash_cards = generate_flash_cards_from_documents(topic, card_count)
            if st.session_state.flash_cards:
                st.success("Flash cards generated successfully!")
            else:
                st.error("Failed to generate flash cards. Please ensure documents are uploaded and processed.")

    if st.session_state.flash_cards:
        cols = st.columns(3)
        for idx, card in enumerate(st.session_state.flash_cards):
            if 'question' in card and 'answer' in card:  # Verify card has both question and answer
                with cols[idx % 3]:
                    st.markdown(
                        generate_flash_card_html(
                            card['question'],
                            card['answer']
                        ),
                        unsafe_allow_html=True
                    )





 elif selected_page == 'Exam mode':
    st.title("Exam Mode")

    # Input for topic and number of questions
    topic_col, slider_col = st.columns([2, 1])
    with topic_col:
        topic = st.text_input("Enter a topic for generating MCQs (optional):")
    with slider_col:
        num_questions = st.slider("Number of questions:", 5, 15, 10)

    # Add difficulty level selection using a selectbox
    st.markdown("#### 🎯 Select Difficulty Level:")
    difficulty = st.selectbox(
        "Choose the difficulty of the MCQs:",
        ["Easy", "Medium", "Advance"],
        index=1,  # Default is Medium
        help="Easy: Basic questions, Medium: Moderate questions, Advance: Advanced questions.",
    )

    st.session_state.num_questions = num_questions

    if st.button("Generate MCQs"):
        with st.spinner("Generating MCQs..."):
            mcqs = generate_mcqs(topic, num_questions, difficulty)  # Pass difficulty to the MCQ generator
            if mcqs:
                st.session_state.mcqs = mcqs  # Save generated MCQs in session state
                st.session_state.user_answers = {i: "None" for i in range(len(mcqs))}  # Initialize answers
                st.session_state.show_results = False

    # Ensure MCQs persist in session state
    if st.session_state.get("mcqs"):
        left_col, right_col = st.columns([2, 1])

        # Display MCQs
        with left_col:
            st.header("MCQs")
            for i, mcq in enumerate(st.session_state.mcqs):
                st.write(f"**Q{i+1}: {mcq['question']}**")
                st.session_state.user_answers[i] = st.radio(
                    f"Select your answer for Q{i+1}:",
                    options=mcq["options"],
                    key=f"mcq_{i}",
                    index=mcq["options"].index(st.session_state.user_answers.get(i, "None"))
                )

            if st.button("Submit Answers"):
                unanswered = [i for i, ans in st.session_state.user_answers.items() if ans == "None"]
                if unanswered:
                    st.error(f"Please answer all questions before submitting. Unanswered: {unanswered}")
                else:
                    st.session_state.show_results = True

        # Evaluation Section
        with right_col:
            st.header("Evaluation")
            if st.session_state.show_results:
                # Display evaluation results
                score, total, percentage, wrong_questions = evaluate_mcqs(
                    st.session_state.mcqs, st.session_state.user_answers
                )
                suggestions = generate_suggestions_from_wrong_answers(wrong_questions)

                # Display basic results
                st.write(f"**Score:** {score}/{total}")
                st.markdown("---")
                st.write(f"**Percentage:** {percentage:.2f}%")
                st.markdown("---")
                st.write("**Correct Answers:**")
                for i in range(total):
                    correct_answer = st.session_state.correct_answers.get(i, "None")
                    st.write(f"Q{i+1}: {correct_answer}")
                st.markdown("---")

                # Display incorrect answers in detail
                if wrong_questions:
                    st.markdown("---")
                    st.subheader("Review Incorrect Answers")
                    st.write("Here are the questions you answered incorrectly:")
                    for idx, q in enumerate(wrong_questions, start=1):
                        st.markdown(f"**Q{idx}: {q['question']}**")
                        st.markdown(f"- **Your Answer:** {q['user_answer']}")
                        st.markdown(f"- **Correct Answer:** {q['correct_answer']}")
                        #st.markdown(f"- **Options:** {', '.join(q['options'])}")

                else:
                    st.success("Great job! You answered all questions correctly!")


                # Display AI suggestions
                st.markdown("---")
                st.markdown("---")
                st.header("AI Suggestions:")
                st.write(suggestions)




 elif selected_page == 'AI Generated Video':
     st.title("Launching soon! stay tuned.")




# Cleanup on session end
def cleanup():
    """Clean up temporary files when the session ends"""
    try:
        clean_directory(local_directory)
        clean_directory(output_directory)
    except Exception as e:
        st.error(f"Error during cleanup: {e}")

# Register the cleanup function to run when the session ends
st.session_state['_cleanup'] = cleanup
