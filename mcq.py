import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import PyPDF2  # Library for PDF processing

load_dotenv()
KEY = os.getenv('OPENAI_API_KEY')  # Correct typo here
llm = ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=0.5)

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. 
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

TEMPLATE2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE,
)

quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)
review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)
generate_evaluate_chain = SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                            output_variables=["quiz", "review"], verbose=True)

st.write("<h1 style='text-align: center;'> MCQs Creator Application With LangChain And OpenAI </h1>", unsafe_allow_html=True)
st.caption("Note: This app acts as an expert MCQs maker. It is assumed that you provide the appropriate subject-related data to generate the quiz for students.")

def get_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        try:
            text = uploaded_file.read().decode("utf-8")
            return text
        except UnicodeDecodeError:
            st.error("Error decoding text. Try uploading with different encoding.")
            return None
    elif uploaded_file.type.startswith("application/pdf"):
        try:
            # Use libraries like PyMuPDF or Camelot to extract text and handle images
            # For example with PyMuPDF
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        return None

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    mcq_count = st.number_input("Number of MCQs", min_value=3, max_value=50, value=3)
    subject = st.text_input("Subject", max_chars=20)
    tone = st.text_input("Complexity Level Of Questions", max_chars=20, value="Simple")

    button = st.form_submit_button("Create MCQs")

if button:
    if uploaded_file is None:
        st.error("Please upload a file.")
    elif not subject:
        st.error("Please provide a subject.")
    else:
        with st.spinner("Loading..."):
            try:
                text = get_text_from_file(uploaded_file)
                if text is None:
                    st.error("Error getting text from file. Please try again.")
                else:
                    with get_openai_callback() as cb:
                        response = generate_evaluate_chain(
                            {
                                "text": text,
                                "number": mcq_count,
                                "subject": subject,
                                "tone": tone,
                                "response_json": json.dumps(RESPONSE_JSON),
                            }
                        )
                        quiz_json_str = response.get("quiz", None)
                        print("Quiz JSON String:", quiz_json_str)  # Debugging statement
                        try:
                            if quiz_json_str is not None:
                                # Extract only the JSON response
                                quiz_json_str = quiz_json_str.split("RESPONSE_JSON", 1)[-1].strip()
                                quiz = json.loads(quiz_json_str)
                            else:
                                st.error("Quiz JSON string is empty.")
                                quiz = None
                        except json.JSONDecodeError as e:
                            st.error(f"Error decoding quiz: {e}")
                            st.error("Quiz JSON String:", quiz_json_str)
                            quiz = None
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error(f"Error: {e}")
                quiz = None
            finally:
                if quiz is not None and isinstance(quiz, dict):
                    try:
                        quiz_list = [question for question in quiz.values()]
                        df = pd.DataFrame(quiz_list)
                        df.index += 1

                        st.subheader("Generated MCQs:")
                        st.table(df)

                        # Assign the value of 'review' from the response
                        review = response.get("review", "")

                        st.subheader("Review:")
                        st.text_area(label="", value=review, height=100)

                        download_button = st.download_button(
                            label="Download MCQ Data",
                            data=df.to_csv().encode('utf-8'),
                            file_name=f"{subject}_MCQ_Data.csv",
                            key='download_button'
                        )

                    except json.JSONDecodeError as e:
                        st.error(f"Error decoding quiz: {e}")
                else:
                    st.error("No quiz generated by LangChain.")



