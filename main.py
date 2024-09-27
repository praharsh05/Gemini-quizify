import streamlit as st
from quiz_manager import QuizManager
from quiz_generator import QuizGenerator
from chroma_collection_creator import ChromaCollectionCreator
from embedding_client import EmbeddingClient
from document_processor import DocumentProcessor
from dotenv import load_dotenv
import os

if __name__ == "__main__":

    load_dotenv()

    embed_config = {
        "model_name": os.getenv('MODEL_NAME'),
        "project": os.getenv('PROJECT'),
        "location": os.getenv('LOCATION')
    }

    # Add Session State
    if 'question_bank' not in st.session_state or len(st.session_state['question_bank']) == 0:

        # Step 1: init the question bank list in st.session_state
        st.session_state["question_bank"] = []

        screen = st.empty()
        with screen.container():
            st.header("Quiz Builder")

            # Create a new st.form flow control for Data Ingestion
            with st.form("Load Data to Chroma"):
                st.write(
                    "Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

                processor = DocumentProcessor()
                processor.ingest_documents()

                embed_client = EmbeddingClient(**embed_config)

                chroma_creator = ChromaCollectionCreator(
                    processor, embed_client)

                # Step 2: Set topic input and number of questions
                topic_input = st.text_input(
                    "Topic for Generative Quiz", placeholder="Enter the topic of the document")
                questions = st.slider(
                    "Number of Questions", min_value=1, max_value=10, value=1)

                submitted = st.form_submit_button("Submit")

                if submitted:
                    chroma_creator.create_chroma_collection()

                    if len(processor.pages) > 0:
                        st.write(
                            f"Generating {questions} questions for topic: {topic_input}")

                    # Step 3: Initialize a QuizGenerator class using the topic, number of questrions,
                    #  and the chroma collection
                    generator = QuizGenerator(
                        topic_input, questions, chroma_creator)
                    question_bank = generator.generate_quiz()

                    # Step 4: Initialize the question bank list in st.session_state
                    st.session_state["question_bank"] = question_bank

                    # Step 5: Set a display_quiz flag in st.session_state to True
                    st.session_state["display_quiz"] = True

                    # Step 6: Set the question_index to 0 in st.session_state
                    st.session_state["question_index"] = 0

                    screen.empty()

    if st.session_state.get("display_quiz", False):

        with st.container():
            st.header("Generated Quiz Question: ")
            quiz_manager = QuizManager(st.session_state['question_bank'])

            # Format the question and display it
            with st.form("MCQ"):
                # Step 7: Set index_question using the Quiz Manager method get_question_at_index
                #  passing the st.session_state["question_index"]
                index_question = quiz_manager.get_question_at_index(
                    st.session_state["question_index"])

                # Unpack choices for radio button
                choices = []
                for choice in index_question['choices']:
                    key = choice['key']
                    value = choice['value']
                    choices.append(f"{key}) {value}")

                # Display the Question
                st.write(
                    f"{st.session_state['question_index'] + 1}. {index_question['question']}")
                answer = st.radio(
                    "Choose an answer",
                    choices,
                    index=None
                )

                answer_choice = st.form_submit_button("Submit")

                # Step 8: Navigate to the next and previous questions Here we use the
                #  next_question_index method from the QuizManager class
                st.form_submit_button(
                    "Next Question", on_click=lambda: quiz_manager.next_question_index(direction=1))
                st.form_submit_button(
                    "Previous Question", on_click=lambda: quiz_manager.next_question_index(direction=-1))

                if answer_choice and answer is not None:
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
                    st.write(f"Explanation: {index_question['explanation']}")
