import streamlit as st
import pdfplumber
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
import spacy
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Load model and tokenizer for T5
model_name = 'valhalla/t5-small-e2e-qg'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to extract text from PDF
def extract_text_from_pdf(pdf):
    with pdfplumber.open(pdf) as pdf_file:
        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
    return text

# Function to generate questions from text
def generate_questions_from_text(text, model, tokenizer, num_questions=5, start_phrases=["how", "which", "do"]):
    generated_questions = []
    for start_phrase in start_phrases:
        input_text = f"{start_phrase}: {text}"
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate(inputs['input_ids'],
                                 max_length=100,
                                 num_return_sequences=num_questions,
                                 num_beams=5,  # Adjust beam width for diversity
                                 no_repeat_ngram_size=2,  # Avoid repeating n-grams
                                 early_stopping=True  # Stop generation when the model has finished generating sequences
                                 )
        questions = list(set([tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]))  # Convert outputs to a set to ensure unique questions and strip whitespace
        generated_questions.extend(questions)
    return generated_questions[:num_questions]

# Function to format and remove duplicate questions
def format_and_remove_duplicates(generated_questions):
    seen_questions_global = set()
    formatted_questions = []
    question_counter = 1

    for i, question_set in enumerate(generated_questions, start=1):
        question_segments = question_set.split('<sep>')
        questions = []
        for segment in question_segments:
            questions.extend([q.strip() + '?' for q in segment.split('?') if q.strip()])

        unique_questions_local = []
        seen_questions_local = set()

        for q in questions:
            if q not in seen_questions_local:
                unique_questions_local.append(q)
                seen_questions_local.add(q)

        # Add unique questions from the local set to the global set
        for q in unique_questions_local:
            if q not in seen_questions_global:
                formatted_questions.append(f"Question {question_counter}: {q}")
                seen_questions_global.add(q)
                question_counter += 1

    return formatted_questions

# Function to split text into sentences
def split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

# Function to extract answers from text based on the question
def extract_answers(question, sentences):
    doc = nlp(question)
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b', re.IGNORECASE)

    for sentence in sentences:
        if pattern.search(sentence):
            return sentence

    return "Answer not found in the text."

# Function to calculate ROUGE scores
def calculate_rouge(reference_questions, formatted_questions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []

    for i, gen_q in enumerate(formatted_questions):
        # Join reference questions for comparison
        reference = reference_questions[i] if i < len(reference_questions) else reference_questions[-1]
        # Calculate ROUGE scores
        score = scorer.score(reference[0], gen_q.split(":")[1].strip())
        scores.append(score)

    return scores

# Function to get conversational chain with Google Generative AI
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, api_key=genai_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Streamlit app
def main():
    st.set_page_config(page_title="Question-Answer Bot")
    st.title("Question-Answer Bot")

    if "question_answer_pairs" not in st.session_state:
        st.session_state.question_answer_pairs = []

    if "rouge_scores" not in st.session_state:
        st.session_state.rouge_scores = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                for pdf in pdf_docs:
                    raw_text = extract_text_from_pdf(pdf)
                    sentences = split_into_sentences(raw_text)
                    generated_questions = generate_questions_from_text(raw_text, model, tokenizer, num_questions=5)
                    formatted_questions = format_and_remove_duplicates(generated_questions)
                    question_answer_pairs = [(q, extract_answers(q, sentences)) for q in formatted_questions]
                    st.session_state.question_answer_pairs.extend(question_answer_pairs)

                reference_questions = [
                    ["What is immortality within human reach?"],
                    ["By what century is the prospect of living up to 5000 years might well become reality?"],
                    ["What does the idea that death is a key to life are at best based on dubious science?"],
                    ["What do Chipko activists in TehriGarhwal sing praising their hills as paradise, the place of Gods, where the mountains bloom with rare plants and dense cedars?"],
                    ["What was the name of?"],
                    ["By what century is the prospect of living up to 5000 years based on dubious science?"],
                    ["What does the scientific fraternity rarely take seriously?"],
                    ["What did Chipko activists sing in the 1970s?"],
                    ["What was the name of the movement to save the indigenous forests of oak and rhododendron from being felled by the Forest Department?"],
                    ["What was the name of the movement to save the indigenous forests of oak and rhododendron?"],
                    ["What does the idea that death is key to life are at best based on?"],
                    ["What do Chipko activists in TehriGarhwal sing?"],
                    ["What did ChipKo protest against?"]
                ]
                st.session_state.rouge_scores = calculate_rouge(reference_questions, [q for q, _ in st.session_state.question_answer_pairs])
                st.success("Done")

    st.header("Generated Question-Answer Pairs")
    for i, (question, answer) in enumerate(st.session_state.question_answer_pairs, 1):
        st.subheader(f"Q{i}: {question}")
        st.write(f"A{i}: {answer}")
        if i <= len(st.session_state.rouge_scores):
            rouge_score = st.session_state.rouge_scores[i-1]
            st.write(f"ROUGE score for Question {i}:")
            for metric, score in rouge_score.items():
                st.write(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-score={score.fmeasure:.4f}")

if __name__ == "__main__":
    main()
