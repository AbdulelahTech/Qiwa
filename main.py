from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import openai
import os
# from dotenv.main import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
import docx
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS


llm = ChatOpenAI(model='gpt-4-turbo',api_key='sk-proj-wpZrq4sZaw26n1MgtN1iT3BlbkFJ0eSFGg4b7fkvK2PLeqS2')
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
app = Flask(__name__)
CORS(app)

chain = None
docsearch: FAISS = None

# Routes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    text = data.get('data')
    user_input = text

    try:
        docs = docsearch.similarity_search(user_input)
        input = {'input_documents': docs, 'question': user_input}
        output = chain.invoke(input=input)
        return jsonify({"response": True, "message": output["output_text"]})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message": error_message, "response": False})


# Functions
def init_qa_system(doc_path):
    global chain, docsearch

    doc = docx.Document(doc_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + ' '

    # split into chunks
    char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                               chunk_overlap=200, length_function=len)

    text_chunks = char_text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(text_chunks, embeddings)

    chain = load_qa_chain(llm, chain_type="stuff")


if __name__ == '__main__':
    doc_path = './trainingData/laborLaw.docx'  # Replace with the actual path
    init_qa_system(doc_path)
    app.run(debug=True)
