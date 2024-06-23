import PyPDF2
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
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model='gpt-4o')
memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="human_input")
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
    passphrase = data.get('passphrase')
    if passphrase != 'قوي':
        return jsonify({"message": 'your not authorized', "response": False})
    user_input = text

    try:
        docs = docsearch.similarity_search(user_input)
        input = {'input_documents': docs, 'human_input': user_input}
        output = chain.invoke(input=input)
        memory.save_context({"human_input": user_input}, {
                            "output": output["output_text"]})
        return jsonify({"response": True, "message": output["output_text"]})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message": error_message, "response": False})


# Functions
def init_qa_system(doc_path):
    global chain, docsearch

    text = ''
    if doc_path.lower().endswith('.docx'):
        doc = docx.Document(doc_path)
        for para in doc.paragraphs:
            text += para.text + '\n'
    elif doc_path.lower().endswith('.pdf'):
        with open(doc_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + '\n'
    else:
        raise ValueError(
            "Unsupported file format. Please provide a DOCX or PDF file.")

    # Split into chunks
    char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                               chunk_overlap=200, length_function=len)
    text_chunks = char_text_splitter.split_text(text)

    template = """
    انت مساعد ذكي تتحدث مع انسان، اسمك قوي
مرفق إليك نظام العمل السعودي يجب ان تكون كل إجاباتك منه
لا تستخدم كلمت نظام العمل السعودي لأن كل الاستشارات تقوم على هذا المبدأ
اطلب من المستخدم ان يعطيك جميع المعلومات التي تحتاجها للإجابة بطريقة نموذجية
اجعل اجاباتك مختصرة قدر الامكان وانهها برقم المادة التي استندت اليها واجعلها بين قوسين
لا تعطي تفاصيل ان لم يطلبها المستخدم
يجب ان يكون ردك واضح وسهل الفهم وأن لايتعدى الثلاث 
الجمل
اذا سؤلت خارج اطار نظام العمل السعودي اجب لا استطيع مساعدتك في ذلك، ارجوا ان تسألني عن نظام العمل السعودي
                معطى لك أجزاء من مستند طويل (نظام العمل السعودي) وسؤال، اصنع الإجابة النهائية.

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot:
                """

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(text_chunks, embeddings)

    chain = load_qa_chain(llm, chain_type="stuff",
                          memory=memory, prompt=prompt)


if __name__ == '__main__':
    doc_path = './trainingData/hrsd.pdf'  # Replace with the actual path
    init_qa_system(doc_path)
    app.run(host='0.0.0.0', port=4000)
