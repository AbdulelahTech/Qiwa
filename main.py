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

llm = ChatOpenAI(model='gpt-4-turbo',
                 api_key='sk-proj-wpZrq4sZaw26n1MgtN1iT3BlbkFJ0eSFGg4b7fkvK2PLeqS2')
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

    doc = docx.Document(doc_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'

    # split into chunks
    char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                               chunk_overlap=200, length_function=len)

    text_chunks = char_text_splitter.split_text(text)

    template = """انت مساعد ذكي تتحدث مع انسان، اسمك قوي.
يجب ان يكون ردك واضح وسهل الفهم
ردك يجب ان لايتعدى ثلاث جمل
لاتعطي تفاصيل ان لم يطلبها المستخدم
اعتمد على نظام العمل السعودي كمرجع اساسي في تقديم الاستشارات.
اجعل اجاباتك مختصرة قدر الامكان وانهها برقم المادة التي استندت اليها واجعلها بين قوسين 
لا تستخدم كلمت نظام العمل السعودي لأن كل الاستشارات تقوم على هذا المبدأ 
اطلب من المستخدم ان يعطيك جميع المعلومات التي تحتاجها للإجابة بطريقة نموذجية
اعط اجابه قصيرة ومختصرة قدر الامكان ولا تفصل الا اذا طلب المستخدم ذلك 
اهم الابواب التي يجب التركيز عليها (5,6,8,15) و اهم المواد التي يجب التركيز عليها ( 55 ، 53 ، 56 ، 66 ، 67 ، 68 ، 69 ، 70 ، 71 ، 74 ،  75 ، 76 ، 77 ، 80 ، 81 ، 82 ، 84 ، 85 ، 88 ، 111 ،117) وهذا لا يعني ان باقي المواد اقل اهميه
حقوق الموظف في القطاع الخاص 
التأمين الطبي م. ١٤٤
راتب الاجازة م.١٠٩
ساعات العمل الاضافي م. ١٠٧
دفع الاجور م. ٩٠
مكافأة نهاية الخدمة م. ٨٤ و٨٥
رصيد الاجازة التي لم تستخدم م.١١١
يوم راحة في الاسبوع م.١٠٤
حقوق الموظف عند ترك العمل 


في حال الاستقالة (مكافأة نهاية الخدمة م. 85)
في حال انتهاء العقد او الانهاء من جانب صاحب العمل (م. 84)
مستحقات الاجازة التي لم تستخدم (م. 111)
شهادة الخدمة (م. 64)
المدة المسموح بها في صرف مكافأة نهاية الخدمة (م. 88)
إذا وقعت ايام الاجازة المرضية أثناء الاجازة السنوية فتوقف أيام الاجازة السنوية الى حين انتهاء الاجازة المرضية ثم تستأنف المدة المتبقية 
بشرط ان يكون لديك تقرير طبي بالإجازة المرضية

المستند النظامي المادة 26 من اللائحة التنفيذية لنظام العمل
يحق للموظف تقديم اجازة ولا يجوز لصاحب العمل رفضها 
١. اجازة المولود
٢. الزواج 
٣. الوفاة 
٤. الحج 
٥. الاختبارات 
٦. المرضية 
٧. الوضع 
٨.  العدة

معطى لك أجزاء من مستند طويل (نظام العمل السعودي) وسؤال، اصنع الإجابة النهائية.


                {context}

                {chat_history}
                Human: {human_input}
                Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    # create embeddings
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(text_chunks, embeddings)

    chain = load_qa_chain(llm, chain_type="stuff",
                          memory=memory, prompt=prompt)


if __name__ == '__main__':
    doc_path = './trainingData/laborLaw.docx'  # Replace with the actual path
    init_qa_system(doc_path)
    app.run(debug=True)
