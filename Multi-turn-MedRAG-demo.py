

'''
1.	环境准备：安装必要的库。
2.	数据准备：将不同格式的文档转换为纯文本格式，并存储在 Milvus 中。
3.	模型加载：加载预训练的 LlamaIndex 模型。
4.	数据检索：实现从 Milvus 中检索相关文档的功能。
5.	多轮对话管理：实现多轮对话逻辑，维护对话状态。
6.	构建Web服务：构建一个简单的Web服务接口，用于接收用户输入并返回模型回复。
'''
'''
1. 环境准备
首先，安装必要的库：
Bash
深色版本
'''
!pip install llama-index pymilvus langchain

'''
2. 数据准备
将不同格式的文档转换为纯文本
Python
深色版本
'''
import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup

def convert_pdf_to_text(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def convert_docx_to_text(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def convert_html_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        text = soup.get_text()
    return text

def convert_txt_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def process_documents(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.pdf'):
                text = convert_pdf_to_text(file_path)
            elif file_path.endswith('.docx'):
                text = convert_docx_to_text(file_path)
            elif file_path.endswith('.html'):
                text = convert_html_to_text(file_path)
            elif file_path.endswith('.txt'):
                text = convert_txt_to_text(file_path)
            else:
                continue
            documents.append({
                'file_path': file_path,
                'text': text
            })
    return documents

'''
3. 数据存储到 Milvus
Python
深色版本
'''

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 连接到 Milvus
connections.connect("default", host="localhost", port="19530")

# 定义集合 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]

schema = CollectionSchema(fields, "Medical Documents Collection")

# 创建集合
collection_name = "medical_docs"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# 插入数据
def insert_documents(documents):
    from sentence_transformers import SentenceTransformer

    # 加载句子编码器
    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # 编码文档
    texts = [doc['text'] for doc in documents]
    embeddings = encoder.encode(texts)

    # 插入数据
    data = [
        [i for i in range(len(documents))],
        [doc['text'] for doc in documents],
        embeddings.tolist()
    ]
    collection.insert(data)

# 加载和插入文档
documents = process_documents('/path/to/your/documents')
insert_documents(documents)

'''
4. 加载模型
Python
深色版本
'''
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import MilvusVectorStore

# 创建 Milvus 向量存储
vector_store = MilvusVectorStore(
    host="localhost",
    port="19530",
    collection_name="medical_docs"
)

# 创建 LlamaIndex 索引
index = VectorStoreIndex.from_vector_store(vector_store)

'''
5. 数据检索
Python
深色版本
'''
def search_documents(query, top_k=5):
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return response.source_nodes

'''
6. 多轮对话管理
Python
深色版本
'''
class DialogueManager:
    def __init__(self):
        self.history = []

    def add_to_history(self, user_input, response):
        self.history.append((user_input, response))

    def get_context(self):
        return " ".join([f"User: {u} Assistant: {r}" for u, r in self.history])

dialogue_manager = DialogueManager()

'''
7. 构建Web服务
Python
深色版本
'''
from flask import Flask, request, jsonify

app = Flask(__name__)

def generate_response(user_input):
    context = dialogue_manager.get_context()
    # 搜索相关文档
    related_docs = search_documents(user_input)
    context += " ".join([doc.node.text for doc in related_docs])
    
    # 使用 LlamaIndex 获取回复
    query_engine = index.as_query_engine()
    response = query_engine.query(user_input, context=context)
    dialogue_manager.add_to_history(user_input, response.response)
    return response.response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

