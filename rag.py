# 实现一个简单的RAG，以便从PDF文档或word文档中查询文本内容。
# 因为python环境兼容问题，这里我修改了word文档的加载类。
 
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.tools import StructuredTool
from common import *

 
def get_file_extension(filename: str) -> str:
    return filename.split(".")[-1]

 
class FileLoadFactory:
    @staticmethod
    def get_loader(filename: str):
        filename = filename.strip()
        ext = get_file_extension(filename)
        if ext == "pdf":
            return PyPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return Docx2txtLoader(filename)
        else:
            raise NotImplementedError(f"File extension {ext} not supported.")

 
def load_docs(filename: str) -> List[Document]:
    file_loader = FileLoadFactory.get_loader(filename)
    pages = file_loader.load_and_split()
    return pages

# #### 使用 RAG 查询文档
def ask_docment(
        filename: str,
        query: str,
) -> str:
    """根据一个PDF文档的内容，回答一个问题"""

    raw_docs = load_docs(filename)
    if len(raw_docs) == 0:
        return "抱歉，文档内容为空"
    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        add_start_index=True,
                    )
    documents = text_splitter.split_documents(raw_docs)
    if documents is None or len(documents) == 0:
        return "无法读取文档内容"
    db = Chroma.from_documents(documents, OpenAIEmbeddings(model="text-embedding-ada-002"))
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(
            temperature=0,
            model_kwargs={
                "seed": 42
            },
        ),  # 语言模型
        chain_type="stuff",  # prompt的组织方式
        retriever=db.as_retriever()  # 检索器
    )
    response = qa_chain.invoke(query+"(请用中文回答)")
    return response

 
# #### 定义 tool 函数
document_qa_tool = StructuredTool.from_function(
    func=make_safe_tool(ask_docment),
    name="AskDocument",
    description="根据一个Word或PDF文档的内容，回答一个问题。考虑上下文信息，确保问题对相关概念的定义表述完整。",
) 

 
# #### 使用示例
# 使用示例
# filename = "./data/autogpt-demo/供应商资格要求.pdf"
# query = "销售额的达标标准是多少？"
# result = ask_docment(filename, query)
# print(type(result).__name__)
# print(result)

 
# 使用示例
# filename = "./data/autogpt-demo/供应商资格要求.pdf"
# query = "销售额的达标标准是多少？"
# result = ask_docment(filename, query)
# print(type(result).__name__)
# print(result)

 
# <div class="alert alert-warning">
#     <b>工具的输出必须是字符串</b><br>
# make_safe_tool已经确保工具执行结果是字符串，而不是字典等其他类型。    
# </div>

 
# result = document_qa_tool.invoke({"filename": filename, "query": "销售额的达标标准是多少？"})
# print(type(result).__name__)
# print(result)

 
# 使用示例
# filename = "./data/autogpt-demo/求职简历.docx"
# query = "工作经历如何？"
# print(ask_docment(filename, query))