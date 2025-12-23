import os
import pytesseract
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from pymilvus import connections, utility


#step 1 - identify file type
def filetype(file_path):
    if file_path.startswith(('http://', 'https://')):
        return 'url'
    elif file_path.endswith('.txt'):
        return 'text'
    elif file_path.endswith('.pdf'):
        return 'pdf'
    elif file_path.endswith('.docx'):
        return 'docx'
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        return 'image'
    else:
        return 'unknown file type'
    

# step 2 - route to appropriate extraction function
def extract_content(file_path):
    ftype= filetype(file_path)
    
    if ftype == 'url':
        return extract_from_url(file_path)
    elif ftype == 'text':
        return extract_from_text(file_path)
    elif ftype == 'pdf':
        return extract_from_pdf(file_path)
    elif ftype == 'docx':
        return extract_from_docx(file_path)
    elif ftype == 'image':
        return extract_from_image(file_path)
    else:
        raise ValueError("Unsupported file type for extraction")
    
# step 3 - define extraction functions for each file type
def extract_from_url(input):
    loader = WebBaseLoader(input)
    return loader.load()

def extract_from_text(input):
    loader = TextLoader(input)
    return loader.load()

def extract_from_pdf(input):
    loader = PyPDFLoader(input)
    return loader.load()

def extract_from_docx(input):
    loader = Docx2txtLoader(input)
    return loader.load()

def extract_from_image(input):
    text = pytesseract.image_to_string(input)
    return [
        Document(
            page_content=text,
            metadata={"source": input, "type": "image"}
        )
    ]


# vectorstore
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Milvus(
    embedding_function=embedding_model,
    collection_name="RAG_Collection",
    connection_args={"host": "localhost", "port": "19530"}
)

# processing function
def process_input(file_path):
    # 1. Extract content
    docs = extract_content(file_path)

    # 2. split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, add_start_index=True)
    docs_split = splitter.split_documents(docs)

    #3. clean metadata
    for doc in docs_split:
        doc.metadata = {
            "source": doc.metadata.get("source", file_path)}

    # 3. Store in Milvus
    vectorstore.add_documents(documents=docs_split)

    return len(docs_split)


# processing query
llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct", temperature=0.1, max_tokens=512)
def retrieve_context(query: str):
    retrieved_docs = vectorstore.similarity_search(query, k=6)
    if not retrieved_docs:
        return "No context found for this query.", []

    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = [doc.metadata for doc in retrieved_docs]

    answer_response = llm.generate([
        [HumanMessage(content=f"Answer the question using ONLY the context below:\n\n{context_text}\n\nQuestion: {query}")]
    ])
    
    answer_text = answer_response.generations[0][0].text
    return answer_text, sources


# Example usage'

file_path = "example.pdf"
chunks_added = process_input(file_path)
print(f"Loaded {chunks_added} chunks into vector store")

query = "Describe your professional experience"
answer, sources = retrieve_context(query)
print("Answer:", answer)
print("Sources:", sources)
