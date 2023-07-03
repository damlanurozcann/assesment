from fastapi import FastAPI, UploadFile, File
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv
import uvicorn
import os

app = FastAPI()
vector_store = FAISS()
doc_loader = TextLoader()
embedder = OpenAIEmbeddings()
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(embedding_function=embedder.embed_text, index=vector_store, docstore=doc_loader, index_to_docstore_id=vector_store.add_documents)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    content = await file.read()
    chunks = doc_loader.split_pdf(content)

    embeddings = embedder.embed_chunks(chunks)
    vector_store.add_documents(embeddings)

    return {"message": "PDF dosyası başarıyla yüklendi ve parçalara ayrılarak vector veritabanına eklendi."}


@app.post("/question")
async def ask_question(question: str):

    question_embedding = embedder.embed_text(question)

    similar_docs = vector_store.most_similar(question_embedding, k=5)
    results = []
    for doc in similar_docs:
        result = {
            "document_id": doc.id,
            "score": doc.score,
            "text": doc_loader.get_text(doc.id)
        }
        results.append(result)

    answer = llm.ask_question(question, api_key=openai_api_key)
    results.append(answer)

    return {"results": results}