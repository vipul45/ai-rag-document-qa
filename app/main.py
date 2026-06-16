from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Document Q&A")

# Vector Store
embeddings = OpenAIEmbeddings()
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="documents",
    connection=os.getenv("DATABASE_URL"),
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "PDF only for now")
    content = await file.read()
    with open(file.filename, "wb") as f:
        f.write(content)
    loader = PyPDFLoader(file.filename)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store.add_documents(chunks)
    return {"status": "ingested", "chunks": len(chunks)}

@app.post("/query")
async def query(request: QueryRequest):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini")
    template = """Answer based on context: {context}
Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(request.question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)