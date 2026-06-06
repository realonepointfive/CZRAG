import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "source": os.path.basename(pdf_path),
            "page": page_index,
            "text": text.strip()
        })
    return pages


def build_documents_from_pdf(pdf_path: str) -> list[Document]:
    extracted_pages = extract_text_from_pdf(pdf_path)
    return [
        Document(
            page_content=page["text"],
            metadata={"source": page["source"], "page": page["page"]}
        )
        for page in extracted_pages
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"chunk-{idx}"
    return chunks


def load_llm() -> ChatOpenAI:
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        raise RuntimeError("未找到 SILICONFLOW_API_KEY，请在 .env 中设置。")
    return ChatOpenAI(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        temperature=0.2,
        base_url="https://api.siliconflow.cn/v1",
        api_key=key,
    )


def load_embeddings() -> OpenAIEmbeddings:
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        raise RuntimeError("未找到 SILICONFLOW_API_KEY，请在 .env 中设置。")
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=key,
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )
