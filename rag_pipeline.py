import os
import shutil
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional

# 加载 .env 环境变量
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
    documents = []
    for page in extracted_pages:
        documents.append(
            Document(
                page_content=page["text"],
                metadata={
                    "source": page["source"],
                    "page": page["page"]
                }
            )
        )
    return documents


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
    siliconflow_key = os.environ.get("SILICONFLOW_API_KEY")
    if siliconflow_key:
        return ChatOpenAI(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            temperature=0.2,
            base_url="https://api.siliconflow.cn/v1",
            api_key=siliconflow_key
        )
    raise RuntimeError(
        "未检测到 LLM API Key。请设置 OPENAI_API_KEY 或 SILICONFLOW_API_KEY。"
    )


def handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, 0o700)
        func(path)
    except Exception:
        pass


def ensure_vectorstore_directory(base_dir: str) -> str:
    target = Path(base_dir)
    if target.exists():
        try:
            shutil.rmtree(target, onerror=handle_remove_readonly)
            print(f"🔄 已删除旧向量数据库：{target}")
            return str(target)
        except Exception as err:
            fallback = target.parent / f"{target.name}_{uuid.uuid4().hex[:8]}"
            print(f"⚠️ 无法删除旧向量数据库，改用新目录：{fallback}")
            return str(fallback)
    return str(target)


def load_embeddings() -> object:
    siliconflow_key = os.environ.get("SILICONFLOW_API_KEY")
    if siliconflow_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="BAAI/bge-m3",           # multilingual model on SiliconFlow
                api_key=siliconflow_key,
                base_url="https://api.siliconflow.cn/v1",
                check_embedding_ctx_length=False  # avoid OpenAI-specific token check
            )
        except Exception as e:
            raise RuntimeError(f"SiliconFlow embeddings 初始化失败: {e}")

    raise RuntimeError(
        "未找到 SILICONFLOW_API_KEY，请在 .env 文件中设置。"
    )


def build_vectorstore(chunks: list[Document], persist_directory: str = "./pdf_chroma_db", embeddings: Optional[object] = None) -> Chroma:
    """Build Chroma vectorstore with explicit embeddings (multilingual preferred)."""
    if embeddings is None:
        embeddings = load_embeddings()
    persist_directory = ensure_vectorstore_directory(persist_directory)
    '''return Chroma.from_documents(
        documents=chunks,
        persist_directory=persist_directory,
        collection_name="pdf_documents",
        embedding_function=embeddings,
    )'''
    vectorstore = Chroma(
        collection_name="pdf_documents",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def dedupe_search_results(chunks: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    seen = set()
    unique_chunks = []
    for doc, score in chunks:
        chunk_id = doc.metadata.get("chunk_id")
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            chunk_id,
            doc.page_content[:120]
        )
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append((doc, score))
    return unique_chunks


def retrieve_top_chunks(vectorstore: Chroma, query: str, k: int = 3):
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)


def answer_question_with_llm(llm: ChatOpenAI, question: str, chunks: list[tuple[Document, float]]) -> str:
    context = []
    for doc, score in chunks:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context.append(
            f"来源: {source} 第{page}页\n{doc.page_content}"
        )

    prompt = ChatPromptTemplate.from_template(
        "你是一位助手。请根据下面的文档片段回答用户问题。\n\n文档片段：\n{context}\n\n问题：{question}\n\n请用中文简明回答。"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": "\n\n".join(context), "question": question})


def print_search_results(chunks: list[tuple[Document, float]]):
    for rank, (doc, score) in enumerate(chunks, start=1):
        print("=" * 60)
        print(f"结果 {rank}")
        print(f"  来源: {doc.metadata.get('source', 'unknown')} 页: {doc.metadata.get('page', '?')} chunk_id: {doc.metadata.get('chunk_id', '?')}")
        print(f"  相似度: {score:.4f}")
        print(f"  内容预览: {doc.page_content[:120].replace('\n', ' ').strip()}...")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("用法: python day3_rag_pipeline.py <pdf_path_or_directory> [query]")
        print("示例: python day3_rag_pipeline.py sample.pdf \"什么是RAG？\"")
        print("示例: python day3_rag_pipeline.py ./pdf_sources \"什么是影响力最大化\"")
        return

    pdf_input = sys.argv[1]
    if not Path(pdf_input).exists():
        print(f"错误：找不到路径: {pdf_input}")
        return

    pdf_files = []
    
    # 如果是目录，获取所有 PDF 文件
    if Path(pdf_input).is_dir():
        pdf_files = list(Path(pdf_input).glob("*.pdf"))
        if not pdf_files:
            print(f"错误：{pdf_input} 目录中没有 PDF 文件")
            return
        print(f"📁 找到 {len(pdf_files)} 个 PDF 文件")
    else:
        pdf_files = [Path(pdf_input)]

    # 处理所有 PDF 文件
    all_documents = []
    for pdf_path in pdf_files:
        print(f"📄 读取 PDF: {pdf_path.name}")
        documents = build_documents_from_pdf(str(pdf_path))
        all_documents.extend(documents)
    
    chunks = split_documents(all_documents)
    print(f"🔪 已生成 {len(chunks)} 个 chunks")

    print("💾 构建 Chroma 向量数据库...")
    vectorstore = build_vectorstore(chunks)

    if len(sys.argv) >= 3:
        question = sys.argv[2]
    else:
        question = input("请输入问题: ")

    print("\n🔍 开始检索最相关的文档片段...\n")
    raw_chunks = retrieve_top_chunks(vectorstore, question, k=10)
    top_chunks = dedupe_search_results(raw_chunks)[:4]
    print_search_results(top_chunks)

    try:
        llm = load_llm()
        print("\n🤖 使用 LLM 生成答案...\n")
        answer = answer_question_with_llm(llm, question, top_chunks)
        print("最终答案:\n", answer)
    except RuntimeError as e:
        print(f"\n⚠️ LLM 未启用：{e}")
        print("你已经看到了检索结果，可以根据这些内容手动回答。")


if __name__ == "__main__":
    main()
