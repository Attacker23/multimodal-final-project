import os
import uuid
import shutil
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from .pdf_utils import extract_text_from_pdf
from .embeddings import embed_text
from .topic_classifier import classify_text_to_topic_by_descriptions
from config import (
    VECTOR_DB_DIR,
    PAPER_COLLECTION_NAME,
    TEXT_EMBEDDING_MODEL,
    PAPER_DIR,
    DEFAULT_TOPIC,
    TOPIC_MIN_SIM,
    TOPIC_DESCRIPTIONS,

)


def get_paper_collection():
    client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(allow_reset=True),
    )
    return client.get_or_create_collection(PAPER_COLLECTION_NAME)


def _list_pdfs(path: str) -> List[str]:
    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        return [path]
    pdfs = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return pdfs


def _safe_filename(name: str) -> str:
    # Windows 不允许的字符简单处理
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name


def _archive_pdf(pdf_path: str, topic: str, move: bool = True) -> str:
    """
    把 pdf 移动/复制到 data/papers/<topic>/ 下，返回新路径
    """
    topic = _safe_filename(topic or DEFAULT_TOPIC)
    target_dir = os.path.join(PAPER_DIR, topic)
    os.makedirs(target_dir, exist_ok=True)

    base = os.path.basename(pdf_path)
    base = _safe_filename(base)
    target_path = os.path.join(target_dir, base)

    # 避免重名覆盖：追加短 uuid
    if os.path.exists(target_path):
        stem, ext = os.path.splitext(base)
        target_path = os.path.join(target_dir, f"{stem}_{uuid.uuid4().hex[:8]}{ext}")

    if move:
        shutil.move(pdf_path, target_path)
    else:
        shutil.copy2(pdf_path, target_path)

    return target_path


def add_and_classify_papers(
    path: str,
    topics: List[str],
    move_to_topic_dir: bool = True,
):
    """
    单文件/批量添加：自动判别主题 -> 归档到 data/papers/<topic>/ -> 写入向量库
    """
    os.makedirs(PAPER_DIR, exist_ok=True)
    pdf_paths = _list_pdfs(path)
    if not pdf_paths:
        print("未找到任何 PDF 文件。")
        return

    collection = get_paper_collection()

    for pdf_path in pdf_paths:
        print(f"处理: {pdf_path}")
        try:
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                print("  -> PDF 无可提取文本，跳过")
                continue

            # 用前几千字做判别即可
            snippet = text[:6000]

            topic, sim = classify_text_to_topic_by_descriptions(
                snippet,
                topic_labels=topics,  # 命令行短标签
                topic_descriptions=TOPIC_DESCRIPTIONS,  # 内部描述
                model_name=TEXT_EMBEDDING_MODEL,
                min_sim=TOPIC_MIN_SIM,
                default_topic=DEFAULT_TOPIC,
            )

            new_path = _archive_pdf(pdf_path, topic, move=move_to_topic_dir)
            print(f"  -> 主题: {topic}  相似度: {sim:.3f}")
            print(f"  -> 归档到: {new_path}")

            emb = embed_text(snippet, TEXT_EMBEDDING_MODEL)
            doc_id = str(uuid.uuid4())

            collection.add(
                ids=[doc_id],
                embeddings=[emb],
                metadatas=[{
                    "path": os.path.abspath(new_path),
                    "topic": topic,
                    "topic_sim": sim,
                }],
                documents=[text[:10000]],
            )

        except Exception as e:
            print(f"  -> 失败: {e}")
            continue

    print("添加完成。当前论文数量：", collection.count())


def organize_existing_folder(
    path: str,
    topics: List[str],
    move_to_topic_dir: bool = True,
):
    """
    批量整理：扫描指定目录下所有 PDF，自动判别主题并归档到 data/papers/<topic>/
    同时把每篇论文也加入向量库（便于检索）。
    """
    add_and_classify_papers(
        path=path,
        topics=topics,
        move_to_topic_dir=move_to_topic_dir,
    )


def search_papers(query: str, top_k: int = 5, topic: Optional[str] = None):
    collection = get_paper_collection()
    if collection.count() == 0:
        print("向量库为空，请先添加论文。")
        return

    q_emb = embed_text(query, TEXT_EMBEDDING_MODEL)

    where = {"topic": topic} if topic else None
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where,
    )

    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    print(f"查询：{query}")
    if topic:
        print(f"过滤主题：{topic}")
    print("最相关的论文：")
    for i, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
        print(f"{i}. {meta.get('path')}  topic={meta.get('topic')}  距离: {dist:.4f}")