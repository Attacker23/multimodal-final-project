from __future__ import annotations

import hashlib
import os
from typing import List

import chromadb
from chromadb.config import Settings

from .embeddings import embed_text, embed_images
from config import (
    VECTOR_DB_DIR,
    IMAGE_COLLECTION_NAME,
    IMAGE_DIR,
    IMAGE_EMBEDDING_MODEL,
    IMAGE_TEXT_EMBEDDING_MODEL,
)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def get_image_collection():
    client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(allow_reset=True),
    )
    # 关键：用 cosine 距离空间（配合 normalize_embeddings=True）
    return client.get_or_create_collection(
        IMAGE_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _list_images(path: str) -> List[str]:
    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in EXTS:
        return [path]
    imgs = []
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                imgs.append(os.path.join(root, f))
    # 稳定排序，便于复现
    imgs.sort()
    return imgs


def _stable_id(path: str) -> str:
    # 同一路径固定 id：避免重复索引导致库越来越乱
    ap = os.path.abspath(path)
    return hashlib.sha1(ap.encode("utf-8")).hexdigest()


def index_images(path: str):
    os.makedirs(IMAGE_DIR, exist_ok=True)

    img_paths = _list_images(path)
    if not img_paths:
        print("未找到任何图片文件。")
        return

    collection = get_image_collection()

    print(f"共发现 {len(img_paths)} 张图片，开始建立索引...")
    embs = embed_images(img_paths, IMAGE_EMBEDDING_MODEL)

    ids = [_stable_id(p) for p in img_paths]
    metadatas = [{"path": os.path.abspath(p)} for p in img_paths]

    # 用 upsert：同路径重复跑不会增加重复条目
    collection.upsert(
        ids=ids,
        embeddings=embs,
        metadatas=metadatas,
        documents=None,
    )

    # 简单自检
    if embs:
        print("embedding dim:", len(embs[0]))
        print("first 5 vals:", embs[0][:5])

    print("图片索引完成。当前图片数量：", collection.count())


def search_images(query: str, top_k: int = 5):
    collection = get_image_collection()
    if collection.count() == 0:
        print("图片向量库为空，请先索引图片。")
        return

    # 以文搜图：用同一个 CLIP 的文本 embedding（512 维）
    q_emb = embed_text(query, IMAGE_TEXT_EMBEDDING_MODEL)

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    print(f"文本查询：{query}")
    print("最相关的图片：")
    for i, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
        # cosine 距离：越小越相似（0 最好）
        print(f"{i}. {meta.get('path')}  距离: {dist:.4f}")