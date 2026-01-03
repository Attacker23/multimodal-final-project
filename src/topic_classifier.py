from __future__ import annotations
from typing import Dict, List, Tuple
import math

from .embeddings import embed_text


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)


def classify_text_to_topic_by_descriptions(
    text: str,
    topic_labels: List[str],
    topic_descriptions: Dict[str, str],
    model_name: str,
    min_sim: float,
    default_topic: str = "Other",
) -> Tuple[str, float]:
    """
    输入短标签列表 topic_labels（如 CV/NLP/RL），
    实际用 topic_descriptions[label] 的描述去匹配。
    返回 (best_label, similarity)，低于阈值返回 default_topic。
    """
    labels = [t.strip() for t in topic_labels if t.strip()]
    labels = [t for t in labels if t in topic_descriptions]

    if not labels:
        return default_topic, 0.0

    prompts = []
    for lab in labels:
        desc = topic_descriptions.get(lab, lab)
        prompts.append(f"This paper is mainly about {desc}.")

    text_emb = embed_text(text, model_name)
    prompt_embs = embed_text(prompts, model_name)

    best_label, best_sim = default_topic, -1.0
    for lab, e in zip(labels, prompt_embs):
        s = _cosine(text_emb, e)
        if s > best_sim:
            best_sim = s
            best_label = lab

    if best_sim < min_sim:
        return default_topic, float(best_sim)
    return best_label, float(best_sim)