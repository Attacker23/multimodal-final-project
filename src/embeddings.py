from __future__ import annotations

from typing import List, Union

import torch
from sentence_transformers import SentenceTransformer

_text_models = {}
_image_models = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_text_model(model_name: str) -> SentenceTransformer:
    if model_name not in _text_models:
        _text_models[model_name] = SentenceTransformer(model_name, device=DEVICE)
    return _text_models[model_name]


def get_image_model(model_name: str) -> SentenceTransformer:
    if model_name not in _image_models:
        _image_models[model_name] = SentenceTransformer(model_name, device=DEVICE)
    return _image_models[model_name]


def embed_text(texts: Union[str, List[str]], model_name: str) -> List[float] | List[List[float]]:
    """
    Return:
      - if texts is str: List[float]
      - if texts is List[str]: List[List[float]]
    """
    model = get_text_model(model_name)
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 关键：L2 normalize -> cosine 更稳定
    ).tolist()


def embed_images(image_paths: Union[str, List[str]], model_name: str) -> List[List[float]]:
    model = get_image_model(model_name)
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    return model.encode(
        image_paths,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 关键
    ).tolist()