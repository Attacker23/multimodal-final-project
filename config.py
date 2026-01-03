import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PAPER_DIR = os.path.join(DATA_DIR, "papers")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# 向量数据库存放位置
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

# Chroma collection 名
PAPER_COLLECTION_NAME = "papers"
IMAGE_COLLECTION_NAME = "images"

# 论文：文本嵌入模型（384 维）
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 图片：图像嵌入模型（512 维）
IMAGE_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-L-14"

# 以文搜图：文本查询也必须用同一个 CLIP（512 维）
IMAGE_TEXT_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-L-14"

# 主题归档配置
DEFAULT_TOPIC = "Other"
TOPIC_MIN_SIM = 0.20  # 可调

# 短标签 -> 描述（用于匹配，不影响目录名）
TOPIC_DESCRIPTIONS = {
    "CV": "computer vision, image classification, object detection, segmentation, OCR, tracking, SLAM, 3D vision, diffusion, ViT, CLIP",
    "NLP": "natural language processing, language model, LLM, transformer, attention, pretraining, finetuning, instruction tuning, RAG, information extraction, text classification",
    "RL": "reinforcement learning, agent, policy gradient, Q-learning, actor-critic, offline RL, exploration, reward, MDP, control, robotics",
}