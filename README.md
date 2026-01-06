# 本地 AI 文献与图像管理助手

一个本地可跑的最小多模态管理工具，支持：
- **论文 PDF 管理**：添加/索引论文、自动归类归档、语义搜索论文
- **图片管理**：批量索引图片、**以文搜图**（Text-to-Image Retrieval）

项目以命令行方式提供统一入口 `main.py`，满足“一键调用”要求。

---

## 1. 项目结构

```
.
├─ data/
│  ├─ images/          # 图片归档目录
│  ├─ papers/          # 论文归档目录
│  └─ vector_db/       # ChromaDB 持久化目录
├─ scripts/
│  └─ reset_image_collection.py   # 重置图片向量库（重建 collection 用）
├─ src/
│  ├─ embeddings.py    # 文本/图片 embedding
│  ├─ image_manager.py # 图片索引与检索
│  ├─ paper_manager.py # 论文索引、分类、检索
│  ├─ pdf_utils.py
│  └─ topic_classifier.py
├─ config.py           # 项目配置（目录、模型、collection 名称等）
├─ download.py         # 可选：数据集下载脚本（如 Flickr8k）
├─ main.py             # 统一命令行入口
└─ requirements.txt
```

---

## 2. 核心功能

### 2.1 论文 PDF
- 添加/索引论文：从 PDF 提取文本，生成向量并写入向量库
- 自动归类：按主题（topics）对论文归档到对应文件夹
- 语义搜索：输入自然语言 query，返回最相关论文

### 2.2 图片（以文搜图）
- 批量索引本地图片文件夹（jpg/png/webp 等）
- 输入自然语言描述，检索返回最相关图片路径（Top-K）

---

## 3. 环境配置与依赖安装

### 3.1 Python 环境
建议 Python 3.10+（本项目在 Python 3.11 环境开发）。

### 3.2 安装依赖
```bash
pip install -r requirements.txt
```

如遇到 CLIP 模型加载的兼容性问题（huggingface_hub repo id 校验报错），建议升级三件套：
```bash
pip install -U sentence-transformers transformers huggingface_hub
```

---

## 4. 配置说明（config.py）

在 `config.py` 中配置以下内容：
- `VECTOR_DB_DIR`：ChromaDB 持久化目录（默认 `data/vector_db`）
- `IMAGE_DIR`、`PAPER_DIR`：归档目录
- `IMAGE_COLLECTION_NAME`：图片 collection 名
- `IMAGE_EMBEDDING_MODEL`：图片 embedding 模型（CLIP）
- `IMAGE_TEXT_EMBEDDING_MODEL`：文本查询 embedding 模型（应与图片同一 CLIP 家族）
-（论文部分同理：论文 collection / 模型等）

> 说明：以文搜图必须保证“图片向量”和“文本向量”处于同一语义空间，通常使用同一个 CLIP 模型名即可。

---

## 5. 使用说明（命令行一键调用）

所有功能从项目根目录执行：

### 5.1 添加/索引论文（自动归类并归档）
```bash
python main.py add_paper <path> --topics "Topic1,Topic2" [--copy]
```

示例：
```bash
python main.py add_paper "D:\data\papers" --topics "CV,NLP"
python main.py add_paper "D:\data\papers\a.pdf" --topics "CV,NLP" --copy
```

参数说明：
- `<path>`：PDF 文件或包含 PDF 的文件夹
- `--topics`：主题列表，用逗号分隔
- `--copy`：默认移动文件到归档目录；加 `--copy` 则复制不移动

---

### 5.2 一键整理现有混乱论文文件夹（自动归类归档）
```bash
python main.py organize_papers <path> --topics "Topic1,Topic2" [--copy]
```

示例：
```bash
python main.py organize_papers "D:\data\messy_papers" --topics "CV,NLP"
```

---

### 5.3 语义搜索论文
```bash
python main.py search_paper "<query>" [--top_k 5] [--topic "<TopicName>"]
```

示例：
```bash
python main.py search_paper "diffusion model for image editing" --top_k 10
python main.py search_paper "object detection" --topic "CV"
```

---

### 5.4 索引图片文件夹
```bash
python main.py index_image <path>
```

示例：
```bash
python main.py index_image "D:\data\images"
```

---

### 5.5 以文搜图
```bash
python main.py search_image "<query>" [--top_k 5]
```

示例：
```bash
python main.py search_image "a dog runs through the grass" --top_k 5
python main.py search_image "a child is playing in the water" --top_k 5
```

---

## 6. 向量库重置（图片库重建）

当你修改了图片 collection 的距离空间（如 cosine）或更换模型后，建议重建图片向量库：

```bash
python scripts/reset_image_collection.py
python main.py index_image "D:\data\images"
```

---

## 7. 技术选型说明

- **向量库**：ChromaDB（本地持久化），目录：`data/vector_db`；两个集合：`papers`、`images`
- **论文语义检索模型**：`sentence-transformers/all-MiniLM-L6-v2`（384 维），用于 PDF 文本向量化与检索
- **以文搜图模型**：`sentence-transformers/clip-ViT-L-14`（512 维）
  - 图片 embedding 用它
  - 文本查询 embedding 也用同一个 CLIP，保证图文同一向量空间可比
- 设备：自动选择 **CUDA**（若可用）否则使用 CPU

> 论文检索：将 PDF 文本编码为向量后入库；搜索时对 query 编码后进行 Top-K 近邻检索。  
> 图片检索：对图片生成 CLIP image embedding 入库；对文本 query 生成 CLIP text embedding 检索返回最相近图片。

- **自动归类策略**：用 `TOPIC_DESCRIPTIONS` 做主题匹配，相似度阈值 `TOPIC_MIN_SIM=0.20`；低于阈值归入 `Other`


