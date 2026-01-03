import chromadb
from chromadb.config import Settings
from config import VECTOR_DB_DIR, IMAGE_COLLECTION_NAME

client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR,
    settings=Settings(allow_reset=True),
)

try:
    client.delete_collection(IMAGE_COLLECTION_NAME)
    print(f"Deleted collection: {IMAGE_COLLECTION_NAME}")
except Exception as e:
    print("Delete failed (maybe not exist):", e)