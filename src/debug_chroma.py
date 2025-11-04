# src/debug_chroma.py
import os, yaml
import chromadb
from chromadb.utils import embedding_functions

def main():
    cfg = yaml.safe_load(open("src/config.yaml","r",encoding="utf-8"))
    persist_dir = os.path.abspath(cfg["chroma"]["persist_dir"])
    coll_name   = cfg["chroma"]["collection"]
    model_name  = cfg.get("embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    client = chromadb.PersistentClient(path=persist_dir)
    print("persist_dir:", persist_dir)
    print("collections:", [c.name for c in client.list_collections()])

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    coll = client.get_collection(name=coll_name, embedding_function=ef)
    print("collection.name:", coll.name)
    print("count:", coll.count())
    peek = coll.peek(3)
    print("peek.ids:", (peek.get("ids") or [[]])[0])

if __name__ == "__main__":
    main()
