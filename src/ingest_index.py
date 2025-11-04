# src/ingest_index.py
import os, sys
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from src.utils import load_config, clean_txt

def main():
    cfg = load_config()
    xlsx_path   = cfg["data_path"]
    id_col      = cfg["id_col"]
    text_fields = cfg["text_fields"]
    model_name  = cfg.get("embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    persist_dir = os.path.abspath(cfg["chroma"]["persist_dir"])
    coll_name   = cfg["chroma"]["collection"]
    space       = cfg["chroma"].get("metric", "cosine")

    # 0) Lectura & armado del documento
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"[ERR] No pude leer {xlsx_path}: {e}")
        sys.exit(1)

    for c in text_fields:
        if c not in df.columns:
            print(f"[WARN] Columna '{c}' no existe. La creo vacía.")
            df[c] = ""
        df[c] = df[c].map(clean_txt)

    if id_col not in df.columns:
        print(f"[WARN] id_col '{id_col}' no existe. Uso índice como ID.")
        df[id_col] = df.index.astype(str)
    df[id_col] = df[id_col].astype(str)
    if df[id_col].duplicated().any():
        df[id_col] = df[id_col] + "__" + df.index.astype(str)

    df["doc"] = df[text_fields].agg(". ".join, axis=1).str.strip()
    df = df[df["doc"].astype(bool)].reset_index(drop=True)
    if df.empty:
        print("[ERR] Ninguna fila tiene contenido en text_fields. Revisa 'text_fields' en config.yaml.")
        print("Columnas reales:", list(df.columns))
        sys.exit(1)

    # 1) PersistentClient (garantiza persistencia)
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    coll = client.get_or_create_collection(name=coll_name, metadata={"hnsw:space": space}, embedding_function=ef)

    # 2) Limpiar colección y cargar
    try:
        existing = coll.count()
        if existing:
            coll.delete(where={})
            print(f"[INFO] Limpiada colección '{coll_name}' (borrados {existing} docs).")
    except Exception:
        pass

    ids   = df[id_col].tolist()
    docs  = df["doc"].tolist()
    metas = []
    for _, row in df.iterrows():
        m = {"id_caso": row[id_col]}
        for c in text_fields:
            m[f"m_{c}"] = row.get(c, "")
        metas.append(m)

    coll.add(ids=ids, documents=docs, metadatas=metas)
    # Con PersistentClient no hace falta client.persist(), pero no estorba:
    try:
        client.persist()
    except Exception:
        pass

    cnt = coll.count()
    print(f"[OK] Indexados {cnt} documentos en '{coll_name}' (dir: {persist_dir}).")
    peek = coll.peek(3)
    ids_preview = (peek.get("ids") or [[]])[0]
    print("[OK] Peek ids:", ids_preview[0] if ids_preview else "—")

if __name__ == "__main__":
    main()
