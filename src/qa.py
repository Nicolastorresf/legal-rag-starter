# src/qa.py
from src.utils import load_config, expand_query
import argparse, os
import chromadb
from chromadb.utils import embedding_functions

def get_collection(cfg, allow_create=True, debug=False):
    persist_dir = os.path.abspath(cfg["chroma"]["pointless" if False else "persist_dir"])
    client = chromadb.PersistentClient(path=persist_dir)
    model_name = cfg.get("embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    name = cfg["chroma"]["collection"]
    meta = {"hnsw:space": cfg["chroma"].get("metric", "cosine")}
    coll = client.get_or_create_collection(name=name, metadata=meta, embedding_function=ef) if allow_create \
           else client.get_collection(name=name,  embedding_function=ef)
    if debug:
        print(f"[DEBUG] persist_dir={persist_dir} collections={[c.name for c in client.list_collections()]} count={coll.count()}")
    return coll

def retrieve(cfg, query: str, debug=False):
    coll = get_collection(cfg, allow_create=True, debug=debug)
    out = coll.query(
        query_texts=[query],
        n_results=int(cfg["retrieval"].get("k", 5)),
        include=["metadatas","documents","distances"]
    )
    docs  = (out.get("documents") or [[]])[0]
    metas = (out.get("metadatas") or [[]])[0]
    dists = (out.get("distances") or [[]])[0]
    ids   = (out.get("ids") or [[]])[0]
    if debug:
        print(f"[DEBUG] raw_hits={len(docs)}")
    if not docs:
        return []
    n = min(len(docs), len(metas), len(dists))
    hits = []
    for i in range(n):
        hid = ids[i] if i < len(ids) else f"hit_{i}"
        hits.append({"id": hid, "doc": docs[i], "distance": dists[i], **(metas[i] or {})})
    return hits

def answer_template(query: str, hits: list, cfg: dict) -> dict:
    thresh   = float(cfg["retrieval"].get("distance_max", 0.70))
    min_hits = int(cfg["retrieval"].get("min_hits", 1))
    filtered = [h for h in hits if h["distance"] <= thresh]
    use = filtered[:3] if len(filtered) >= min_hits else hits[:3]
    if not use:
        return {"answer": "No hay evidencia suficiente en el archivo para responder esa pregunta.", "fuentes": []}
    weak = len(filtered) < min_hits
    intro = f"Sobre “{query}”, encontré {len(filtered) if not weak else len(hits)} referencias {'claras' if not weak else 'posibles'} en el archivo."
    bullets = []
    for h in use:
        tema = h.get("m_Tema - subtema") or ""
        sint = h.get("m_síntesis") or h.get("m_sintesis") or ""
        if len(sint) > 140: sint = sint[:140] + "…"
        bullets.append(f"- Caso {h['id']}: {tema[:120]} — {sint}".rstrip())
    if (weak): intro += " (evidencia débil)."
    return {"answer": intro + "\n\n" + "\n".join(bullets),
            "fuentes": [{"id_caso": h["id"], "distance": round(h["distance"],4)} for h in use]}

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True, help="Consulta, ej: 'acoso escolar' o 'PIAR'")
    p.add_argument("--debug", action="store_true", help="Imprime info de depuración")  # <-- fix aquí
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config()
    queries = [args.q] + expand_query(args.q, cfg.get("synonyms", {}))
    pool = {}
    for q in queries:
        hits = retrieve(cfg, q, debug=args.debug)
        for h in hits:
            keep = pool.get(h["id"])
            if (not keep) or (h["distance"] < keep["distance"]):
                pool[h["id"]] = h
    ranked = sorted(pool.values(), key=lambda x: x["distance"])
    if args.debug:
        print(f"[DEBUG] k={cfg['retrieval'].get('k')} threshold={cfg['retrieval'].get('distance_max')}")
        for h in ranked[:10]:
            tema = (h.get("m_Tema - subtema") or "")[:80]
            print(f"  - ID {h['id']} dist={h['distance']:.4f} tema={tema}")
    out = export_answer(args.q, ranked, cfg)
    print("\n=== RESPUESTA ===")
    print(out["answer"])
    if out["fuentes"]:
        print("\nFuentes:")
        for f in out["fuentes"]:
            print(f"• ID {f['id_caso']}: dist={f['distance']}")

def export_answer(q, ranked, cfg):
    return answer_template(q, ranked, cfg)

if __name__ == "__main__":
    main()
