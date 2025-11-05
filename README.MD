# PoC — Asesor legal con RAG sobre Excel de sentencias (ChromaDB)

> **Objetivo.** Responder en lenguaje coloquial preguntas sobre demandas relacionadas con **redes sociales** a partir de un Excel con sentencias, usando **búsqueda semántica** con embeddings y **ChromaDB**.

---

## 🧾 Entregables

- **Código funcional** (scripts en `src/`).
- **Informe** con los títulos solicitados (incluido más abajo).
- **Guía de ejecución** paso a paso.

---

## 🚀 Quickstart (3 comandos)

```bash
pip install -r requirements.txt
python -m src.ingest_index
python -m src.qa --q "acoso escolar" --debug
```

<details>
<summary>Usar venv (opcional)</summary>

```bash
# crear y activar entorno
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```
</details>

---

## 🗂️ Estructura del repositorio

```
.
├─ .chroma/                     # índice persistente (se genera al indexar)
├─ Caso/
│  └─ Caso de consultoria 2.pdf
├─ Datos/
│  └─ sentencias_pasadas.xlsx
├─ src/
│  ├─ config.yaml               # configuración (ruta, modelo, sinónimos, umbrales)
│  ├─ ingest_index.py           # indexa el Excel en ChromaDB
│  ├─ debug_chroma.py           # verifica colección/contador/peek
│  ├─ qa.py                     # consulta semántica + armado de respuesta
│  └─ utils.py                  # helpers: load_config, limpieza, sinónimos
├─ README.md
└─ requirements.txt
```

---

## ⚙️ Configuración 


```yaml
data_path: "Datos/sentencias_pasadas.xlsx"
id_col: "ID"                   # o "Radicado", etc.
text_fields:
  - "Tema - subtema"
  - "Hechos"
  - "síntesis"                 # si es "sintesis" en tu archivo, cámbialo
  - "Sentencia"

embedding_model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

chroma:
  persist_dir: ".chroma"
  collection: "sentencias_lex"
  metric: "cosine"

retrieval:
  k: 8
  min_hits: 1
  distance_max: 0.35          # menor = más estricto

synonyms:
  "acoso escolar": ["bullying", "matoneo", "hostigamiento escolar", "violencia escolar"]
  "piar": ["plan individual de ajustes razonables", "ajustes razonables", "inclusión educativa", "discapacidad", "NEE"]
  "redes sociales": ["facebook", "instagram", "tiktok", "x", "twitter", "red social"]
```

---

## ▶️ Cómo ejecutar

```bash
# 1) Instalar dependencias
pip install -r requirements.txt

# 2) Indexar el Excel en ChromaDB
python -m src.ingest_index

# 3) Verificar la colección
python -m src.debug_chroma

# 4) Consultas (QA)
python -m src.qa --q "acoso escolar" --debug
python -m src.qa --q "PIAR" --debug
python -m src.qa --q "redes sociales" --debug
```

> Tip: si recuperas pocos resultados, sube `retrieval.distance_max` a `0.40` y/o `retrieval.k` a `10–12`.

---

## 🧠 Cómo funciona (resumen técnico)

- **Indexación**: se concatenan los campos listados en `text_fields` y se guardan como documento + metadatos (`m_<campo>`).
- **Embeddings**: `paraphrase-multilingual-mpnet-base-v2` (robusto en español).
- **Vector DB**: ChromaDB persistente (HNSW, métrica `cosine`).
- **Consulta**: expansión de sinónimos, top-k y filtrado por umbral `distance_max`.
- **Respuesta**: plantilla en lenguaje simple con bullets (usa metadatos como `m_Tema - subtema` y `m_síntesis`) + lista de **Fuentes** con ID y distancia.

---

## 📊 Informe (con resultados reales pegados)

### 1) Explicación del caso
PoC para responder en lenguaje coloquial preguntas sobre **demandas y sentencias asociadas a redes sociales** a partir de un **Excel**. La salida incluye **evidencia trazable** (IDs y distancias) para validación rápida.

### 2) Supuestos
- El Excel incluye un identificador (`id_col`) y campos de texto relevantes (Tema/Hechos/Síntesis/Sentencia).
- La respuesta se construye a partir de los documentos recuperados (RAG “ligero”, sin LLM remoto).
- Idioma principal: español con variaciones; se usan sinónimos.

### 3) Formas para resolver el caso y opción tomada
- **Alternativas**: (1) TF-IDF/regex; (2) **Embeddings + ChromaDB** ✅; (3) RAG con re-ranking + LLM; (4) Clasificación/finetuning.
- **Elegida**: (2) por equilibrio entre precisión, reproducibilidad y evidencia trazable.

### 4) Resultados del análisis de los datos y los modelos

**Acoso escolar — sentencia y detalle**  
_(comando: `python -m src.qa --q "acoso escolar" --debug`)_

```text
=== RESPUESTA ===
Sobre “acoso escolar”, encontré 2 referencias claras en el archivo.

- Caso T-168/22: ACCION DE TUTELA CONTRA INSTITUCION EDUCATIVA-Procedencia ACOSO ESCOLAR O BULLYING-Manejo por parte de las instituciones — La accionante, actuando en representación de una hija menor de edad, alega que la institución educativa accionada vulneró derechos fundament…
- Caso T-252/23: ACCION DE TUTELA PARA RECLAMAR INDEMNIZACION DE PERJUICIOS-Procedencia excepcional ACOSO ESCOLAR O BULLYING-Manejo por p — La accionante, actuando en representación de su hijo menor de edad, cuestionó a la accionada y formuló pretensiones que se dividieron en dos…

Fuentes:
• ID T-168/22: dist=0.2691
• ID T-252/23: dist=0.3387
```

**PIAR — casos y sentencias**  
_(comando: `python -m src.qa --q "PIAR" --debug`)_

```text
=== RESPUESTA ===
Sobre “PIAR”, encontré 1 referencias claras en el archivo.

- Caso T-085/23: ACCESIBILIDAD Y ADAPTABILIDAD COMO COMPONENTES ESENCIALES DEL DERECHO A LA EDUCACION CARENCIA ACTUAL DE OBJETO POR HECHO — La accionante actúa en representación de un hijo menor de edad que fue diagnosticado con el síndrome Landau Kleffner, el cual le genera difi…

Fuentes:
• ID T-085/23: dist=0.2873
```

**Redes sociales — ejemplos y sentencias**  
_(comando: `python -m src.qa --q "redes sociales" --debug`)_

```text
=== RESPUESTA ===
Sobre “redes sociales”, encontré 40 referencias posibles en el archivo. (evidencia débil).

- Caso T-031/20: ACCION DE TUTELA CONTRA PARTICULARES-No se requiere solicitud previa de rectificación de publicación en Facebook DERECHO — La problemática planteada en este asunto tuvo origen en un conflicto de índole personal suscitado entre dos servidores públicos que compartí…
- Caso T-610/19: CARENCIA ACTUAL DE OBJETO POR HECHO SUPERADO-El video subido a la red social fue eliminado de la plataforma DERECHO A LA — La accionante, actuando en nombre de su hija menor de edad, aduce que varios medios de comunicación vulneraron los derechos a la honra, al b…
- Caso T-155/19: DERECHO A LA INTIMIDAD-Alcance y contenido DERECHOS A LA INTIMIDAD, BUEN NOMBRE Y HONRA FRENTE A LIBERTAD DE EXPRESION Y — En el presente caso, se atribuye la vulneración de los derechos fundamentales a una publicación realizada por la accionada en la red social …

Fuentes:
• ID T-031/20: dist=0.4207
• ID T-610/19: dist=0.4319
• ID T-155/19: dist=0.4694
```

> **Nota:** en “redes sociales” las distancias son superiores al umbral por defecto (`0.35`), por eso el motor las marca como **evidencia débil**. Para reforzar, puedes subir `retrieval.k` y/o aumentar `distance_max` a `0.40`.

### 5) Futuros ajustes o mejoras
- **Re-ranking** con cross-encoder para ordenar mejor los hits.
- **LLM generativo** para redacciones más ricas y verificación de consistencia.
- **Guardrails**: si no hay evidencia < umbral, devolver “no se encontró evidencia suficiente”.
- **UI** (Gradio/Streamlit) y **API REST** (FastAPI).
- **Despliegue** en Azure/AWS (contenedor + volumen para `.chroma`).
- **Mejoras de datos**: normalización de encabezados/acentos y enriquecimiento con etiquetas.

### 6) Apreciaciones y comentarios (opcional)
La PoC demuestra que, **sin LLM**, ya se logra una búsqueda útil y **explicable** (IDs + distancias), con bajo costo y fácil reproducción.

---

## 🧰 Troubleshooting

- **No encuentra “síntesis”** → usa exactamente el nombre real (p. ej., `sintesis`) en `text_fields`.
- **Pocos resultados** → sube `retrieval.k` (10–12) y/o `retrieval.distance_max` (0.40).
- **IDs duplicados** → el indexador añade sufijos `__<row>` para evitar choques.
- **Cambiar modelo** → ajusta `embedding_model` en `config.yaml` y re-indexa.

---

## 🗃️ Comandos útiles

```bash
# Re-crear índice desde cero
if [ -d ".chroma" ]; then rm -rf .chroma; fi
python -m src.ingest_index

# Verificar colección
python -m src.debug_chroma

# Consultas de ejemplo
python -m src.qa --q "acoso escolar" --debug
python -m src.qa --q "PIAR" --debug
python -m src.qa --q "redes sociales" --debug
```

---

## 📌 Licencia
Uso interno para evaluación técnica.
