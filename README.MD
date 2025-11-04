# Informe — PoC Legal RAG (Open Source)

**Fecha:** {{hoy}}  
**Vector DB:** Chroma (open-source) — persistente local.  
**Alcance:** MVP en 6 horas (RAG básico con guardrails).

## 1. Caso y objetivo
(Resumen del problema y dato origen: Excel)

## 2. Supuestos
- Columnas usadas: {{id_col}} | {{text_fields}}
- Política: responder SOLO con evidencia; si no hay, decirlo.

## 3. Método
Ingesta/limpieza → Indexación Chroma → Recuperación top-k (k={{k}}, distance_max={{distance_max}}) → Respuesta coloquial con citas (IDs de caso).

## 4. Resultados
- Consultas probadas: “acoso escolar”, “PIAR”, …
- Ejemplos de salida (pegar respuesta + IDs citados).
- % “sin evidencia”: {{x%}}

## 5. Limitaciones y próximos pasos
Híbrido con BM25 / reranker, verificador de evidencias, UI y Docker.
