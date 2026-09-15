# RAG setup

Install:

```bash
pip install sentence-transformers
```

Build index:

```bash
python src/rag/build_index.py
```

Test retrieval:

```bash
python src/rag/retrieve.py "Why can an HP-OFF fault be missed by the frozen threshold?"
```

This phase tests retrieval only. The next phase sends the retrieved context plus ML/XAI outputs to an LLM.
