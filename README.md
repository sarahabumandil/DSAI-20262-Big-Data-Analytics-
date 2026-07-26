# Ledger — Multimodal Financial RAG (Vercel-ready)

A rewrite of the original Streamlit app as a stateless FastAPI backend +
static HTML/CSS/JS frontend, designed to deploy cleanly on Vercel.

## Architecture

```
project/
├── api/
│   ├── main.py                 FastAPI app (Vercel serverless entrypoint)
│   ├── document_processors.py  PDF/PPTX/image/audio/text extraction (in-memory)
│   └── utils.py                Groq calls (vision, whisper, chat) + text chunking
├── public/
│   └── index.html              Static frontend (fetch()-based, no build step)
├── vercel.json                 Routes /api/* to the Python function, everything
│                                else to the static /public folder
└── requirements.txt            Backend dependencies
```

Every request is stateless: the function reads the upload, calls Groq Cloud
for any vision/speech/text work, writes chunks straight into Weaviate Cloud,
and returns. Nothing is cached or written to local disk, so there's nothing
for Vercel's ephemeral, read-only filesystem or ~10–60s execution limit to
trip over.

## Required environment variables (set in Vercel → Project → Settings → Environment Variables)

| Variable            | Purpose                                              |
|---------------------|-------------------------------------------------------|
| `GROQ_API_KEY`       | Groq Cloud API key — powers text, vision, and speech |
| `WEAVIATE_URL`       | Your Weaviate Cloud (WCS) cluster REST endpoint       |
| `WEAVIATE_API_KEY`   | Your Weaviate Cloud API key                           |

Optional overrides (defaults shown):

| Variable              | Default                          |
|-----------------------|-----------------------------------|
| `GROQ_TEXT_MODEL`      | `llama-3.3-70b-versatile`        |
| `GROQ_VISION_MODEL`    | `llama-3.2-11b-vision-preview`   |
| `GROQ_WHISPER_MODEL`   | `whisper-large-v3`               |

Groq's model lineup changes — check https://console.groq.com/docs/models for
the current list and update these env vars if a model is renamed or retired.

## Weaviate setup

The backend creates its collection (`FinancialRagChunk`) automatically on
first upload, configured to use Weaviate Cloud's built-in embedding service
(`text2vec-weaviate`) — no separate embeddings API key required. If your
cluster doesn't have that module available, swap the `vectorizer_config` in
`api/main.py`'s `ensure_collection()` for `Configure.Vectorizer.text2vec_cohere()`
or `.text2vec_openai()` and supply the matching header/key.

## Deploying

```bash
npm i -g vercel      # if you don't already have it
vercel                # from the project root, follow the prompts
vercel env add GROQ_API_KEY
vercel env add WEAVIATE_URL
vercel env add WEAVIATE_API_KEY
vercel --prod
```

## What changed from the original Streamlit app, and why

- **No local ML models.** The original loaded `meta-llama/Llama-3.2-11B-Vision`
  and a HuggingFace embedding model directly via `torch`/`transformers`. That
  alone is several GB and would blow past Vercel's serverless function size
  limit, plus there's no GPU on serverless. Vision now goes through Groq's
  hosted vision model; embeddings happen inside Weaviate Cloud.
- **No local Weaviate / no `localhost:8080`.** Points at Weaviate Cloud
  instead, so the vector index survives across serverless invocations.
- **No disk writes.** The original saved extracted table/image files to
  `vectorstore/...` on disk (via `open(..., "wb")`, `pandas.to_excel`,
  `pixmap.save`). Vercel's filesystem is read-only outside `/tmp`, so all of
  that now stays in memory and is described/summarized inline instead of
  saved as separate files.
- **PPTX only, no LibreOffice.** The original converted `.ppt`/`.pptx` to PDF
  via a local LibreOffice binary to render slide images. There's no
  LibreOffice on Vercel, so slides are now read directly with `python-pptx`
  (text + speaker notes). Legacy binary `.ppt` isn't supported — export to
  `.pptx` first.
- **No NVIDIA Deplot dependency.** Chart/graph description is handled by the
  same Groq vision call, prompted specifically to transcribe the underlying
  data, removing an extra third-party API dependency.
- **Session-scoped, not global.** Streamlit's `st.session_state` doesn't
  exist in a stateless function, so every upload/query is tagged with a
  `session_id` (generated client-side, kept in `localStorage`) and filtered
  in Weaviate, so concurrent users never see each other's documents.
