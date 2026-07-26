# utils.py
#
# Lightweight, serverless-friendly helpers for the Multimodal Financial RAG API.
#
# Design notes (important for Vercel deployment):
# - No torch / transformers / local model weights. Vision, text, and speech
#   all go through the Groq Cloud API, which returns in a fraction of a
#   second and keeps the deployed function small enough to fit Vercel's
#   serverless size limits.
# - Every function here is pure / stateless: given bytes in, text out. No
#   files are written to disk anywhere in this module.
# - Model names are read from environment variables with sensible defaults
#   so you can swap models the moment Groq ships something newer, without a
#   redeploy of code (only an env var change).

import os
import base64
from functools import lru_cache
from typing import List, Tuple

import fitz  # PyMuPDF
from groq import Groq

# ---------------------------------------------------------------------------
# Groq client + model configuration
# ---------------------------------------------------------------------------

VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """Lazily create a single Groq client per warm serverless instance."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it in your Vercel project's "
            "Environment Variables settings."
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Vision (image / chart / table description)
# ---------------------------------------------------------------------------

def _image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def describe_image(
    image_bytes: bytes,
    prompt: str = (
        "Describe what you see in this image in detail, including any "
        "visible text, numbers, labels, or data."
    ),
) -> str:
    """Send an image straight from memory to Groq's vision model."""
    client = get_groq_client()
    data_url = _image_bytes_to_data_url(image_bytes)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def is_graph(description: str) -> bool:
    """Heuristic: does this image description look like a chart/graph/table?"""
    keywords = ("graph", "chart", "plot", "table", "diagram", "axis", "trend")
    lowered = description.lower()
    return any(k in lowered for k in keywords)


def describe_chart_or_table(image_bytes: bytes) -> str:
    """Ask specifically for the underlying data of a financial chart/table."""
    prompt = (
        "This image is a financial chart, graph, or table. Transcribe the "
        "underlying data as precisely as possible: axis labels, series "
        "names, approximate values, trends, and any notable figures. Write "
        "it as plain text suitable for a search index, not prose."
    )
    return describe_image(image_bytes, prompt=prompt)


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """Transcribe an earnings-call recording or voice memo via Groq Whisper."""
    client = get_groq_client()
    transcription = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=(filename, audio_bytes),
        response_format="text",
    )
    if isinstance(transcription, str):
        return transcription
    return getattr(transcription, "text", str(transcription))


# ---------------------------------------------------------------------------
# LLM chat completion (RAG answer generation)
# ---------------------------------------------------------------------------

def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Text chunking (replaces LangChain's RecursiveCharacterTextSplitter so we
# don't have to pull in the full LangChain dependency tree)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 50) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------------------------------------------------------
# PDF layout helpers (pure functions, no external services)
# ---------------------------------------------------------------------------

def extract_text_around_item(
    text_blocks, bbox, page_height, threshold_percentage: float = 0.1
) -> Tuple[str, str]:
    """Find the text block immediately above/below a bounding box (e.g. a
    table or image) so we can use it as a caption hint."""
    before_text, after_text = "", ""
    vertical_threshold_distance = page_height * threshold_percentage
    horizontal_threshold_distance = bbox.width * threshold_percentage

    for block in text_blocks:
        block_bbox = fitz.Rect(block[:4])
        vertical_distance = min(
            abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1)
        )
        horizontal_overlap = max(
            0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0)
        )

        if (
            vertical_distance <= vertical_threshold_distance
            and horizontal_overlap >= -horizontal_threshold_distance
        ):
            if block_bbox.y1 < bbox.y0 and not before_text:
                before_text = block[4]
            elif block_bbox.y0 > bbox.y1 and not after_text:
                after_text = block[4]
                break

    return before_text, after_text


def process_text_blocks(text_blocks, char_count_threshold: int = 500):
    """Group adjacent text blocks into passages of roughly char_count_threshold
    characters, so short headings/paragraphs aren't indexed as isolated
    fragments."""
    current_group = []
    grouped_blocks = []
    current_char_count = 0

    for block in text_blocks:
        if block[-1] == 0:  # text-type block
            block_text = block[4]
            block_char_count = len(block_text)

            if current_char_count + block_char_count <= char_count_threshold:
                current_group.append(block)
                current_char_count += block_char_count
            else:
                if current_group:
                    grouped_content = "\n".join(b[4] for b in current_group)
                    grouped_blocks.append((current_group[0], grouped_content))
                current_group = [block]
                current_char_count = block_char_count

    if current_group:
        grouped_content = "\n".join(b[4] for b in current_group)
        grouped_blocks.append((current_group[0], grouped_content))

    return grouped_blocks
