"""Idea Forge file intake: text extraction from uploaded documents (PDF, DOCX, HTML, notebooks, code, text)."""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..gen.ideaparse import extract_document_text, parse_input

router = APIRouter(prefix="/api/forge")
MAX_BYTES = 25 * 1024 * 1024


class ExtractIn(BaseModel):
    name: str
    data_b64: str


@router.post("/extract")
def extract(body: ExtractIn) -> dict:
    try:
        data = base64.b64decode(body.data_b64, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "The file could not be decoded.") from None
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 25 MB.")
    text = extract_document_text(body.name, data)
    if not text.strip():
        raise HTTPException(422, "No text could be extracted from this file (scanned PDFs need OCR first).")
    inp = parse_input(text)
    return {"text": text, "chars": len(text), "format": inp.format, "formats": inp.formats,
            "key_sentences": inp.key_sentences, "truncated": len(text) >= 400_000}


__all__ = ["router"]
