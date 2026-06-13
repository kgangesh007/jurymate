import os
import io
import base64
import fitz                        # PyMuPDF
from pptx import Presentation
from pptx.util import Inches
from PIL import Image, ImageDraw
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import ollama
from core.database import (
    register_document, update_document_status, add_team
)

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma")
COLLECTION_NAME = "jurymate"

# ── ChromaDB setup ─────────────────────────────────────────
_chroma_client = None
_collection     = None

def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

# ── Vision helpers ─────────────────────────────────────────
def _is_llava_available() -> bool:
    """Check if LLaVA model is available in Ollama"""
    try:
        models = ollama.list()
        return any(
            "llava" in m["name"].lower()
            for m in models.get("models", [])
        )
    except Exception:
        return False

def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def _describe_image_with_llava(img: Image.Image, slide_num: int) -> str:
    """Send slide image to LLaVA and get full description"""
    try:
        img_b64 = _image_to_base64(img)
        resp = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": f"""This is slide {slide_num} from a hackathon project presentation.
Please describe everything visible in detail including:
- All text visible on the slide
- Architecture diagrams and their components
- Technology names, tools, frameworks mentioned
- Flow diagrams and what they represent
- Any metrics, accuracy numbers, or results shown
- Any dataset names or model names visible
Be thorough and specific. Extract maximum information.""",
                "images": [img_b64]
            }]
        )
        return resp["message"]["content"]
    except Exception as e:
        return f"[Vision analysis unavailable: {str(e)}]"

def _slide_to_pil_image(slide, width=1280, height=720) -> Image.Image:
    """
    Convert a PPTX slide to PIL Image.
    Renders text content visually so LLaVA can read it.
    """
    img  = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y_pos = 30
    for shape in slide.shapes:
        # Extract text from text frames
        if hasattr(shape, "text_frame"):
            for para in shape.text_frame.paragraphs:
                text = para.text.strip()
                if text:
                    # Wrap long text
                    words  = text.split()
                    line   = ""
                    for word in words:
                        if len(line + word) < 90:
                            line += word + " "
                        else:
                            draw.text((30, y_pos), line.strip(),
                                      fill=(0, 0, 0))
                            y_pos += 22
                            line   = word + " "
                    if line.strip():
                        draw.text((30, y_pos), line.strip(),
                                  fill=(0, 0, 0))
                        y_pos += 22
                    y_pos += 8

        # Extract images from picture shapes
        if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
            try:
                img_bytes = shape.image.blob
                pic       = Image.open(io.BytesIO(img_bytes))
                # Resize to fit remaining space
                max_w = width - 60
                max_h = max(100, height - y_pos - 30)
                pic.thumbnail((max_w, max_h), Image.LANCZOS)
                img.paste(pic, (30, y_pos))
                y_pos += pic.height + 15
            except Exception:
                pass

        if y_pos > height - 30:
            break

    return img

def _slide_has_images(slide) -> bool:
    """Check if slide contains picture shapes"""
    return any(shape.shape_type == 13 for shape in slide.shapes)

def _slide_text_is_thin(slide, threshold=80) -> bool:
    """Check if slide has very little extractable text"""
    text = " ".join(
        shape.text for shape in slide.shapes
        if hasattr(shape, "text")
    ).strip()
    return len(text) < threshold

# ── Parsers ────────────────────────────────────────────────
def parse_pdf(file_bytes: bytes) -> tuple[list[str], int]:
    """Extract text from PDF pages"""
    doc   = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return pages, len(pages)

def parse_pptx(file_bytes: bytes,
               use_vision: bool = True) -> tuple[list[str], int]:
    """
    Parse PPTX with vision fallback for image-heavy slides.

    For each slide:
    1. Extract text normally
    2. If slide has images OR very little text → also run LLaVA
    3. Combine text + vision description into rich content
    """
    prs          = Presentation(io.BytesIO(file_bytes))
    slides       = []
    llava_ready  = use_vision and _is_llava_available()

    if use_vision and not llava_ready:
        print("⚠️  LLaVA not found — falling back to text-only extraction.")
        print("    To enable vision: ollama pull llava")

    for slide_num, slide in enumerate(prs.slides, start=1):
        # Step 1 — extract all text
        text_parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text_frame"):
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        text_parts.append(t)
            elif hasattr(shape, "text") and shape.text.strip():
                text_parts.append(shape.text.strip())

        text_content = "\n".join(text_parts).strip()

        # Step 2 — decide if vision is needed
        needs_vision = (
            llava_ready and (
                _slide_has_images(slide) or
                _slide_text_is_thin(slide)
            )
        )

        if needs_vision:
            print(f"   🔍 Running vision on slide {slide_num}...")
            slide_img    = _slide_to_pil_image(slide)
            vision_desc  = _describe_image_with_llava(slide_img, slide_num)

            # Combine: text + vision
            if text_content:
                combined = (
                    f"{text_content}\n\n"
                    f"[Visual content from slide {slide_num}]: {vision_desc}"
                )
            else:
                combined = (
                    f"[Slide {slide_num} — visual content]: {vision_desc}"
                )
            slides.append(combined)
        else:
            # Text-only
            if text_content:
                slides.append(text_content)
            else:
                slides.append(f"[Slide {slide_num} — no extractable content]")

    return slides, len(slides)

# ── Main ingestion ─────────────────────────────────────────
def ingest_document(team_name: str, filename: str,
                    file_bytes: bytes) -> dict:
    """
    Full pipeline:
    1. Register in SQLite  → status: pending
    2. Parse file (text + vision if needed)
    3. Update             → status: processing
    4. Chunk + embed + store in ChromaDB
    5. Update             → status: indexed / failed
    """
    file_type    = "pdf" if filename.lower().endswith(".pdf") else "pptx"
    file_size_kb = len(file_bytes) // 1024

    # Step 1 — register in SQLite
    add_team(team_name)
    doc_id = register_document(
        team_name, filename, file_type, file_size_kb
    )

    try:
        # Step 2 — parse
        update_document_status(doc_id, "processing")
        print(f"\n📄 Parsing {filename} for team '{team_name}'...")

        if file_type == "pdf":
            pages, total_pages = parse_pdf(file_bytes)
        else:
            pages, total_pages = parse_pptx(file_bytes, use_vision=True)

        print(f"   ✅ Parsed {total_pages} slides/pages")

        # Step 3 — chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        all_chunks   = []
        all_metadata = []
        all_ids      = []

        for page_num, page_text in enumerate(pages, start=1):
            if not page_text.strip():
                continue
            chunks = splitter.split_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc{doc_id}_p{page_num}_c{chunk_idx}"
                all_chunks.append(chunk)
                all_metadata.append({
                    "team":      team_name,
                    "doc_id":    str(doc_id),
                    "filename":  filename,
                    "file_type": file_type,
                    "page":      page_num,
                })
                all_ids.append(chunk_id)

        # Step 4 — store in ChromaDB in batches
        collection = get_collection()
        if all_chunks:
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                collection.add(
                    documents=all_chunks[i:i+batch_size],
                    metadatas=all_metadata[i:i+batch_size],
                    ids=all_ids[i:i+batch_size]
                )

        print(f"   ✅ {len(all_chunks)} chunks stored in ChromaDB")

        # Step 5 — mark indexed
        update_document_status(
            doc_id, "indexed",
            total_pages=total_pages,
            total_chunks=len(all_chunks)
        )

        return {
            "success":      True,
            "doc_id":       doc_id,
            "total_pages":  total_pages,
            "total_chunks": len(all_chunks)
        }

    except Exception as e:
        print(f"   ❌ Ingestion failed: {str(e)}")
        update_document_status(
            doc_id, "failed", error_message=str(e)
        )
        return {"success": False, "error": str(e)}

# ── Retrieval ──────────────────────────────────────────────
def retrieve_context(query: str, team_name: str = None,
                     n_results: int = 5) -> list[dict]:
    """
    Hybrid search: vector (ChromaDB) + keyword (BM25)
    Returns list of {text, filename, page, team, score}
    """
    collection = get_collection()
    where      = {"team": team_name} if team_name else None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except Exception:
        return []

    chunks = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text":     doc,
                "filename": meta.get("filename", ""),
                "page":     meta.get("page", 0),
                "team":     meta.get("team", ""),
                "score":    round(1 - dist, 3)
            })

    # BM25 re-rank
    if chunks:
        try:
            from rank_bm25 import BM25Okapi
            tokenised   = [c["text"].lower().split() for c in chunks]
            bm25        = BM25Okapi(tokenised)
            bm25_scores = bm25.get_scores(query.lower().split())
            for i, c in enumerate(chunks):
                c["score"] = round(
                    c["score"] * 0.6 + bm25_scores[i] * 0.4, 3
                )
            chunks.sort(key=lambda x: x["score"], reverse=True)
        except Exception:
            pass

    return chunks

def delete_team_documents(team_name: str):
    """Remove all ChromaDB entries for a team"""
    collection = get_collection()
    collection.delete(where={"team": team_name})