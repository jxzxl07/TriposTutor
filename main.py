from __future__ import annotations
import io
import json
import os
import re
import sys, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GEMINI_API_KEY = "AIzaSyDamaXEf7v-XvvEsfkSkgvDVAxMMog05hM"

# UI
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QMessageBox,
)

# PDF rendering & parsing
try:
    from pypdf import PdfReader 
except Exception:
    PdfReader = None  # type: ignore

try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:
    convert_from_path = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

# Gemini
try:
    import google.generativeai as genai  # type: ignore
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDamaXEf7v-XvvEsfkSkgvDVAxMMog05hM")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = "gemini-2.0-flash-exp"
        MODEL_AVAILABLE = True
    else:
        MODEL_AVAILABLE = False
except Exception:
    genai = None  # type: ignore
    MODEL_AVAILABLE = False
    GEMINI_MODEL = "gemini-2.0-flash-exp"

# Embeddings (optional)
USE_EMBEDDINGS = False
EMBEDDER = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore

    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    USE_EMBEDDINGS = True
except Exception:
    USE_EMBEDDINGS = False
    EMBEDDER = None

# Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PDF_DIR = BASE_DIR / "tripos_data"
DB_PATH = DATA_DIR / "tripos_questions.json"
PROGRESS_PATH = DATA_DIR / "progress.json"
SESSIONS_DIR = DATA_DIR / "sessions"
CACHE_DIR = DATA_DIR / "pdf_cache"

for d in (DATA_DIR, SESSIONS_DIR, PDF_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =========================
# PDF extraction & DB build
# =========================
def extract_questions_bulletproof(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    ULTRA-ROBUST question extraction - line-by-line processing.
    Returns: list of {qnum, page, text, confidence}
    """
    if not PdfReader:
        raise RuntimeError("pypdf not installed")
    
    print(f"  Extracting from {pdf_path.name}...")
    reader = PdfReader(str(pdf_path))
    
    # Process line-by-line (more robust than full-text regex)
    questions = []
    current_q = None
    
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            # Normalize immediately
            text = normalize_text(text)
        except Exception as e:
            print(f"  [WARN] Page {page_num} extraction failed: {e}")
            continue
        
        lines = text.split('\n')
        
        for line in lines:
            # Match question patterns: "1 ", "1. ", "1)", "Question 1", etc.
            match = re.match(r'^(?:\s*)((?:Question\s+)?(\d{1,2})[.)\s:]+)(.*)$', line, re.IGNORECASE)
            
            if match:
                qnum = int(match.group(2))
                
                # Save previous question
                if current_q and len(current_q['text']) > 20:
                    questions.append(current_q)
                
                # Start new question
                current_q = {
                    'qnum': qnum,
                    'page': page_num,
                    'text': match.group(3).strip(),
                    'confidence': 'high'
                }
            elif current_q:
                # Continue current question
                current_q['text'] += ' ' + line.strip()
    
    # Don't forget last question
    if current_q and len(current_q['text']) > 20:
        questions.append(current_q)
    
    # Fallback: If no questions found, use page-based splitting
    if not questions:
        print(f"  [WARN] No question markers found, using page-based extraction")
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = normalize_text(page.extract_text() or "")
                if len(text.strip()) > 50:
                    questions.append({
                        'qnum': page_num,
                        'page': page_num,
                        'text': text.strip(),
                        'confidence': 'medium'
                    })
            except:
                pass
    
    # Clean up question text
    for q in questions:
        # Remove excessive whitespace
        q['text'] = re.sub(r'\s+', ' ', q['text']).strip()
        # Limit to reasonable length
        if len(q['text']) > 5000:
            q['text'] = q['text'][:5000] + "..."
    
    print(f"  ✓ Extracted {len(questions)} questions")
    return questions


def build_or_load_db(force_rebuild=False):
    """
    Build comprehensive database with metadata
    """
    if DB_PATH.exists() and not force_rebuild:
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"✓ Loaded {len(data)} questions from cache")
                return data
        except:
            print("⚠ Cache corrupted, rebuilding...")
    
    # Backup old DB before rebuilding
    if DB_PATH.exists():
        backup_path = DATA_DIR / "tripos_questions.backup.json"
        import shutil
        shutil.copy(DB_PATH, backup_path)
        print(f"[BUILD DB] Backed up old database to {backup_path.name}")
    
    print("\n🔨 Building question database...")
    all_questions = []
    
    pdf_files = sorted([f for f in PDF_DIR.glob("*.pdf")])
    
    if not pdf_files:
        print(f"⚠ No PDF files found in {PDF_DIR}")
        return []
    
    print(f"[BUILD DB] Found {len(pdf_files)} PDF files")
    
    for path in pdf_files:
        try:
            questions = extract_questions_bulletproof(path)
            print(f"[BUILD DB] {path.name}: extracted {len(questions)} questions")
            
            # Show a sample of extracted text for first question
            if questions and len(questions) > 0:
                sample_text = questions[0]['text'][:100]
                print(f"  Sample: {sample_text}...")
                
        except Exception as exc:
            print(f"[BUILD DB] Error extracting {path.name}: {exc}")
            continue
        
        # Parse filename for metadata
        fname = path.name
        year = None
        paper = None
        m = re.search(r'(\d{4})', fname)
        if m:
            year = int(m.group(1))
        m = re.search(r'[_\-\s](\d)(?:\D|$)', fname)
        if m:
            paper = int(m.group(1))
        
        for q in questions:
            item = {
                "id": f"{fname}__q{q['qnum']}_p{q['page']}",
                "filename": fname,
                "year": year,
                "paper": paper,
                "qnum": q['qnum'],
                "page": q['page'],
                "text": q['text'],
                "confidence": q['confidence'],
                "topics": [],
                "difficulty": None
            }
            all_questions.append(item)
    
    # Save
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Built database with {len(all_questions)} questions from {len(pdf_files)} PDFs\n")
    return all_questions


# =========================
# Optional embeddings build
# =========================
embeddings_index: Optional[Dict[str, Any]] = None

def normalize_text(text: str) -> str:
    """Normalize text for better searching - handle ligatures and unicode."""
    import unicodedata
    # Normalize unicode (handle ligatures like ﬀ, ﬁ, ﬂ)
    text = unicodedata.normalize('NFKD', text)
    # Replace common ligatures that might not normalize
    replacements = {
        'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        'ﬅ': 'st', 'ﬆ': 'st',}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def build_embeddings_if_needed(db: List[Dict[str, Any]]) -> None:
    """Build embedding index if EMBEDDER is available (optional)."""
    global embeddings_index
    if not USE_EMBEDDINGS or EMBEDDER is None:
        return
    texts = [q["text"][:500] for q in db]
    embs = EMBEDDER.encode(texts, show_progress_bar=False)
    embeddings_index = {"ids": [q["id"] for q in db], "embs": embs}
    print("✓ Built embeddings")


# =========================
# Search
# =========================
def search_questions(db: List[Dict[str, Any]], query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
    """Search DB for query (token matching fallback)."""
    q = normalize_text(query.strip().lower())
    if not q:
        return []
    
    # DEBUG: Print database size
    print(f"[DEBUG] Database has {len(db)} total questions")
    
    # Show sample of what's in the database
    if db:
        sample = db[0]
        sample_text = normalize_text(sample.get('text', ''))
        print(f"[DEBUG] Sample question text (first 200 chars): {sample_text[:200]}")
        print(f"[DEBUG] Sample filename: {sample.get('filename', 'N/A')}")
    
    # More lenient filtering - only require 50 chars minimum
    valid_db = [item for item in db if len(item.get("text", "")) > 50]
    print(f"[DEBUG] After filtering: {len(valid_db)} valid questions")
    
    if not valid_db:
        print("[DEBUG] No valid questions found, using all DB entries")
        valid_db = db
    
    results: List[Tuple[Dict[str, Any], float]] = []
    tokens = set(q.split())
    print(f"[DEBUG] Search tokens: {tokens}")
    
    for idx, item in enumerate(valid_db):
        text = normalize_text(item.get("text") or "").lower()
        filename = normalize_text(item.get("filename") or "").lower()
        score = 0
        
        # Exact phrase match
        if q in text:
            score += 1000
        
        # Token matching
        text_tokens = set(text.split())
        matching = tokens.intersection(text_tokens)
        score += len(matching) * 100
        
        # Filename matching
        for token in tokens:
            if token in filename:
                score += 50
        
        if score > 0:
            normalized = score / (1 + len(text) / 1000)
            results.append((item, normalized))
        
        # Show why first few items didn't match
        if idx < 3 and score == 0:
            preview = text[:150] if text else "(empty)"
            print(f"[DEBUG] Question {idx+1} (score=0): {preview}...")
    
    print(f"[DEBUG] Found {len(results)} matches for query '{q}'")
    
    # If no results, return ALL questions so user can see what's available
    if not results:
        print("[DEBUG] No matches found - returning all questions for browsing")
        results = [(item, 1.0) for item in valid_db[:top_k]]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# =========================
# Session/progress utils
# =========================
def load_progress() -> Dict[str, Any]:
    """Load progress JSON if present."""
    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_progress(progress: Dict[str, Any]) -> None:
    """Save progress JSON."""
    with open(PROGRESS_PATH, "w", encoding="utf-8") as fh:
        json.dump(progress, fh, indent=2, ensure_ascii=False)


def save_session_log(question_id: str, chat_history: List[Dict[str, str]]) -> None:
    """Save per-question chat history."""
    path = SESSIONS_DIR / f"{question_id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"id": question_id, "history": chat_history, "updated": datetime.now(timezone.utc).isoformat()}, fh, indent=2, ensure_ascii=False)


def load_session_log(question_id: str) -> List[Dict[str, str]]:
    """Load per-question chat history."""
    path = SESSIONS_DIR / f"{question_id}.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("history", [])
    except Exception:
        return []


# =========================
# Gemini wrapper
# =========================
def ask_gemini_supervisor(question_text: str, user_message: str, chat_history_str: str, hint_level: int = 1) -> str:
    """Ask Gemini for a hint/explanation with more conversational and context-aware responses."""
    if not MODEL_AVAILABLE:
        hints = {
            1: "💡 Think about the fundamental concept here. What's the core idea?",
            2: "💡 Try breaking this into steps: 1) What do you know? 2) What do you need? 3) How do you connect them?",
            3: "💡 Here's a structured approach: Start with definitions, then build up to the solution step by step.",
        }
        return hints.get(hint_level, "Set GEMINI_API_KEY to enable AI hints.")

    try:
        # More elaborate system prompt to create a more engaging persona
        system_prompt = """You are an exceptional Cambridge Computer Science supervisor named Dr. Alex Thompson. 
Your teaching philosophy:
- Use the Socratic method to guide students to discover answers
- Be encouraging, supportive, and slightly witty
- Provide context and relate concepts to real-world scenarios
- Ask probing questions that help students think critically
- Adapt your explanation style based on the student's previous responses
- Never give away the full answer unless explicitly requested

Communication style:
- Conversational but academically rigorous
- Use analogies and relatable examples
- Show genuine interest in the student's learning process
- Vary your response length and depth based on the student's understanding

Student level: Cambridge Computer Science Tripos
Tone: Friendly, patient, intellectually stimulating"""

        # Dynamic instruction based on conversation context and hint level
        context_instructions = {
            1: "Provide a gentle, thought-provoking nudge. Ask a leading question that hints at the solution approach.",
            2: "Give a more substantial guidance, outlining key considerations and potential solution strategies without revealing too much.",
            3: "Offer a comprehensive roadmap, breaking down the problem into manageable steps and explaining the reasoning behind each step.",
        }
        instruct = context_instructions.get(hint_level, context_instructions[1])

        # Analyze previous chat history for context
        prev_context = "No previous context" if not chat_history_str else chat_history_str[-1000:]

        # Craft a prompt that encourages a more natural, context-aware response
        prompt = f"""{system_prompt}

SPECIFIC QUESTION CONTEXT:
{question_text[:1000]}

PREVIOUS CONVERSATION:
{prev_context}

INSTRUCTION: {instruct}

Student's latest message: {user_message}

Generate a response that:
1. Acknowledges the student's current level of understanding
2. Provides targeted guidance
3. Encourages further thinking
4. Maintains an engaging, supportive tone"""

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        # Add some randomness to prevent repetitive responses
        if hasattr(response, "text"):
            text = response.text.strip()
            # Add minor variations to make responses feel more natural
            variations = [
                "Hmm, interesting approach. ",
                "Let's dig a bit deeper. ",
                "That's a great starting point. ",
                "I see where you're going with this. ",
            ]
            if text and random.random() < 0.3:  # 30% chance of adding a variation
                text = random.choice(variations) + text
            return text
        
        return str(response)
    
    except Exception as exc:
        return f"⚠ AI Error: {exc}"



# =========================
# PDFLoaderThread (QThread)
# =========================
class PDFLoaderThread(QThread):
    """Thread to render PDF pages into a single combined QPixmap and emit page offsets."""

    finished = pyqtSignal(object, list, list)  # QPixmap, positions, heights
    error = pyqtSignal(str)

    def __init__(self, pdf_path: Path, target_page: int = 1) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.target_page = int(target_page)

    def run(self) -> None:
        """Convert PDF to images, combine, cache, and emit QPixmap + metadata."""
        try:
            if convert_from_path is None or Image is None:
                self.error.emit("pdf2image or pillow not installed")
                return

            cache_file = CACHE_DIR / f"{self.pdf_path.stem}_combined.png"
            cache_meta = CACHE_DIR / f"{self.pdf_path.stem}_meta.json"

            if cache_file.exists() and cache_meta.exists():
                with open(cache_meta, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                pixmap = QPixmap(str(cache_file))
                self.finished.emit(pixmap, meta.get("positions", []), meta.get("heights", []))
                return

            images = convert_from_path(str(self.pdf_path), dpi=150)
            if not images:
                self.error.emit("Could not render PDF (pdf2image returned no images)")
                return

            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            combined = Image.new("RGB", (max_width, total_height), "white")

            page_positions: List[int] = []
            page_heights: List[int] = []
            y_offset = 0
            for img in images:
                page_positions.append(y_offset)
                page_heights.append(img.height)
                combined.paste(img, (0, y_offset))
                y_offset += img.height

            # Save cache
            combined.save(cache_file, "PNG", optimize=True)
            with open(cache_meta, "w", encoding="utf-8") as fh:
                json.dump({"positions": page_positions, "heights": page_heights}, fh)

            # Convert to QPixmap
            img_byte = io.BytesIO()
            combined.save(img_byte, format="PNG")
            img_byte.seek(0)
            qimg = QImage.fromData(img_byte.getvalue())
            pixmap = QPixmap.fromImage(qimg)
            self.finished.emit(pixmap, page_positions, page_heights)
        except Exception as exc:
            self.error.emit(str(exc))


# =========================
# GUI (single-file)
# =========================
class TriposTutorGUI(QWidget):
    """Main TriposTutor GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tripos Tutor")
        self.resize(1600, 1000)

        # State & data
        self.db: List[Dict[str, Any]] = build_or_load_db()
        if USE_EMBEDDINGS:
            try:
                build_embeddings_if_needed(self.db)
            except Exception:
                pass
        self.progress: Dict[str, Any] = load_progress()
        self.current_topic: Optional[str] = None
        self.current_matches: List[Tuple[Dict[str, Any], float]] = []
        self.current_index: int = 0
        self.current_question: Optional[Dict[str, Any]] = None
        self.chat_history: List[Dict[str, str]] = []
        self.hint_level: int = 1

        # PDF loader thread reference (if active)
        self.pdf_loader: Optional[PDFLoaderThread] = None
        self.page_positions: List[int] = []
        self.page_heights: List[int] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create widgets and layout."""
        self.setStyleSheet(
            """
            QWidget { background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; }
            QListWidget { background: #1e293b; border: 2px solid #334155; border-radius: 12px; padding: 8px; color: #e2e8f0; font-size: 13px; }
            QTextEdit { background: #1e293b; border: 2px solid #334155; border-radius: 12px; padding: 12px; color: #e2e8f0; font-size: 13px; }
            QLineEdit { background: #1e293b; border: 2px solid #334155; border-radius: 10px; padding: 12px; color: #e2e8f0; font-size: 14px; }
            QPushButton { background: #3b82f6; color: white; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 600; font-size: 14px; }
            QLabel { color: #e2e8f0; }
            QScrollArea { background: #1e293b; border: 2px solid #334155; border-radius: 12px; }
            QProgressBar { border: 2px solid #334155; border-radius: 8px; background: #1e293b; text-align: center; color: white; }
            QProgressBar::chunk { background: #3b82f6; border-radius: 6px; }
            """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        header = QLabel("🎓 TriposTutor — Ultimate Edition")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #3b82f6; padding: 12px;")
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)

        search_label = QLabel("🔍 Search Questions")
        search_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        left_layout.addWidget(search_label)

        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("e.g., algorithms, graphs, complexity...")
        self.topic_input.returnPressed.connect(self.on_search)
        left_layout.addWidget(self.topic_input)

        self.search_btn = QPushButton("🔍 Search")
        self.search_btn.clicked.connect(self.on_search)
        left_layout.addWidget(self.search_btn)

        results_label = QLabel("📚 Results")
        results_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(results_label)

        self.list_widget = QListWidget()

        self.list_widget.setStyleSheet("""
        QListWidget {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 6px;
            font-size: 13px;
            color: #e2e8f0;
            outline: none;
        }

        QListWidget::item {
            background-color: transparent;
            padding: 10px 8px;
            margin: 4px 0;
            border-radius: 8px;
            line-height: 1.3em;
        }

        QListWidget::item:hover {
            background-color: #1e293b;
            border: 1px solid #3b82f6;
        }

        QListWidget::item:selected {
            background-color: #3b82f6;
            color: white;
            font-weight: 600;
            border: 1px solid #60a5fa;
        }

        QListWidget::item:selected:!active {
            background-color: #2563eb;
            color: #fff;
        }

        QListWidget QScrollBar:vertical {
            background: #0f172a;
            width: 10px;
            margin: 8px 0 8px 0;
            border-radius: 8px;
        }

        QListWidget QScrollBar::handle:vertical {
            background: #334155;
            min-height: 20px;
            border-radius: 8px;
        }

        QListWidget QScrollBar::handle:vertical:hover {
            background: #475569;
        }

        QListWidget QScrollBar::add-line:vertical,
        QListWidget QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)


        self.list_widget.itemClicked.connect(self.on_question_selected)
        left_layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.rebuild_btn = QPushButton("🔄")
        self.rebuild_btn.setToolTip("Rebuild Database")
        self.rebuild_btn.clicked.connect(self.on_rebuild)
        btn_layout.addWidget(self.rebuild_btn)

        self.import_btn = QPushButton("📁")
        self.import_btn.setToolTip("Import PDFs")
        self.import_btn.clicked.connect(self.on_import)
        btn_layout.addWidget(self.import_btn)

        left_layout.addLayout(btn_layout)
        splitter.addWidget(left_widget)

        # Right panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(12)

        pdf_label = QLabel("📄 Question")
        pdf_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(pdf_label)

        self.question_image = QLabel()
        self.question_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.question_image.setScaledContents(False)

        self.question_scroll = QScrollArea()
        self.question_scroll.setWidget(self.question_image)
        self.question_scroll.setWidgetResizable(True)
        self.question_scroll.setMinimumHeight(300)
        right_layout.addWidget(self.question_scroll, stretch=4)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        right_layout.addWidget(self.progress_bar)

        text_label = QLabel("📝 Extracted Text")
        text_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        right_layout.addWidget(text_label)

        self.question_box = QTextEdit()
        self.question_box.setReadOnly(True)
        self.question_box.setMinimumHeight(30) 
        right_layout.addWidget(self.question_box)

        self.question_box.setStyleSheet("""
            QListWidget {
                background-color: #f8f9fa;
                border: 1px solid #d0d7de;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
                color: #1a1a1a;
            }

            QListWidget::item {
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 4px;
            }

            QListWidget::item:hover {
                background-color: #e0e7ff;
            }

            QListWidget::item:selected {
                background-color: #4f46e5;
                color: white;
                font-weight: 500;
            }
        """)


        chat_label = QLabel("💬 Chat with AI Supervisor")
        chat_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(chat_label)

        self.chat_box = QTextEdit()
        self.chat_box.setReadOnly(True)
        self.chat_box.setMinimumHeight(250)
        right_layout.addWidget(self.chat_box, stretch=3)

        input_layout = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type your answer, ask for /hint, or use /next, /skip, /show_answer...")
        self.input_line.returnPressed.connect(self.on_send)
        input_layout.addWidget(self.input_line, stretch=5)

        self.send_btn = QPushButton("📤")
        self.send_btn.clicked.connect(self.on_send)
        input_layout.addWidget(self.send_btn)

        self.hint_btn = QPushButton("💡")
        self.hint_btn.setToolTip("Get a hint")
        self.hint_btn.clicked.connect(self.on_hint)
        input_layout.addWidget(self.hint_btn)

        self.next_btn = QPushButton("⏭️")
        self.next_btn.setToolTip("Next question")
        self.next_btn.clicked.connect(self.on_next)
        input_layout.addWidget(self.next_btn)

        self.mark_btn = QPushButton("✅")
        self.mark_btn.setToolTip("Mark as solved")
        self.mark_btn.clicked.connect(self.on_mark_solved)
        input_layout.addWidget(self.mark_btn)

        right_layout.addLayout(input_layout)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        self.status = QLabel("✨ Ready! Search for a topic to begin.")
        self.status.setStyleSheet("background: #1e293b; padding: 12px; border-radius: 8px; border: 2px solid #334155;")
        main_layout.addWidget(self.status)

        self.display_system("👋 Welcome! Search for questions by topic, then chat with your AI supervisor for hints.")

    # -----------------------
    # Actions
    # -----------------------
    def on_search(self) -> None:
        """Search for topic and populate result list."""
        topic = self.topic_input.text().strip()
        if not topic:
            QMessageBox.information(self, "No Query", "Enter a search term!")
            return

        self.status.setText(f"🔍 Searching for '{topic}'...")
        QApplication.processEvents()

        matches = search_questions(self.db, topic, top_k=50)
        self.current_matches = matches
        self.current_index = 0

        self.list_widget.clear()
        for item, score in matches:
            filename = item.get("filename", "Unknown")
            qnum = item.get("qnum", "?")
            page = item.get("page", "?")
            text = item.get("text", "")
            confidence = item.get("confidence", "medium")
            title = f"{filename} Q{qnum} (Page {page})"
            snippet = text[:120].replace("\n", " ") if text else "No text available"
            list_item = QListWidgetItem(f"{title}\n{snippet}...\n[{confidence} confidence, score: {score:.2f}]")
            self.list_widget.addItem(list_item)

        if matches:
            self.show_question(matches[0][0])
            self.status.setText(f"✅ Found {len(matches)} questions for '{topic}'")
            self.display_system(f"Found {len(matches)} matches. Showing top result.")
        else:
            self.status.setText(f"❌ No matches for '{topic}'")
            self.display_system("No questions found. Try different keywords.")

    def on_question_selected(self, _: QListWidgetItem) -> None:
        """Handle user selecting a list item."""
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.current_matches):
            self.current_index = row
            self.show_question(self.current_matches[row][0])

    def show_question(self, item: Dict[str, Any]) -> None:
        """Display question text, load its PDF image and chat history."""
        self.current_question = item
        filename = item.get("filename", "Unknown")
        qnum = item.get("qnum", "?")
        page = item.get("page", "?")
        text = item.get("text", "No text available")

        self.question_box.setPlainText(f"[{filename}] Q{qnum} (Page {page})\n\n{text}")

        # Load chat history
        question_id = item.get("id", f"{filename}_q{qnum}")
        self.chat_history = load_session_log(question_id) or []
        self.refresh_chat()

        # Ensure any existing loader is stopped before starting a new one
        if self.pdf_loader is not None and self.pdf_loader.isRunning():
            try:
                self.pdf_loader.finished.disconnect()
            except Exception:
                pass
            try:
                self.pdf_loader.error.disconnect()
            except Exception:
                pass
            self.pdf_loader.quit()
            self.pdf_loader.wait(timeout=2000)
            self.pdf_loader = None

        # Start new loader thread
        pdf_path = PDF_DIR / filename
        if pdf_path.exists():
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Loading PDF...")
            target_page = int(page) if isinstance(page, int) else 1
            loader = PDFLoaderThread(pdf_path, target_page)
            # keep reference
            self.pdf_loader = loader
            loader.finished.connect(self.on_pdf_loaded)
            loader.error.connect(self.on_pdf_error)
            loader.start()
        else:
            self.question_image.setText(f"📄 PDF not found: {filename}")
        self.hint_level = 1
        self.status.setText(f"📖 Question {qnum} from {filename}")

    def on_pdf_loaded(self, pixmap: QPixmap, positions: List[int], heights: List[int]) -> None:
        """Slot called when PDFLoaderThread emits finished."""
        # Hide progress UI
        self.progress_bar.setVisible(False)
        self.page_positions = positions
        self.page_heights = heights

        # Scale combined pixmap to available width (keep original pixmap for scaling math)
        if pixmap.isNull():
            self.question_image.setText("❌ Failed to render PDF image")
            return

        avail_width = 800
        scaled = pixmap.scaledToWidth(avail_width, Qt.TransformationMode.SmoothTransformation)
        self.question_image.setPixmap(scaled)

        # Scroll to target page (compute scaled offset)
        if self.current_question and self.page_positions:
            page_val = self.current_question.get("page", 1)
            try:
                target_idx = max(int(page_val) - 1, 0)
            except (ValueError, TypeError):
                target_idx = 0
            if 0 <= target_idx < len(self.page_positions):
                # scale factor = scaled.width / original_pixmap.width
                orig_w = pixmap.width() or 1
                scale_factor = scaled.width() / orig_w
                scroll_pos = int(self.page_positions[target_idx] * scale_factor)
                QTimer.singleShot(100, lambda: self.question_scroll.verticalScrollBar().setValue(scroll_pos))

        # cleanup loader reference safely after finishing
        if self.pdf_loader is not None and isinstance(self.pdf_loader, QThread):
            # disconnect signals and delete reference — thread should be finished
            try:
                self.pdf_loader.finished.disconnect(self.on_pdf_loaded)
            except Exception:
                pass
            try:
                self.pdf_loader.error.disconnect(self.on_pdf_error)
            except Exception:
                pass
            self.pdf_loader = None

    def on_pdf_error(self, error_msg: str) -> None:
        """Slot called when PDF thread signals an error."""
        self.progress_bar.setVisible(False)
        self.question_image.setText(f"❌ Error loading PDF:\n{error_msg}")
        # cleanup
        if self.pdf_loader is not None:
            try:
                self.pdf_loader.finished.disconnect()
            except Exception:
                pass
            try:
                self.pdf_loader.error.disconnect()
            except Exception:
                pass
            self.pdf_loader = None

    def on_send(self) -> None:
        """Handle send (user message)."""
        msg = self.input_line.text().strip()
        if not msg:
            return
        if not self.current_question:
            QMessageBox.warning(self, "No Question", "Select a question first!")
            return
        self.input_line.clear()
        question_id = self.current_question.get("id", "unknown")

        if msg.lower() in ("/next", "/skip"):
            self.on_next()
            return
        if msg.lower() == "/hint":
            self.on_hint()
            return
        if msg.lower() == "/show_answer":
            self.show_full_answer()
            return
        if msg.lower() == "/clear":
            self.chat_history = []
            self.refresh_chat()
            save_session_log(question_id, self.chat_history)
            return

        self.chat_history.append({"role": "user", "content": msg})
        self.display_user(msg)

        chat_str = "\n".join(f"{m['role']}: {m['content']}" for m in self.chat_history[-10:])
        response = ask_gemini_supervisor(self.current_question.get("text", ""), msg, chat_str, self.hint_level)

        self.chat_history.append({"role": "assistant", "content": response})
        self.display_ai(response)
        save_session_log(question_id, self.chat_history)

    def on_hint(self) -> None:
        """Increase hint level and request hint."""
        if not self.current_question:
            QMessageBox.warning(self, "No Question", "Select a question first!")
            return
        self.hint_level = min(self.hint_level + 1, 3)
        chat_str = "\n".join(f"{m['role']}: {m['content']}" for m in self.chat_history[-10:])
        response = ask_gemini_supervisor(self.current_question.get("text", ""), f"I need a hint (level {self.hint_level})", chat_str, self.hint_level)
        self.chat_history.append({"role": "system", "content": f"[Hint Level {self.hint_level}]"})
        self.chat_history.append({"role": "assistant", "content": response})
        self.display_system(f"[Hint Level {self.hint_level}/3]")
        self.display_ai(response)
        question_id = self.current_question.get("id", "unknown")
        save_session_log(question_id, self.chat_history)

    def show_full_answer(self) -> None:
        """Request a full solution from the model (explicit)."""
        if not self.current_question:
            return
        chat_str = "\n".join(f"{m['role']}: {m['content']}" for m in self.chat_history[-10:])
        question_text = self.current_question.get("text", "No question text available")
        prompt = f"""Provide a COMPLETE, detailed solution to this question:

{question_text}

Previous discussion:
{chat_str}

Now give the full answer with all steps, calculations, and explanations."""
        try:
            if MODEL_AVAILABLE:
                model = genai.GenerativeModel(GEMINI_MODEL)
                response = model.generate_content(prompt)
                answer = response.text if hasattr(response, "text") else str(response)
            else:
                answer = "Set GEMINI_API_KEY to get full solutions."
        except Exception as exc:
            answer = f"Error: {exc}"

        self.chat_history.append({"role": "system", "content": "[FULL SOLUTION]"})
        self.chat_history.append({"role": "assistant", "content": answer})
        self.display_system("[📖 FULL SOLUTION]")
        self.display_ai(answer)
        question_id = self.current_question.get("id", "unknown")
        save_session_log(question_id, self.chat_history)

    def on_next(self) -> None:
        """Move to next search result."""
        if not self.current_matches:
            QMessageBox.information(self, "No Results", "Search for questions first!")
            return
        self.current_index = (self.current_index + 1) % len(self.current_matches)
        self.show_question(self.current_matches[self.current_index][0])
        self.list_widget.setCurrentRow(self.current_index)

    def on_mark_solved(self) -> None:
        """Mark current question as solved and advance."""
        if not self.current_question:
            QMessageBox.warning(self, "No Question", "Select a question first!")
            return
        qid = self.current_question.get("id", "unknown")
        self.progress[qid] = {"solved": True, "date": datetime.now(timezone.utc).isoformat()}
        save_progress(self.progress)
        self.display_system("✅ Marked as solved! Moving to next question...")
        QTimer.singleShot(1000, self.on_next)

    def on_rebuild(self) -> None:
        """Rebuild DB from PDFs."""
        reply = QMessageBox.question(self, "Rebuild Database", "This will re-extract all questions from PDFs. Continue?")
        if reply == QMessageBox.StandardButton.Yes:
            self.status.setText("🔨 Rebuilding database...")
            QApplication.processEvents()
            self.db = build_or_load_db(force_rebuild=True)
            if USE_EMBEDDINGS:
                try:
                    build_embeddings_if_needed(self.db)
                except Exception:
                    pass
            self.current_matches = []
            self.list_widget.clear()
            self.status.setText(f"✅ Database rebuilt: {len(self.db)} questions")

    def on_import(self) -> None:
        """Import PDFs into tripos_data and rebuild if imported."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select PDF files", str(PDF_DIR), "PDF Files (*.pdf)")
        if not files:
            return
        imported = 0
        for file_path in files:
            dest = PDF_DIR / Path(file_path).name
            if not dest.exists():
                import shutil

                shutil.copy(file_path, dest)
                imported += 1
        if imported > 0:
            QMessageBox.information(self, "Import Complete", f"Imported {imported} PDF(s). Rebuilding database...")
            self.on_rebuild()
        else:
            QMessageBox.information(self, "Import", "No new files to import.")

    # -----------------------
    # Chat display helpers
    # -----------------------
    def display_system(self, text: str) -> None:
        """Display a system message in chat."""
        self.chat_box.append(f'<div style="color: #94a3b8; font-style: italic; margin: 8px 0;">{text}</div>')
        self.chat_box.verticalScrollBar().setValue(self.chat_box.verticalScrollBar().maximum())

    def display_user(self, text: str) -> None:
        """Display user's message in chat."""
        self.chat_box.append(
            f'<div style="background: #3b82f6; color: white; padding: 12px; '
            f'border-radius: 12px; margin: 8px 0; max-width: 80%;"><b>You:</b> {text}</div>'
        )
        self.chat_box.verticalScrollBar().setValue(self.chat_box.verticalScrollBar().maximum())

    def display_ai(self, text: str) -> None:
        """Display assistant message in chat."""
        text_html = text.replace("**", "<b>").replace("\n", "<br>")
        self.chat_box.append(
            f'<div style="background: #1e293b; border: 2px solid #334155; padding: 12px; border-radius: 12px; margin: 8px 0;">'
            f'<b style="color: #3b82f6;">🧑‍🏫 Supervisor:</b><br>{text_html}</div>'
        )
        self.chat_box.verticalScrollBar().setValue(self.chat_box.verticalScrollBar().maximum())

    def refresh_chat(self) -> None:
        """Reload chat history into the chat box UI."""
        self.chat_box.clear()
        for msg in self.chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                self.display_user(content)
            elif role == "assistant":
                self.display_ai(content)
            elif role == "system":
                self.display_system(content)

    # -----------------------
    # Clean shutdown
    # -----------------------
    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Ensure background threads are stopped before the window is destroyed."""
        try:
            if self.pdf_loader is not None and self.pdf_loader.isRunning():
                try:
                    self.pdf_loader.finished.disconnect()
                except Exception:
                    pass
                try:
                    self.pdf_loader.error.disconnect()
                except Exception:
                    pass
                self.pdf_loader.quit()
                # wait up to 2s
                self.pdf_loader.wait(timeout=2000)
                self.pdf_loader = None
        except Exception:
            # best-effort cleanup; don't block exit indefinitely
            pass
        # call base closeEvent
        super().closeEvent(event)


# =========================
# Entry point
# =========================
def main() -> None:
    """Start application."""
    app = QApplication(sys.argv)
    app.setApplicationName("TriposTutor")
    # Check dependencies
    missing: List[str] = []
    if not PdfReader:
        missing.append("pypdf")
    if convert_from_path is None:
        missing.append("pdf2image")
    if missing:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Missing Dependencies")
        msg.setText(f"Please install: {', '.join(missing)}")
        msg.setDetailedText(
            "Run:\n"
            f"pip install {' '.join(missing)}\n\n"
            "Also ensure poppler is installed:\n"
            "- macOS: brew install poppler\n"
            "- Ubuntu: sudo apt-get install poppler-utils\n"
            "- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/"
        )
        msg.exec()
        # continue to allow partial UI usage
    # Launch UI
    window = TriposTutorGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
