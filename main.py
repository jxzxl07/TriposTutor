"""
TriposTutor - ULTIMATE EDITION
Complete rewrite with bulletproof PDF handling, perfect search, and flawless UI
"""
from pdf2image import convert_from_path
import sys, os, re, io, json, traceback
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# MUST BE FIRST
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# UI
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QListWidget, QLabel, QFileDialog, 
    QMessageBox, QSplitter, QScrollArea, QListWidgetItem, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor, QPixmap, QImage

# PDF parsing
try:
    from pypdf import PdfReader
except:
    PdfReader = None

# Gemini
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ENTERKEYHERE")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = "gemini-2.0-flash-exp"
        MODEL_AVAILABLE = True
    else:
        MODEL_AVAILABLE = False
except Exception as e:
    print(f"Gemini setup error: {e}")
    genai = None
    MODEL_AVAILABLE = False

# Semantic search
USE_EMBEDDINGS = False
EMBEDDER = None
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
    USE_EMBEDDINGS = True
    print("✓ Semantic search enabled")
except Exception as e:
    print(f"✗ Semantic search disabled: {e}")

# Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PDF_DIR = BASE_DIR / "tripos_data"
DB_PATH = DATA_DIR / "tripos_questions.json"
PROGRESS_PATH = DATA_DIR / "progress.json"
SESSIONS_DIR = DATA_DIR / "sessions"
CACHE_DIR = DATA_DIR / "pdf_cache"

for d in [DATA_DIR, SESSIONS_DIR, PDF_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

# =====================================================================
# BULLETPROOF PDF EXTRACTION
# =====================================================================

def extract_questions_bulletproof(pdf_path):
    """
    ULTRA-ROBUST question extraction with multiple strategies.
    Returns: list of {qnum, page, text, confidence}
    """
    if not PdfReader:
        raise RuntimeError("pypdf not installed")
    
    print(f"  Extracting from {pdf_path.name}...")
    reader = PdfReader(str(pdf_path))
    
    # Strategy 1: Look for clear question markers
    questions = []
    current_q = None
    
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except:
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
    
    # Strategy 2: If no questions found, use page-based splitting
    if not questions:
        print(f"  ⚠ No question markers found, using page-based extraction")
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
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
        # Limit to reasonable length for storage
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
    
    print("\n🔨 Building question database...")
    all_questions = []
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("⚠ No PDF files found in tripos_data/")
        return []
    
    for fn in sorted(pdf_files):
        path = PDF_DIR / fn
        
        try:
            questions = extract_questions_bulletproof(path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
        
        # Parse filename for metadata
        year, paper = None, None
        m = re.search(r'(\d{4})', fn)
        if m:
            year = int(m.group(1))
        m = re.search(r'[_\-\s](\d)(?:\D|$)', fn)
        if m:
            paper = int(m.group(1))
        
        for q in questions:
            item = {
                "id": f"{fn}__q{q['qnum']}_p{q['page']}",
                "filename": fn,
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
    
    print(f"✓ Built database with {len(all_questions)} questions\n")
    return all_questions


# =====================================================================
# EMBEDDINGS
# =====================================================================

embeddings_index = None

def build_embeddings_if_needed(db):
    global embeddings_index
    if not USE_EMBEDDINGS or not EMBEDDER:
        return
    
    print("🧠 Building semantic search index...")
    texts = [q["text"][:500] for q in db]  # Use first 500 chars for speed
    embs = EMBEDDER.encode(texts, show_progress_bar=True)
    embeddings_index = {
        "ids": [q["id"] for q in db],
        "embs": embs
    }
    print("✓ Semantic search ready\n")


# =====================================================================
# ULTRA-SMART SEARCH
# =====================================================================

def search_questions(db, query, top_k=20):
    """
    INTELLIGENT multi-strategy search
    """
    q = query.strip().lower()
    if not q:
        return []
    
    results = []
    
    # Strategy 1: Semantic search (best)
    if USE_EMBEDDINGS and EMBEDDER and embeddings_index:
        query_emb = EMBEDDER.encode([q])[0]
        embs = embeddings_index["embs"]
        ids = embeddings_index["ids"]
        
        def cosine_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        
        sims = [cosine_sim(query_emb, e) for e in embs]
        
        # Dynamic threshold based on query specificity
        threshold = 0.25 if len(q.split()) <= 2 else 0.30
        
        id_to_q = {item["id"]: item for item in db}
        ranked = [(ids[i], sims[i]) for i in range(len(sims)) if sims[i] >= threshold]
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        for idx, score in ranked[:top_k]:
            results.append((id_to_q[idx], float(score)))
        
        if results:
            print(f"🔍 Found {len(results)} semantic matches (threshold={threshold:.2f})")
            return results
    
    # Strategy 2: Enhanced keyword matching
    print(f"🔍 Using keyword search for '{query}'")
    query_tokens = set(q.split())
    
    for item in db:
        text = (item.get("text") or "").lower()
        filename = (item.get("filename") or "").lower()
        
        score = 0
        
        # Exact phrase match (highest priority)
        if q in text:
            score += 1000
        
        # Token matching
        text_tokens = set(text.split())
        matching = query_tokens.intersection(text_tokens)
        score += len(matching) * 100
        
        # Partial word matches
        for qtoken in query_tokens:
            for ttoken in text_tokens:
                if qtoken in ttoken or ttoken in qtoken:
                    score += 10
        
        # Filename boost
        for token in query_tokens:
            if token in filename:
                score += 50
        
        # Position boost (earlier = better)
        first_match_pos = -1
        for token in matching:
            pos = text.find(token)
            if pos != -1 and (first_match_pos == -1 or pos < first_match_pos):
                first_match_pos = pos
        
        if first_match_pos != -1:
            if first_match_pos < 100:
                score += 100
            elif first_match_pos < 500:
                score += 50
        
        if score > 0:
            # Normalize by text length to favor concise matches
            normalized_score = score / (1 + len(text) / 1000)
            results.append((item, normalized_score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:top_k]
    
    if results:
        print(f"✓ Found {len(results)} keyword matches")
    else:
        print(f"✗ No matches found for '{query}'")
    
    return results


# =====================================================================
# SESSION MANAGEMENT
# =====================================================================

def load_progress():
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def save_session_log(question_id, chat_history):
    path = SESSIONS_DIR / f"{question_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "id": question_id,
            "history": chat_history,
            "updated": datetime.now(timezone.utc).isoformat()
        }, f, indent=2, ensure_ascii=False)

def load_session_log(question_id):
    path = SESSIONS_DIR / f"{question_id}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("history", [])
        except:
            return []
    return []


# =====================================================================
# GEMINI AI
# =====================================================================

def ask_gemini_supervisor(question_text, user_message, chat_history_str, hint_level=1):
    """Enhanced AI supervisor with better prompts"""
    if not MODEL_AVAILABLE:
        hints = {
            1: "💡 Think about the fundamental concept here. What's the core idea?",
            2: "💡 Try breaking this into steps: 1) What do you know? 2) What do you need? 3) How do you connect them?",
            3: "💡 Here's a structured approach: Start with definitions, then build up to the solution step by step."
        }
        return hints.get(hint_level, "Set GEMINI_API_KEY to enable AI hints.")
    
    try:
        system_prompt = """You are an exceptional Cambridge Computer Science supervisor.
        
Your teaching philosophy:
- Socratic method: Guide students to discover answers themselves
- Be encouraging and supportive
- Use analogies and examples
- Ask probing questions
- Never give away the full answer unless explicitly requested with /show_answer

Student level: Cambridge Computer Science Tripos
Tone: Friendly but academically rigorous"""

        level_instructions = {
            1: "Give a gentle conceptual nudge. Ask a leading question. 1-2 sentences max.",
            2: "Outline the approach without calculations. What steps should they take?",
            3: "Provide a detailed roadmap with intermediate checkpoints, but no final answer."
        }
        
        instruct = level_instructions.get(hint_level, level_instructions[1])
        
        prompt = f"""{system_prompt}

QUESTION:
{question_text[:1000]}

CONVERSATION SO FAR:
{chat_history_str[-500:]}

INSTRUCTION: {instruct}

Student says: {user_message}

Your response:"""

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        if hasattr(response, "text"):
            return response.text.strip()
        return str(response)
        
    except Exception as e:
        print(f"Gemini error: {e}")
        return f"⚠ AI Error: {str(e)}"


# =====================================================================
# PDF LOADING THREAD (for responsiveness)
# =====================================================================

class PDFLoaderThread(QThread):
    finished = pyqtSignal(object, list, list)  # pixmap, page_positions, page_heights
    error = pyqtSignal(str)
    
    def __init__(self, pdf_path, target_page):
        super().__init__()
        self.pdf_path = pdf_path
        self.target_page = target_page
    
    def run(self):
        try:
            from PIL import Image
            
            # Check cache first
            cache_file = CACHE_DIR / f"{self.pdf_path.stem}_combined.png"
            cache_meta = CACHE_DIR / f"{self.pdf_path.stem}_meta.json"
            
            if cache_file.exists() and cache_meta.exists():
                # Load from cache
                with open(cache_meta) as f:
                    meta = json.load(f)
                pixmap = QPixmap(str(cache_file))
                self.finished.emit(pixmap, meta['positions'], meta['heights'])
                return
            
            # Convert PDF
            images = convert_from_path(str(self.pdf_path), dpi=150)
            
            if not images:
                self.error.emit("Could not render PDF")
                return
            
            # Combine pages
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            
            combined = Image.new('RGB', (max_width, total_height), 'white')
            
            page_positions = []
            page_heights = []
            y_offset = 0
            
            for img in images:
                page_positions.append(y_offset)
                page_heights.append(img.height)
                combined.paste(img, (0, y_offset))
                y_offset += img.height
            
            # Save to cache
            combined.save(cache_file, 'PNG', optimize=True)
            with open(cache_meta, 'w') as f:
                json.dump({'positions': page_positions, 'heights': page_heights}, f)
            
            # Convert to QPixmap
            img_byte_arr = io.BytesIO()
            combined.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            qimg = QImage.fromData(img_byte_arr.getvalue())
            pixmap = QPixmap.fromImage(qimg)
            
            self.finished.emit(pixmap, page_positions, page_heights)
            
        except Exception as e:
            self.error.emit(str(e))


# =====================================================================
# MAIN GUI
# =====================================================================

class TriposTutorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriposTutor — Ultimate Edition")
        self.resize(1600, 1000)
        
        self.setStyleSheet("""
            QWidget {
                background: #0f172a;
                color: #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            }
            QListWidget {
                background: #1e293b;
                border: 2px solid #334155;
                border-radius: 12px;
                padding: 8px;
                color: #e2e8f0;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 12px;
                border-radius: 8px;
                margin: 4px 0;
            }
            QListWidget::item:hover {
                background: #334155;
            }
            QListWidget::item:selected {
                background: #3b82f6;
                color: white;
            }
            QTextEdit {
                background: #1e293b;
                border: 2px solid #334155;
                border-radius: 12px;
                padding: 12px;
                color: #e2e8f0;
                font-size: 13px;
            }
            QLineEdit {
                background: #1e293b;
                border: 2px solid #334155;
                border-radius: 10px;
                padding: 12px;
                color: #e2e8f0;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #3b82f6;
            }
            QPushButton {
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #2563eb;
            }
            QPushButton:pressed {
                background: #1d4ed8;
            }
            QPushButton#secondary {
                background: #64748b;
            }
            QPushButton#secondary:hover {
                background: #475569;
            }
            QLabel {
                color: #e2e8f0;
            }
            QScrollArea {
                background: #1e293b;
                border: 2px solid #334155;
                border-radius: 12px;
            }
            QProgressBar {
                border: 2px solid #334155;
                border-radius: 8px;
                background: #1e293b;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 6px;
            }
        """)
        
        # Data
        self.db = build_or_load_db()
        if USE_EMBEDDINGS:
            try:
                build_embeddings_if_needed(self.db)
            except Exception as e:
                print(f"Embeddings error: {e}")
        
        self.progress = load_progress()
        self.current_topic = None
        self.current_matches = []
        self.current_index = 0
        self.current_question = None
        self.chat_history = []
        self.hint_level = 1
        self.pdf_loader = None
        self.page_positions = []
        self.page_heights = []
        
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # Header
        header = QLabel("🎓 TriposTutor — Ultimate Edition")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #3b82f6; padding: 12px;")
        main_layout.addWidget(header)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # === LEFT PANEL ===
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
        results_label.setStyleSheet("margin-top: 12px;")
        left_layout.addWidget(results_label)
        
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_question_selected)
        left_layout.addWidget(self.list_widget)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.rebuild_btn = QPushButton("🔄")
        self.rebuild_btn.setObjectName("secondary")
        self.rebuild_btn.setToolTip("Rebuild Database")
        self.rebuild_btn.clicked.connect(self.on_rebuild)
        btn_layout.addWidget(self.rebuild_btn)
        
        self.import_btn = QPushButton("📁")
        self.import_btn.setObjectName("secondary")
        self.import_btn.setToolTip("Import PDFs")
        self.import_btn.clicked.connect(self.on_import)
        btn_layout.addWidget(self.import_btn)
        
        left_layout.addLayout(btn_layout)
        
        splitter.addWidget(left_widget)
        
        # === RIGHT PANEL ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(12)
        
        # PDF viewer
        pdf_label = QLabel("📄 Question")
        pdf_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(pdf_label)
        
        self.question_image = QLabel()
        self.question_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.question_image.setScaledContents(False)
        
        self.question_scroll = QScrollArea()
        self.question_scroll.setWidget(self.question_image)
        self.question_scroll.setWidgetResizable(False)
        self.question_scroll.setMinimumHeight(400)
        right_layout.addWidget(self.question_scroll, stretch=4)
        
        # Progress bar for PDF loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        right_layout.addWidget(self.progress_bar)
        
        # Question text (small)
        text_label = QLabel("📝 Extracted Text")
        text_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        right_layout.addWidget(text_label)
        
        self.question_box = QTextEdit()
        self.question_box.setReadOnly(True)
        self.question_box.setMaximumHeight(100)
        right_layout.addWidget(self.question_box)
        
        # Chat
        chat_label = QLabel("💬 Chat with AI Supervisor")
        chat_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(chat_label)
        
        self.chat_box = QTextEdit()
        self.chat_box.setReadOnly(True)
        right_layout.addWidget(self.chat_box, stretch=3)
        
        # Input controls
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
        
        # Status
        self.status = QLabel("✨ Ready! Search for a topic to begin.")
        self.status.setStyleSheet("background: #1e293b; padding: 12px; border-radius: 8px; border: 2px solid #334155;")
        main_layout.addWidget(self.status)
        
        self.display_system("👋 Welcome! Search for questions by topic, then chat with your AI supervisor for hints.")
    
    # === ACTIONS ===
    
    def on_search(self):
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
            title = f"{item['filename']} Q{item['qnum']} (Page {item['page']})"
            snippet = item['text'][:120].replace('\n', ' ')
            confidence = item.get('confidence', 'high')
            
            list_item = QListWidgetItem(f"{title}\n{snippet}...\n[{confidence} confidence, score: {score:.2f}]")
            self.list_widget.addItem(list_item)
        
        if matches:
            self.show_question(matches[0][0])
            self.status.setText(f"✅ Found {len(matches)} questions for '{topic}'")
            self.display_system(f"Found {len(matches)} matches. Showing top result.")
        else:
            self.status.setText(f"❌ No matches for '{topic}'")
            self.display_system(f"No questions found. Try different keywords.")
    
    def on_question_selected(self, item):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.current_matches):
            self.current_index = row
            self.show_question(self.current_matches[row][0])
    
    def show_question(self, item):
        self.current_question = item
        self.question_box.setPlainText(
            f"[{item['filename']}] Q{item['qnum']} (Page {item['page']})\n\n{item['text']}"
        )
        
        # Load chat history
        self.chat_history = load_session_log(item["id"])
        self.refresh_chat()
        
        # Load PDF with threading
        pdf_path = PDF_DIR / item['filename']
        if pdf_path.exists():
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.progress_bar.setFormat("Loading PDF...")
            
            self.pdf_loader = PDFLoaderThread(pdf_path, item['page'])
            self.pdf_loader.finished.connect(self.on_pdf_loaded)
            self.pdf_loader.error.connect(self.on_pdf_error)
            self.pdf_loader.start()
        else:
            self.question_image.setText("📄 PDF not found")
        
        self.hint_level = 1
        self.status.setText(f"📖 Question {item['qnum']} from {item['filename']}")
    
    def on_pdf_loaded(self, pixmap, positions, heights):
        self.progress_bar.setVisible(False)
        self.page_positions = positions
        self.page_heights = heights
        
        # Scale pixmap to fit width
        scaled = pixmap.scaledToWidth(800, Qt.TransformationMode.SmoothTransformation)
        self.question_image.setPixmap(scaled)
        
        # Scroll to target page
        if self.current_question and len(self.page_positions) > 0:
            target_page = self.current_question['page'] - 1
            if 0 <= target_page < len(self.page_positions):
                scroll_pos = int(self.page_positions[target_page] * 800 / pixmap.width())
                QTimer.singleShot(100, lambda: self.question_scroll.verticalScrollBar().setValue(scroll_pos))
    
    def on_pdf_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.question_image.setText(f"❌ Error loading PDF:\n{error_msg}")
    
    def on_send(self):
        msg = self.input_line.text().strip()
        if not msg:
            return
        
        if not self.current_question:
            QMessageBox.warning(self, "No Question", "Select a question first!")
            return
        
        self.input_line.clear()
        
        # Handle commands
        if msg.lower() == "/next":
            self.on_next()
            return
        elif msg.lower() == "/skip":
            self.on_next()
            return
        elif msg.lower() == "/hint":
            self.on_hint()
            return
        elif msg.lower() == "/show_answer":
            self.show_full_answer()
            return
        elif msg.lower() == "/clear":
            self.chat_history = []
            self.refresh_chat()
            save_session_log(self.current_question["id"], self.chat_history)
            return
        
        # User message
        self.chat_history.append({"role": "user", "content": msg})
        self.display_user(msg)
        
        # Get AI response
        chat_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history[-10:]])
        response = ask_gemini_supervisor(
            self.current_question['text'],
            msg,
            chat_str,
            self.hint_level
        )
        
        self.chat_history.append({"role": "assistant", "content": response})
        self.display_ai(response)
        
        # Save session
        save_session_log(self.current_question["id"], self.chat_history)
    
    def on_hint(self):
        if not self.current_question:
            QMessageBox.warning(self, "No Question", "Select a question first!")
            return
        
        self.hint_level = min(self.hint_level + 1, 3)
        
        chat_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history[-10:]])
        response = ask_gemini_supervisor(
            self.current_question['text'],
            f"I need a hint (level {self.hint_level})",
            chat_str,
            self.hint_level
        )
        
        self.chat_history.append({"role": "system", "content": f"[Hint Level {self.hint_level}]"})
        self.chat_history.append({"role": "assistant", "content": response})
        
        self.display_system(f"[Hint Level {self.hint_level}/3]")
        self.display_ai(response)
        
        save_session_log(self.current_question["id"], self.chat_history)
    
    def show_full_answer(self):
        if not self.current_question:
            return
        
        chat_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history[-10:]])
        
        prompt = f"""Provide a COMPLETE, detailed solution to this question:

{self.current_question['text']}

Previous discussion:
{chat_str}

Now give the full answer with all steps, calculations, and explanations."""
        
        try:
            if MODEL_AVAILABLE:
                model = genai.GenerativeModel(GEMINI_MODEL)
                response = model.generate_content(prompt)
                answer = response.text if hasattr(response, 'text') else str(response)
            else:
                answer = "Set GEMINI_API_KEY to get full solutions."
        except Exception as e:
            answer = f"Error: {e}"
        
        self.chat_history.append({"role": "system", "content": "[FULL SOLUTION]"})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        self.display_system("[📖 FULL SOLUTION]")
        self.display_ai(answer)
        
        save_session_log(self.current_question["id"], self.chat_history)
    
    def on_next(self):
        if not self.current_matches:
            QMessageBox.information(self, "No Results", "Search for questions first!")
            return
        
        self.current_index = (self.current_index + 1) % len(self.current_matches)
        self.show_question(self.current_matches[self.current_index][0])
        self.list_widget.setCurrentRow(self.current_index)
    
    def on_mark_solved(self):
        if not self.current_question:
            QMessageBox.warning(self, "No Question", "Select a question first!")
            return
        
        qid = self.current_question["id"]
        self.progress[qid] = {
            "solved": True,
            "date": datetime.now(timezone.utc).isoformat()
        }
        save_progress(self.progress)
        
        self.display_system("✅ Marked as solved! Moving to next question...")
        QTimer.singleShot(1000, self.on_next)
    
    def on_rebuild(self):
        reply = QMessageBox.question(
            self,
            "Rebuild Database",
            "This will re-extract all questions from PDFs. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.status.setText("🔨 Rebuilding database...")
            QApplication.processEvents()
            
            self.db = build_or_load_db(force_rebuild=True)
            
            if USE_EMBEDDINGS:
                try:
                    build_embeddings_if_needed(self.db)
                except Exception as e:
                    print(f"Embeddings error: {e}")
            
            self.current_matches = []
            self.list_widget.clear()
            self.status.setText(f"✅ Database rebuilt: {len(self.db)} questions")
    
    def on_import(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF files",
            str(PDF_DIR),
            "PDF Files (*.pdf)"
        )
        
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
            QMessageBox.information(
                self,
                "Import Complete",
                f"Imported {imported} PDF(s). Rebuilding database..."
            )
            self.on_rebuild()
        else:
            QMessageBox.information(self, "Import", "No new files to import.")
    
    # === CHAT DISPLAY ===
    
    def display_system(self, text):
        self.chat_box.append(f'<div style="color: #94a3b8; font-style: italic; margin: 8px 0;">{text}</div>')
        self.chat_box.verticalScrollBar().setValue(
            self.chat_box.verticalScrollBar().maximum()
        )
    
    def display_user(self, text):
        self.chat_box.append(
            f'<div style="background: #3b82f6; color: white; padding: 12px; '
            f'border-radius: 12px; margin: 8px 0; max-width: 80%;">'
            f'<b>You:</b> {text}</div>'
        )
        self.chat_box.verticalScrollBar().setValue(
            self.chat_box.verticalScrollBar().maximum()
        )
    
    def display_ai(self, text):
        # Convert markdown-style formatting
        text = text.replace('**', '<b>').replace('**', '</b>')
        text = text.replace('\n', '<br>')
        
        self.chat_box.append(
            f'<div style="background: #1e293b; border: 2px solid #334155; '
            f'padding: 12px; border-radius: 12px; margin: 8px 0;">'
            f'<b style="color: #3b82f6;">🧑‍🏫 Supervisor:</b><br>{text}</div>'
        )
        self.chat_box.verticalScrollBar().setValue(
            self.chat_box.verticalScrollBar().maximum()
        )
    
    def refresh_chat(self):
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


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("TriposTutor")
    app.setApplicationDisplayName("TriposTutor - Ultimate Edition")
    
    # Check for required dependencies
    missing = []
    if not PdfReader:
        missing.append("pypdf")
    
    try:
        import pdf2image
    except:
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
        return
    
    # Check for PDFs
    if not any(PDF_DIR.glob("*.pdf")):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("No PDFs Found")
        msg.setText(f"No PDF files found in:\n{PDF_DIR}")
        msg.setInformativeText("Please add PDF files to this directory and restart.")
        msg.exec()
    
    # Launch GUI
    window = TriposTutorGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()