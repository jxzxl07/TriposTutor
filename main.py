from pdf2image import convert_from_path
import sys, os, re, io, json, traceback
from pathlib import Path
from datetime import datetime, timezone

# MUST BE FIRST - Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# UI
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QListWidget, QLabel, QFileDialog, QMessageBox, QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QPixmap, QImage

# PDF parsing
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# Gemini & env
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBeogS9NZ7ncr6XX8TfcaQHVtDVfIVcGzE")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = "gemini-2.5-flash"
        MODEL_AVAILABLE = True
    else:
        MODEL_AVAILABLE = False
except Exception as e:
    print(f"Gemini setup error: {e}")
    genai = None
    MODEL_AVAILABLE = False

# Optional semantic search
USE_EMBEDDINGS = False
EMBEDDER = None
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
    USE_EMBEDDINGS = True
except Exception as e:
    print(f"Embeddings disabled: {e}")
    USE_EMBEDDINGS = False

# ----------------------
# Files & paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PDF_DIR = BASE_DIR / "tripos_data"
DB_PATH = DATA_DIR / "tripos_questions.json"
PROGRESS_PATH = DATA_DIR / "progress.json"
SESSIONS_DIR = DATA_DIR / "sessions"

# ensure directories
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# ----------------------
# Utility: PDF -> question extraction
def extract_questions_with_pages(pdf_path):
    """
    Extract questions from PDF AND track which page they're on.
    Returns list of questions with page numbers.
    """
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. Install with `pip install pypdf`.")
    
    reader = PdfReader(str(pdf_path))
    questions = []
    current_question = None
    
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception:
            text = ""
        
        if not text:
            continue
        
        # Look for question numbers on this page
        # Pattern matches: "1 ", "1. ", "2)", "10  " at start of line
        pattern = re.compile(r'(?m)^(?:\s*)(\d{1,2})(?:[.)\s]+)(.*)$')
        
        lines = text.split('\n')
        page_text = []
        
        for line in lines:
            match = pattern.match(line)
            if match:
                # Found a new question number
                qnum = int(match.group(1))
                
                # Save previous question if exists
                if current_question:
                    questions.append(current_question)
                
                # Start new question
                current_question = {
                    "qnum": qnum,
                    "page": page_num,
                    "text": match.group(2).strip() if match.group(2) else ""
                }
            elif current_question:
                # Continue current question text
                current_question["text"] += " " + line.strip()
        
    # Don't forget the last question
    if current_question:
        questions.append(current_question)
    
    # Fallback: if no questions found, treat each page as a question
    if not questions:
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    questions.append({
                        "qnum": page_num,
                        "page": page_num,
                        "text": text.strip()
                    })
            except Exception:
                pass
    
    return questions


def build_or_load_db(force_rebuild=False):
    """
    Build question DB from PDFs in PDF_DIR, save to DB_PATH.
    Now includes page numbers for accurate display.
    """
    if DB_PATH.exists() and not force_rebuild:
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"Loaded {len(data)} questions from {DB_PATH}")
                return data
        except Exception:
            print("Failed to load existing DB; rebuilding.")
    
    # Rebuild
    all_questions = []
    
    for fn in sorted(os.listdir(PDF_DIR)):
        if not fn.lower().endswith(".pdf"):
            continue
        
        path = PDF_DIR / fn
        print(f"Parsing {fn} ...")
        
        try:
            qs = extract_questions_with_pages(path)
        except Exception as e:
            print(f"Error parsing {fn}: {e}")
            traceback.print_exc()
            qs = [{"qnum": 1, "page": 1, "text": ""}]
        
        # Infer year and paper from filename pattern like 2021_1
        m = re.match(r'(\d{4})[_\-]?(\d)', fn)
        year = None
        paper = None
        if m:
            year = int(m.group(1))
            paper = int(m.group(2))
        
        for q in qs:
            # Store complete question with page number
            item = {
                "id": f"{fn}__q{q['qnum']}",
                "filename": fn,
                "year": year,
                "paper": paper,
                "qnum": q['qnum'],
                "page": q.get('page', q['qnum']),  # CRITICAL: store actual page
                "text": q['text'],
                "topics": [],
                "difficulty": None
            }
            all_questions.append(item)
    
    # Save DB
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_questions)} questions to {DB_PATH}")
    return all_questions

# ----------------------
# Optional: embeddings for better search
embeddings_index = None

def build_embeddings_if_needed(db):
    global embeddings_index
    if not USE_EMBEDDINGS or EMBEDDER is None:
        return
    texts = [q["text"] for q in db]
    embs = EMBEDDER.encode(texts, show_progress_bar=False)
    embeddings_index = {"ids": [q["id"] for q in db], "embs": embs}
    print("Built embeddings for DB.")

# ----------------------
# IMPROVED Search function with strict filtering
def search_questions(db, topic_query, top_k=10):
    """
    Return ranked list of matching questions for a free-text topic_query.
    Uses STRICT matching with minimum threshold to avoid irrelevant results.
    """
    q = topic_query.strip().lower()
    if not q:
        return []
    
    results = []
    
    if USE_EMBEDDINGS and EMBEDDER is not None and embeddings_index is not None:
        # Semantic search with STRICT threshold
        query_emb = EMBEDDER.encode([q])[0]
        embs = embeddings_index["embs"]
        ids = embeddings_index["ids"]
        import numpy as np
        
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        
        sims = [cos(query_emb, e) for e in embs]
        
        # STRICT FILTERING: Only include results above threshold
        SIMILARITY_THRESHOLD = 0.35  # Increase for stricter matching
        ranked = [(ids[i], sims[i]) for i in range(len(sims)) if sims[i] >= SIMILARITY_THRESHOLD]
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]
        
        id_to_q = {item["id"]: item for item in db}
        for idx, score in ranked:
            results.append((id_to_q[idx], float(score)))
        
        print(f"Found {len(results)} semantic matches for '{topic_query}'")
        return results
    
    # Fallback: STRICT keyword matching
    query_tokens = set(q.split())
    
    for item in db:
        text = (item.get("text") or "").lower()
        filename = (item.get("filename") or "").lower()
        
        # Count matching tokens
        text_tokens = set(text.split())
        matching_tokens = query_tokens.intersection(text_tokens)
        
        # STRICT: Require at least 50% of query tokens to match
        match_ratio = len(matching_tokens) / len(query_tokens) if query_tokens else 0
        
        if match_ratio >= 0.5:  # At least half the search terms must match
            score = 0
            # Boost for exact phrase match
            if q in text:
                score += 100
            # Boost for token matches
            score += len(matching_tokens) * 10
            # Boost if near start of text
            if any(token in text[:300] for token in matching_tokens):
                score += 20
            
            results.append((item, score))
    
    # Sort and filter
    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:top_k]
    
    print(f"Found {len(results)} keyword matches for '{topic_query}'")
    return results

# ----------------------
# Session & progress management
def load_progress():
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def save_session_log(question_id, chat_history):
    path = SESSIONS_DIR / f"{question_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"id": question_id, "history": chat_history, "updated": datetime.now(timezone.utc).isoformat()}, f, indent=2, ensure_ascii=False)

def load_session_log(question_id):
    path = SESSIONS_DIR / f"{question_id}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("history", [])
        except Exception:
            return []
    return []

# ----------------------
# Gemini integration helpers
def ask_gemini_supervisor(question_text, user_message, chat_history_str, hint_level=1):
    """
    Ask Gemini for a hint/explanation.
    hint_level:
      1 -> gentle nudge / concept hint
      2 -> approach / steps
      3 -> partial outline (still not full answer)
    """
    if not MODEL_AVAILABLE:
        # fallback stub
        if hint_level == 1:
            return "Hint (lvl1): Try to identify the base case and the recursive step. What reduces at each recursion?"
        if hint_level == 2:
            return "Hint (lvl2): Consider writing the recurrence relation for the problem and simplify it. Which invariant holds?"
        if hint_level == 3:
            return "Hint (lvl3): Outline: 1) Identify base case 2) Show recursive reduction 3) Prove termination. Ask /show_answer to reveal full details."
        return "No model available. Set GEMINI_API_KEY in .env to enable AI hints."
    try:
        system_prompt = (
            "You are a helpful Cambridge Computer Science supervisor. "
            "A student is working on a Tripos question. Provide educational, scaffolded hints. "
            "Do NOT give the full solution unless the student explicitly asks for it (by writing /show_answer). "
            "Be concise and ask the student a question to prompt thinking. Use friendly, slightly formal tone."
        )
        # Tailor level instructions
        if hint_level == 1:
            instruct = "Give a short conceptual hint that nudges the student without revealing approach steps. Keep it to 1-2 sentences."
        elif hint_level == 2:
            instruct = "Provide a helpful approach-level hint that outlines the steps to start solving the question. Still avoid full computations."
        else:
            instruct = "Provide a partial outline with steps and intermediate suggestions but avoid full final results. Make it clear it's a partial hint."

        prompt = f"""{system_prompt}

QUESTION:
{question_text}

CONTEXT (chat so far):
{chat_history_str}

INSTRUCTION: {instruct}

Student: {user_message}
Supervisor:"""

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        text = ""
        if hasattr(response, "text"):
            text = response.text
        elif isinstance(response, str):
            text = response
        else:
            try:
                text = response["candidates"][0]["content"][0]["text"]
            except Exception:
                text = str(response)
        return text.strip()
    except Exception as e:
        print("Gemini error:", e)
        traceback.print_exc()
        return "AI Error: " + str(e)

# ----------------------
# GUI implementation
class TriposTutorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriposTutor — AI Tripos Supervisor")
        self.resize(1400, 900)
        self.setStyleSheet("""
            QWidget { background: #f7f7fb; font-family: Arial, Helvetica; }
            QListWidget { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; color: #000000; padding: 4px; }
            QTextEdit#questionBox { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; color: #000000; }
            QTextEdit#chatBox { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; color: #000000; }
            QLineEdit { padding: 8px; border-radius: 8px; border: 1px solid #d1d5db; font-size: 14px; color: #000000; background: #ffffff; }
            QPushButton { padding: 10px 16px; border-radius: 8px; background: #2563eb; color: white; font-weight: bold; }
            QPushButton:hover { background: #1d4ed8; }
            QLabel#imageLabel { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
            QScrollArea { border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff; }
        """)
        
        # Data
        self.db = build_or_load_db()
        if USE_EMBEDDINGS:
            try:
                build_embeddings_if_needed(self.db)
            except Exception as e:
                print("Embeddings build failed:", e)
        self.progress = load_progress()
        
        # State
        self.current_topic = None
        self.current_matches = []
        self.current_index = 0
        self.current_question = None
        self.chat_history = []
        self.hint_level = 1

        # UI layout
        main_layout = QVBoxLayout(self)

        header = QLabel("TriposTutor — Chat-based Tripos Practice")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setStyleSheet("color: #1e293b; padding: 10px;")
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: search & list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6,6,6,6)
        
        search_label = QLabel("Search for questions:")
        search_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(search_label)
        
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("e.g., graphs, algorithms, recursion...")
        self.topic_input.returnPressed.connect(self.on_topic_entered)
        left_layout.addWidget(self.topic_input)

        self.search_btn = QPushButton("🔍 Find Questions")
        self.search_btn.clicked.connect(self.on_topic_entered)
        left_layout.addWidget(self.search_btn)

        results_label = QLabel("Results:")
        results_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        results_label.setStyleSheet("margin-top: 10px;")
        left_layout.addWidget(results_label)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_question_selected)
        left_layout.addWidget(self.list_widget)

        # Buttons
        left_btns = QHBoxLayout()
        self.rebuild_btn = QPushButton("🔄 Rebuild")
        self.rebuild_btn.clicked.connect(self.on_rebuild_db)
        left_btns.addWidget(self.rebuild_btn)

        self.import_btn = QPushButton("📁 Import")
        self.import_btn.clicked.connect(self.on_import_pdfs)
        left_btns.addWidget(self.import_btn)

        left_layout.addLayout(left_btns)

        splitter.addWidget(left_widget)

        # Right: question view + chat
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6,6,6,6)

        # PDF image in scroll area
        self.question_image = QLabel()
        self.question_image.setObjectName("imageLabel")
        self.question_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.question_image.setScaledContents(False)

        self.question_scroll = QScrollArea()
        self.question_scroll.setWidget(self.question_image)
        self.question_scroll.setWidgetResizable(False)  # Changed to False for proper scrolling
        self.question_scroll.setMinimumHeight(300)
        right_layout.addWidget(self.question_scroll, stretch=5)

        # Text version (optional, can be hidden)
        text_label = QLabel("Question Text:")
        text_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        right_layout.addWidget(text_label)

        self.question_box = QTextEdit()
        self.question_box.setObjectName("questionBox")
        self.question_box.setReadOnly(True)
        self.question_box.setFont(QFont("Arial", 11))
        self.question_box.setMaximumHeight(120)
        right_layout.addWidget(self.question_box, stretch=1)
        
        chat_label = QLabel("Chat:")
        chat_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        right_layout.addWidget(chat_label)
        
        self.chat_box = QTextEdit()
        self.chat_box.setObjectName("chatBox")
        self.chat_box.setReadOnly(True)
        self.chat_box.setFont(QFont("Arial", 11))
        right_layout.addWidget(self.chat_box, stretch=3)

        # Input + action buttons
        bottom_layout = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type your answer or ask for help... (Commands: /hint /next /skip /show_answer)")
        self.input_line.returnPressed.connect(self.on_send)
        bottom_layout.addWidget(self.input_line)

        self.send_btn = QPushButton("📤 Send")
        self.send_btn.clicked.connect(self.on_send)
        bottom_layout.addWidget(self.send_btn)

        self.hint_btn = QPushButton("💡 Hint")
        self.hint_btn.clicked.connect(self.on_hint)
        bottom_layout.addWidget(self.hint_btn)

        self.next_btn = QPushButton("⏭️ Next")
        self.next_btn.clicked.connect(self.on_next_question)
        bottom_layout.addWidget(self.next_btn)

        self.mark_done_btn = QPushButton("✅ Solved")
        self.mark_done_btn.clicked.connect(self.on_mark_done)
        bottom_layout.addWidget(self.mark_done_btn)

        right_layout.addLayout(bottom_layout)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

        # Status bar
        self.status = QLabel("Ready. Enter a topic to search for questions.")
        self.status.setStyleSheet("padding: 8px; background: #e0e7ff; border-radius: 4px; color: #1e293b;")
        main_layout.addWidget(self.status)

        # Initialize
        self.update_list([])
        self.display_system("👋 Welcome to TriposTutor! Enter a topic to begin. Use /hint for hints, /next to move on.")

    # ----------------- UI helpers -----------------
    def update_list(self, matches):
        self.list_widget.clear()
        for item, score in matches:
            title = f"{item.get('filename','')} Q{item.get('qnum','?')}"
            snippet = (item.get('text') or "")[:100].replace("\n", " ")
            self.list_widget.addItem(f"{title}\n{snippet}...")
        if not matches:
            self.list_widget.addItem("(No matches found - try different keywords)")

    def show_question(self, item):
        self.question_box.setPlainText(f"[{item.get('filename')}] Question {item.get('qnum')}\n\n{item.get('text')}")
        self.current_question = item
        self.show_question_image(item)
        hist = load_session_log(item["id"])
        self.chat_history = hist[:] if isinstance(hist, list) else []
        self.refresh_chat_box()
        self.status.setText(f"📄 Viewing: {item.get('filename')} Q{item.get('qnum')}")

    def show_question_image(self, item):
        """Display entire PDF and auto-scroll to the correct page"""
        try:
            pdf_filename = item.get('filename', '')
            pdf_path = PDF_DIR / pdf_filename
            
            if not pdf_path.exists():
                self.question_image.setText("PDF file not found")
                return
            
            # Get the page number for this question
            target_page = item.get('page', 1)
            
            print(f"Loading {pdf_filename}, scrolling to page {target_page}")
            
            # Convert ALL pages (so user can scroll through if needed)
            images = convert_from_path(str(pdf_path), dpi=150)
            
            if images:
                # Stack all pages vertically
                from PIL import Image
                
                total_height = sum(img.height for img in images)
                max_width = max(img.width for img in images)
                
                combined = Image.new('RGB', (max_width, total_height), 'white')
                
                # Track where each page starts (for scrolling)
                page_positions = []
                y_offset = 0
                
                for i, img in enumerate(images):
                    page_positions.append(y_offset)
                    combined.paste(img, (0, y_offset))
                    y_offset += img.height
                
                # Convert to QPixmap
                img_byte_arr = io.BytesIO()
                combined.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                qimg = QImage.fromData(img_byte_arr.getvalue())
                pixmap = QPixmap.fromImage(qimg)
                
                # Set the pixmap
                self.question_image.setPixmap(pixmap)
                self.question_image.adjustSize()
                
                # AUTO-SCROLL to the target page!
                if 0 < target_page <= len(page_positions):
                    target_y = page_positions[target_page - 1]  # Pages are 1-indexed
                    
                    # Use QTimer to scroll after the image is rendered
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(100, lambda: self.question_scroll.verticalScrollBar().setValue(target_y))
            else:
                self.question_image.setText("Could not render PDF pages")
                
        except Exception as e:
            print(f"Error displaying question image: {e}")
            traceback.print_exc()
            self.question_image.setText(f"Error loading PDF: {str(e)}")


    def display_user(self, text):
        self.chat_box.append(f"<div style='text-align:right; color:#0f172a; margin: 8px 0;'><b style='color:#2563eb;'>You:</b> {text}</div>")
        self.chat_box.moveCursor(QTextCursor.MoveOperation.End)



    def display_system(self, text):
        self.chat_box.append(f"<div style='text-align:left; color:#0f172a; margin: 8px 0;'><b style='color:#059669;'>TriposTutor:</b> {text}</div>")
        self.chat_box.moveCursor(QTextCursor.MoveOperation.End)

    def refresh_chat_box(self):
        self.chat_box.clear()
        for speaker, msg in self.chat_history:
            if speaker == "You":
                self.display_user(msg)
            else:
                self.display_system(msg)

    # ----------------- Actions -----------------
    def on_topic_entered(self):
        topic = self.topic_input.text().strip()
        if not topic:
            QMessageBox.information(self, "No topic", "Please type a topic to search for.")
            return
        self.current_topic = topic
        self.status.setText(f"🔍 Searching for '{topic}'...")
        QApplication.processEvents()  # Update UI
        
        matches = search_questions(self.db, topic, top_k=50)
        self.current_matches = matches
        self.current_index = 0
        self.update_list(matches)
        
        if matches:
            first = matches[0][0]
            self.show_question(first)
            self.display_system(f"Found {len(matches)} relevant questions about '{topic}'. Showing the first one.")
        else:
            self.status.setText(f"❌ No matches for '{topic}'")
            self.display_system(f"No questions found matching '{topic}'. Try different keywords or check spelling.")

    def on_question_selected(self, item_widget):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.current_matches):
            return
        qitem = self.current_matches[row][0]
        self.current_index = row
        self.show_question(qitem)
        self.display_system("Loaded selected question.")

    def on_send(self):
        text = self.input_line.text().strip()
        if not text:
            return
        
        if not self.current_question:
            self.display_system("⚠️ No question loaded. Search for a topic first!")
            self.input_line.clear()
            return
            
        self.display_user(text)
        self.chat_history.append(("You", text))
        self.input_line.clear()
        
        if text.startswith("/"):
            self.handle_command(text)
            return
        
        qtext = self.current_question.get("text", "")
        chat_str = "\n".join([f"{s}: {m}" for s,m in self.chat_history[-10:]])
        reply = ask_gemini_supervisor(qtext, text, chat_str, hint_level=self.hint_level)
        self.chat_history.append(("TriposTutor", reply))
        self.display_system(reply)
        save_session_log(self.current_question["id"], self.chat_history)

    def handle_command(self, cmd):
        cmd = cmd.strip().lower()
        if cmd.startswith("/hint"):
            self.on_hint()
        elif cmd.startswith("/next"):
            self.on_next_question()
        elif cmd.startswith("/skip"):
            self.on_next_question(skip_mark=True)
        elif cmd.startswith("/show_answer"):
            if not self.current_question:
                self.display_system("No question loaded.")
                return
            qtext = self.current_question.get("text", "")
            chat_str = "\n".join([f"{s}: {m}" for s,m in self.chat_history[-20:]])
            if not MODEL_AVAILABLE:
                ans = "Full answer requested but GEMINI_API_KEY not set."
            else:
                prompt = (
                    "You are a helpful Cambridge supervisor. The student requested the full solution. "
                    "Provide a clear, step-by-step solution with final answer."
                    f"\n\nQUESTION:\n{qtext}\n\nCONTEXT:\n{chat_str}\n\nSupervisor (full solution):"
                )
                try:
                    model = genai.GenerativeModel(GEMINI_MODEL)
                    response = model.generate_content(prompt)
                    ans = response.text if hasattr(response, "text") else str(response)
                except Exception as e:
                    ans = "AI error: " + str(e)
            self.chat_history.append(("TriposTutor", ans))
            self.display_system(ans)
            save_session_log(self.current_question["id"], self.chat_history)
        else:
            self.display_system("Unknown command. Available: /hint /next /skip /show_answer")

    def on_hint(self):
        if not self.current_question:
            self.display_system("No question loaded. Search for a topic first.")
            return
        self.hint_level = min(3, self.hint_level + 1)
        qtext = self.current_question.get("text", "")
        chat_str = "\n".join([f"{s}: {m}" for s,m in self.chat_history[-20:]])
        reply = ask_gemini_supervisor(qtext, "(user asked for hint)", chat_str, hint_level=self.hint_level)
        self.chat_history.append(("TriposTutor", reply))
        self.display_system(reply)
        save_session_log(self.current_question["id"], self.chat_history)

    def on_next_question(self, skip_mark=False):
        if not self.current_matches:
            self.display_system("No current matches. Search a topic first.")
            return
        self.current_index = (self.current_index + 1) if self.current_index+1 < len(self.current_matches) else 0
        next_item = self.current_matches[self.current_index][0]
        self.hint_level = 1
        self.show_question(next_item)
        self.display_system(f"⏭️ Moved to next question ({self.current_index+1}/{len(self.current_matches)}).")

    def on_mark_done(self):
        if not self.current_question:
            self.display_system("No question loaded.")
            return
        qid = self.current_question["id"]
        self.progress[qid] = {"status": "solved", "when": datetime.now(timezone.utc).isoformat()}
        save_progress(self.progress)
        self.display_system("✅ Marked question as solved. Great work!")

    def on_rebuild_db(self):
        ok = QMessageBox.question(self, "Rebuild DB", "This will re-parse all PDFs in tripos_data/ and rebuild the DB. Continue?",
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ok == QMessageBox.StandardButton.Yes:
            self.status.setText("🔄 Rebuilding database...")
            QApplication.processEvents()
            self.db = build_or_load_db(force_rebuild=True)
            if USE_EMBEDDINGS:
                try:
                    build_embeddings_if_needed(self.db)
                except Exception as e:
                    print("Embeddings error:", e)
            self.display_system("✅ Rebuilt database successfully.")
            self.status.setText("Ready.")
            self.update_list([])

    def on_import_pdfs(self):
        fns, _ = QFileDialog.getOpenFileNames(self, "Select PDF(s) to import", "", "PDF Files (*.pdf)")
        if not fns:
            return
        imported = 0
        for f in fns:
            try:
                dest = PDF_DIR / Path(f).name
                with open(f, "rb") as rf, open(dest, "wb") as wf:
                    wf.write(rf.read())
                imported += 1
            except Exception as e:
                print("Import error:", e)
        self.display_system(f"📁 Imported {imported} PDF(s). Click 'Rebuild DB' to parse them.")

# ----------------------
# Start app
def main():
    app = QApplication(sys.argv)
    window = TriposTutorGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # Ensure DB exists
    try:
        _ = build_or_load_db()
    except Exception as e:
        print("Error building/loading DB:", e)
    main()