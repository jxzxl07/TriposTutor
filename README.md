# Tripos Tutor
Weekend project using googles genai 

Idea: TriposTutor
- Tripos past paper questions chatbot
- User enters prompt about a certain CST topic
- Chatbot replies with questions based on that topic.
- User can ask questions about solving each question.
- Chatbot helps with solving the question, with hints rather than solution.
- Have chatbot remember your progress through each topic (completed, needed hint, skipped).
- Difficulty field for each question.
- Explain concepts before attempting the question.
- Multistep hint system (nudge, hint, partial solution, full solution)

- GUI via PyQt6
- Model via gemini-1.5-flash
- JSON data storage
- PDF parsing


TriposTutor is an AI-powered revision app that helps Cambridge Computer Science students practise Tripos questions interactively.
It extracts past-paper questions from PDFs and acts as a supervisor chatbot — giving gentle, tiered hints instead of full solutions.
Built entirely in one Python file using PyQt6 for the GUI and Google Gemini for AI hints.

🚀 Features
🧩 Automatic question extraction from Tripos PDFs (e.g. 2021_1.pdf, 2022_2.pdf)
🔍 Topic search — find questions by typing “recursion”, “graphs”, etc.
💬 Supervisor-style chat with Gemini (tiered hint levels)
🎓 Hints, not answers — /hint, /next, /skip, /show_answer
💾 Session saving — chat logs, progress, solved status
🧠 Optional semantic search using Sentence Transformers + FAISS
🖥️ Clean PyQt6 interface, all contained in a single file
