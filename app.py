import os
import asyncio
import logging
import requests
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants, LiveKitAPI
from livekit.api.agent_dispatch_service import CreateAgentDispatchRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Explicitly set the template and static folder paths to avoid TemplateNotFound errors
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frontend', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frontend', 'static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# --- Configuration ---
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
MESSAGE_LIMIT = 20
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"

SYSTEM_INSTRUCTION = """You are Triage, a calm and knowledgeable healthcare triage assistant powered by Sarvam AI.
Your role is to guide users who may be dealing with Diabetes, Anaemia, Hypertension, or Pneumonia.

Tone & Style:
- Be calm, clear, and humble. Speak like a composed, caring doctor — not an excited friend.
- Never use dramatic openers like "That's a great question!", "Oh wow!", "Absolutely!", or "Of course!".
- Get straight to the point. Start your response directly with the relevant information.
- Use simple language. Avoid heavy medical jargon.
- Keep responses structured — use short paragraphs or bullet points where helpful.

Your responsibilities:
1. Ask the user about their symptoms, duration, age, and any known conditions to understand their situation.
2. Provide clear, practical guidance on managing or understanding their condition.
3. Suggest daily habits, dietary changes, yoga poses, breathing exercises, and lifestyle routines suited to their condition.
4. If symptoms sound severe (e.g. chest pain, breathlessness, fainting, very high BP or sugar), calmly advise them to visit a nearby clinic or call emergency services (112 / 911) immediately.
5. For mild to moderate concerns, give actionable home management advice and follow-up questions.

Scope:
- You only assist with Diabetes, Anaemia, Hypertension, and Pneumonia.
- If asked about anything else, politely say you specialise in these four conditions and suggest they consult a doctor.

Disclaimer rule:
- Add "Note: I am an AI assistant, not a doctor. Please consult a healthcare professional for personalised medical advice." ONLY ONCE — at the very end of your first response. Never repeat it."""

REPORT_ANALYSIS_INSTRUCTION = """You are Triage, a warm healthcare companion powered by Sarvam AI.
Analyse the medical report below and provide a clear, friendly, easy-to-understand summary.

Structure your response as:
1. What the report shows (in simple language, no jargon)
2. Areas of concern related to Diabetes, Anaemia, Hypertension, or Pneumonia
3. Positive findings (what looks good)
4. Practical next steps — daily habits, diet, yoga, or lifestyle changes tailored to the findings
5. When to see a doctor

Tone: warm, encouraging, and humble. Avoid alarming language. Speak like a caring friend explaining results.
Add a brief disclaimer only at the very end."""

DISCLAIMER = "Disclaimer: I am an AI, not a doctor. Please consult a healthcare professional for personalised medical advice."

# --- Helpers ---
def _normalize_reply(reply: str) -> str:
    clean_reply = (reply or "").strip()
    if not clean_reply:
        clean_reply = "I'm here to help! Could you share a bit more about what you're experiencing?"
    return clean_reply


def _is_relevant_query(message: str) -> bool:
    lowered = message.lower()
    keywords = [
        "diabetes", "diabetic", "blood sugar", "glucose", "insulin", "hba1c",
        "anaemia", "anemia", "haemoglobin", "hemoglobin", "iron", "rbc",
        "hypertension", "blood pressure", "bp", "systolic", "diastolic",
        "pneumonia", "lung", "chest infection", "respiratory", "cough", "breathing",
        "yoga", "diet", "nutrition", "exercise", "habit", "lifestyle", "food",
        "report", "symptoms", "diagnosis", "triage", "medical", "health",
        "tired", "fatigue", "dizzy", "headache", "fever", "sugar", "weight"
    ]
    return any(kw in lowered for kw in keywords)


def _call_sarvam_llm(system_prompt: str, user_message: str) -> tuple[str, int]:
    """Generic Sarvam LLM call."""
    if not SARVAM_API_KEY:
        return "API key not configured. Please set SARVAM_API_KEY in .env", 500
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SARVAM_API_KEY}"
        }
        payload = {
            "model": "sarvam-30b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }
        response = requests.post(SARVAM_CHAT_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _normalize_reply(content), 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Sarvam API Error: {str(e)}")
        return f"Sarvam API error: {str(e)}", 500
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return "An unexpected error occurred.", 500


def call_sarvam_ai(message: str) -> tuple[str, int]:
    if not _is_relevant_query(message):
        return (
            "I specialise in Diabetes, Anaemia, Hypertension, and Pneumonia. "
            "For other health concerns, I'd recommend reaching out to a doctor who can give you the right guidance. "
            "Is there anything related to these four conditions I can help you with?"
        ), 200
    return _call_sarvam_llm(SYSTEM_INSTRUCTION, message)


def analyze_report(report_text: str) -> tuple[str, int]:
    return _call_sarvam_llm(REPORT_ANALYSIS_INSTRUCTION, f"Medical Report:\n{report_text}")

# --- Routes ---
@app.route('/')
def index():
    if 'msg_count' not in session:
        session['msg_count'] = 0
    return render_template('chat.html',
                           limit=MESSAGE_LIMIT,
                           count=session.get('msg_count', 0),
                           livekit_url=LIVEKIT_URL or "")


@app.route('/api/chat', methods=['POST'])
def chat_api():
    msg_count = session.get('msg_count', 0)
    if msg_count >= MESSAGE_LIMIT:
        return jsonify({"error": "Message limit reached. Please refresh to restart your session."}), 403

    data = request.get_json(silent=True) or {}
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    reply, status_code = call_sarvam_ai(user_msg)
    if status_code == 200:
        session['msg_count'] = msg_count + 1
        return jsonify({
            "reply": reply,
            "count": session['msg_count'],
            "limit": MESSAGE_LIMIT
        })

    return jsonify({"error": reply}), status_code


@app.route('/api/analyze-report', methods=['POST'])
def analyze_report_api():
    """Accept a medical report file or text and return LLM analysis."""
    report_text = ""

    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = file.filename.lower()
        try:
            if filename.endswith('.txt'):
                report_text = file.read().decode('utf-8', errors='ignore')
            elif filename.endswith('.pdf'):
                try:
                    import pdfplumber
                    with pdfplumber.open(file) as pdf:
                        report_text = "\n".join(
                            page.extract_text() or "" for page in pdf.pages
                        )
                except ImportError:
                    return jsonify({"error": "PDF support requires pdfplumber. Run: pip install pdfplumber"}), 500
            else:
                return jsonify({"error": "Unsupported file type. Upload a .txt or .pdf file."}), 400
        except Exception as e:
            logger.error(f"File read error: {e}")
            return jsonify({"error": "Failed to read file."}), 500

    # Handle raw text
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        report_text = data.get('report_text', '').strip()
    else:
        return jsonify({"error": "Send a file or JSON with report_text"}), 400

    if not report_text.strip():
        return jsonify({"error": "Report is empty or could not be read."}), 400

    reply, status_code = analyze_report(report_text)
    if status_code == 200:
        return jsonify({"analysis": reply})

    return jsonify({"error": reply}), status_code


@app.route('/api/livekit-token', methods=['POST'])
def livekit_token():
    """Generate a LiveKit access token and dispatch the voice agent to the room."""
    if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL]):
        return jsonify({"error": "LiveKit not configured"}), 500

    data = request.get_json(silent=True) or {}
    identity = data.get('identity', 'user')
    room_name = data.get('room', 'healthcare-voice__en-IN__default')

    try:
        # Generate user token
        token = (
            AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        # Dispatch the voice agent to the room
        async def _dispatch():
            async with LiveKitAPI(
                url=LIVEKIT_URL,
                api_key=LIVEKIT_API_KEY,
                api_secret=LIVEKIT_API_SECRET,
            ) as lk:
                await lk.agent_dispatch.create_dispatch(
                    CreateAgentDispatchRequest(
                        agent_name="healthcare-triage-voice",
                        room=room_name,
                    )
                )

        asyncio.run(_dispatch())
        logger.info(f"Agent dispatched to room: {room_name}")

        return jsonify({"token": token, "url": LIVEKIT_URL, "room": room_name})

    except Exception as e:
        logger.error(f"LiveKit token/dispatch error: {e}")
        # Still return the token even if dispatch fails — agent may auto-join
        try:
            return jsonify({"token": token, "url": LIVEKIT_URL, "room": room_name, "dispatch_warning": str(e)})
        except Exception:
            return jsonify({"error": f"Failed to generate token: {e}"}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    logger.info(f"Starting Triage on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
