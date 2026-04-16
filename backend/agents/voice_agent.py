from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import Agent, WorkerOptions, cli, llm
from livekit.agents.voice import AgentSession
from livekit.plugins import openai, sarvam, silero

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
VOICE_AGENT_NAME = "healthcare-triage-voice"
GREETING_PLAYBACK_DELAY_SECONDS = 0.6
ASSISTANT_EVENT_TOPIC = "healthcare-assistant"


async def _sarvam_run_with_mp3_output(self, output_emitter) -> None:
    request_id = sarvam.tts.utils.shortuuid()
    self._client_request_id = request_id
    self._server_request_id = None
    output_emitter.initialize(
        request_id=request_id,
        sample_rate=self._opts.speech_sample_rate,
        num_channels=1,
        mime_type="audio/mpeg",
        stream=True,
        frame_size_ms=50,
    )

    async def _tokenize_input() -> None:
        word_stream = None
        async for input_value in self._input_ch:
            if isinstance(input_value, str):
                if word_stream is None:
                    tokenizer_instance = (
                        self._opts.word_tokenizer
                        if self._opts.word_tokenizer is not None
                        else sarvam.tts.tokenize.basic.SentenceTokenizer()
                    )
                    word_stream = tokenizer_instance.stream()
                    self._segments_ch.send_nowait(word_stream)
                word_stream.push_text(input_value)
            elif isinstance(input_value, self._FlushSentinel):
                if word_stream:
                    word_stream.end_input()
                word_stream = None

        if word_stream is not None:
            word_stream.end_input()

        self._segments_ch.close()

    async def _process_segments() -> None:
        async for word_stream in self._segments_ch:
            await self._run_ws(word_stream, output_emitter)

    tasks = [
        asyncio.create_task(_tokenize_input()),
        asyncio.create_task(_process_segments()),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        await sarvam.tts.utils.aio.gracefully_cancel(*tasks)
        output_emitter.end_input()


sarvam.tts.SynthesizeStream._run = _sarvam_run_with_mp3_output

LANGUAGE_PROFILES = {
    "en-IN": {"label": "English", "instruction_name": "English"},
    "hi-IN": {"label": "Hindi", "instruction_name": "Hindi"},
    "bn-IN": {"label": "Bengali", "instruction_name": "Bengali"},
    "ta-IN": {"label": "Tamil", "instruction_name": "Tamil"},
    "te-IN": {"label": "Telugu", "instruction_name": "Telugu"},
    "kn-IN": {"label": "Kannada", "instruction_name": "Kannada"},
    "ml-IN": {"label": "Malayalam", "instruction_name": "Malayalam"},
    "mr-IN": {"label": "Marathi", "instruction_name": "Marathi"},
    "gu-IN": {"label": "Gujarati", "instruction_name": "Gujarati"},
    "pa-IN": {"label": "Punjabi", "instruction_name": "Punjabi"},
    "od-IN": {"label": "Odia", "instruction_name": "Odia"},
}

OPENING_GREETINGS = {
    "en-IN": "Hello {user_name}. I am your Healthcare Triage Assistant. To help me understand your health better, could you please upload your medical report if you have one? If not, we can still continue our conversation.",
    "hi-IN": "Namaste {user_name}. Main aapka Healthcare Triage Assistant hoon. Aapki sehat ko behtar samajhne ke liye, kya aap apni medical report upload kar sakte hain? Agar nahi, toh bhi hum baat jaari rakh sakte hain.",
    "mr-IN": "Namaskar {user_name}. Mi tumcha Healthcare Triage Assistant aahe. Tumcha arogya vishayi adhik mahiti milavnyasathi, krupaya tumchi medical report upload karu shakta ka? Nasel tari, amhi amcha samvad suru thevu shakto.",
}


def _require_env() -> None:
    missing = [
        name
        for name, value in {
            "SARVAM_API_KEY": SARVAM_API_KEY,
            "LIVEKIT_URL": LIVEKIT_URL,
            "LIVEKIT_API_KEY": LIVEKIT_API_KEY,
            "LIVEKIT_API_SECRET": LIVEKIT_API_SECRET,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def _parse_room_language(room_name: str | None) -> str:
    normalized = (room_name or "").strip()
    if not normalized.lower().startswith("healthcare-voice__"):
        return "en-IN"

    remainder = normalized[len("healthcare-voice__") :]
    language_code, _separator, _suffix = remainder.partition("__")
    return language_code if language_code in LANGUAGE_PROFILES else "en-IN"


def _safe_user_name(name: str | None, identity: str | None) -> str:
    candidate = (name or "").strip()
    if candidate:
        return candidate.split()[0].strip() or "Citizen"

    fallback = (identity or "").strip()
    return fallback.split()[0].strip() if fallback else "Citizen"


def _build_instructions(language_profile: dict[str, str]) -> str:
    return f"""You are Traiage, a calm and knowledgeable healthcare triage voice assistant powered by Sarvam AI.
Your role is to guide users dealing with Diabetes, Anaemia, Hypertension, or Pneumonia.

Tone & Style:
- Be calm, composed, and humble — like a caring doctor, not an excited friend.
- Never use dramatic openers like "That's a great question!", "Oh wow!", or "Absolutely!".
- Get straight to the point. Speak clearly and naturally for voice conversation.
- Use simple language. Keep sentences short and easy to follow when spoken aloud.

Your responsibilities:
1. Greet the user and ask them to share their symptoms or upload a medical report.
2. Ask follow-up questions one at a time — symptoms, duration, age, known conditions.
3. Provide practical guidance: daily habits, diet, yoga, breathing exercises suited to their condition.
4. If symptoms sound severe (chest pain, breathlessness, fainting, very high BP or sugar), calmly advise them to visit a nearby clinic or call 112 / 911 immediately.
5. Interpret uploaded medical reports in simple, friendly language highlighting findings for the 4 conditions.

Scope:
- You only assist with Diabetes, Anaemia, Hypertension, and Pneumonia.
- For anything else, politely say you specialise in these four and suggest they see a doctor.

Language: Respond strictly in {language_profile['instruction_name']}.
""".strip()


class HealthcareTriageVoiceAgent(Agent):
    def __init__(self, language_profile: dict[str, str], language_code: str) -> None:
        super().__init__(
            instructions=_build_instructions(language_profile),
            stt=sarvam.STT(
                model="saaras:v3",
                language=language_code,
                mode="transcribe",
                flush_signal=True,
            ),
            llm=openai.LLM(
                base_url="https://api.sarvam.ai/v1",
                api_key=SARVAM_API_KEY,
                model="sarvam-30b",
            ),
            tts=sarvam.TTS(
                model="bulbul:v3",
                target_language_code=language_code,
                speaker="priya",
            ),
        )

    @llm.function_tool(description="Saves the extracted medical insights in a structured JSON format.")
    def save_medical_insights(
        self,
        insights_json: str,
    ) -> str:
        """
        Call this function whenever you have extracted new insights from reports or user responses.
        The insights_json should be a stringified JSON containing keys like 'conditions', 'risks', 'dietary_advice', 'precautions', etc.
        """
        print(f"INTERNAL MEDICAL INSIGHTS: {insights_json}")
        return "Insights captured internally."


async def entrypoint(ctx) -> None:
    _require_env()
    await ctx.connect()
    primary_participant = await ctx.wait_for_participant()

    language_code = _parse_room_language(getattr(ctx.room, "name", ""))
    language_profile = LANGUAGE_PROFILES[language_code]
    user_name = _safe_user_name(
        getattr(primary_participant, "name", None),
        getattr(primary_participant, "identity", None),
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        turn_detection="stt",
        min_endpointing_delay=0.4,
        max_endpointing_delay=1.2,
        min_interruption_duration=0.2,
        false_interruption_timeout=1.2,
        resume_false_interruption=True,
    )

    async def _speak_completion(message: str) -> None:
        await session.say(message, allow_interruptions=False, add_to_chat_ctx=False)

    def _on_data_received(data_packet) -> None:
        if getattr(data_packet, "topic", "") != ASSISTANT_EVENT_TOPIC:
            return

        try:
            payload = json.loads(data_packet.data.decode("utf-8"))
        except Exception:
            return

        data_type = payload.get("type")
        if data_type == "medical_report":
            report_content = payload.get("report_content")
            if report_content:
                session.chat_ctx.append(
                    llm.ChatMessage(
                        role="system",
                        content=f"The user has uploaded a medical report: {report_content}. Please interpret this report for the user in a simplified and user-friendly way, highlighting key findings, precautions, and dietary suggestions as per your instructions. Ask one question at a time for follow-up.",
                    )
                )
        elif data_type == "user_structured_data":
            user_data = payload.get("user_data")
            if user_data:
                session.chat_ctx.append(
                    llm.ChatMessage(
                        role="system",
                        content=f"The following structured user data has been provided: {json.dumps(user_data)}. Use this information for triage and risk assessment.",
                    )
                )

    ctx.room.on("data_received", _on_data_received)

    await session.start(
        agent=HealthcareTriageVoiceAgent(language_profile, language_code),
        room=ctx.room,
    )
    await asyncio.sleep(GREETING_PLAYBACK_DELAY_SECONDS)

    greeting = OPENING_GREETINGS.get(language_code) or OPENING_GREETINGS["en-IN"]
    await session.say(greeting.format(user_name=user_name))


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=VOICE_AGENT_NAME,
        )
    )
