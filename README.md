# Traiage - AI Healthcare Triage Assistant

A voice-enabled healthcare triage assistant specializing in Diabetes, Anaemia, Hypertension, and Pneumonia.

## Features

- 💬 Text-based chat with Sarvam AI LLM
- 🎙️ Voice consultation with Sarvam STT/TTS via LiveKit
- 📄 Medical report upload and analysis (PDF/TXT)
- 🧘 Personalized lifestyle, diet, and yoga recommendations
- 🌍 Multi-language support (English, Hindi, and 9+ Indian languages)

## Tech Stack

- **Backend**: Flask (Python)
- **AI**: Sarvam AI (LLM, STT, TTS)
- **Voice**: LiveKit (real-time audio)
- **Frontend**: Vanilla JS + LiveKit Client SDK

## Deployment

### Render.com

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect your repo
4. Add environment variables:
   - `SARVAM_API_KEY`
   - `LIVEKIT_URL`
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`
   - `FLASK_SECRET_KEY`
5. Deploy!

### Voice Agent (Background Worker)

Deploy the voice agent as a separate Background Worker on Render:
- Start command: `python backend/agents/voice_agent.py start`
- Use the same environment variables

## Local Development

```bash
pip install -r requirements.txt
python app.py  # Flask app on port 5001
python backend/agents/voice_agent.py start  # Voice agent
```

## Environment Variables

```env
SARVAM_API_KEY=your_sarvam_key
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
FLASK_SECRET_KEY=your_random_secret
```

## License

MIT
