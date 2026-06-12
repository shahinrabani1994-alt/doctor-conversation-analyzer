from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json
import tempfile

app = Flask(__name__)

@app.route("/")
def home():
    return {
        "status": "ok",
        "service": "doctor-conversation-analyzer"
    }

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

@app.route("/analyze", methods=["POST"])
def analyze():

    if "audio" not in request.files:
        return jsonify({
            "ok": False,
            "error": "No audio file provided"
        }), 400

    audio_file = request.files["audio"]

    try:

        # ذخیره موقت فایل
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            audio_file.save(temp_file.name)

            # Whisper transcription
            with open(temp_file.name, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )

        conversation = transcript.text

        prompt = f"""
You are an experienced medical assistant.

Analyze the following conversation between a doctor and a patient.

Return ONLY valid JSON using this schema:

{{
  "summary": "...",
  "possible_conditions": [
    "...",
    "..."
  ],
  "recommended_actions": [
    "...",
    "..."
  ],
  "urgency": "low|medium|high"
}}

Conversation:

{conversation}
"""

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical conversation analyzer. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        raw_result = response.choices[0].message.content

        try:
            analysis = json.loads(raw_result)
        except Exception:
            analysis = {
                "summary": raw_result,
                "possible_conditions": [],
                "recommended_actions": [],
                "urgency": "medium"
            }

        return jsonify({
            "ok": True,
            "transcript": conversation,
            "result": analysis
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
