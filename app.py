from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)
@app.route("/")
def home():
    return {"status": "ok", "service": "doctor-conversation-analyzer"}
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

@app.route("/analyze", methods=["POST"])
def analyze():

    conversation = request.form.get("conversation", "")

    if not conversation:
        return jsonify({
            "ok": False,
            "error": "No conversation provided"
        }), 400

    prompt = f"""
You are an experienced medical assistant.

Analyze the following conversation between a doctor and a patient.

Return JSON with:

summary
possible_conditions
recommended_actions
urgency

Conversation:

{conversation}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role":"system","content":"You are a medical conversation analyzer."},
            {"role":"user","content":prompt}
        ]
    )

    return jsonify({
        "ok": True,
        "result": response.choices[0].message.content
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
