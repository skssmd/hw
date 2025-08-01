from flask import Flask
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

WHISPER_PROVIDER = os.getenv("WHISPER_PROVIDER", "auto")
print(f"Using Whisper Provider: {WHISPER_PROVIDER}")

@app.route("/")
def index():
    return f"Whisper Provider: {WHISPER_PROVIDER}"

@app.route("/build_srt_file/<project_id>", methods=["POST"])
def build_srt_file(project_id):
    return {"status": "success", "project_id": project_id}, 200
