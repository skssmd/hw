import pytest
import os
from app import app  # Updated import to match your project structure

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_build_srt_file_faster_whisper(monkeypatch, client):
    # Monkeypatch transcribe_local to return a minimal valid verbose_json
    def fake_transcribe_local(audio_bytes, language=None):
        return {
            "text": "Hello",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello",
                    "words": [{"start": 0.0, "end": 0.5, "word": "Hello"}],
                }
            ],
        }

    import utils.asr_local as asr_local
    monkeypatch.setattr(asr_local, "transcribe_local", fake_transcribe_local)

    # Set env var for test
    os.environ["WHISPER_PROVIDER"] = "faster-whisper"

    # Use a test project_id; adjust based on your app’s handling
    project_id = "test_project"

    # Make a POST request to the endpoint
    response = client.post(f"/build_srt_file/{project_id}")

    # Validate the response
    assert response.status_code == 200
    assert b"success" in response.data
