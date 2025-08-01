import pytest
from utils.asr_local import transcribe_local
def test_transcribe_local_adapter(monkeypatch):
    class MockWord:
        def __init__(self, start, end, word):
            self.start = start
            self.end = end
            self.word = word

    class MockSegment:
        def __init__(self):
            self.start = 0.0
            self.end = 1.0
            self.text = "Hi"
            self.words = [MockWord(0.0, 0.5, "Hi")]

    def mock_transcribe(audio_path, **kwargs):  # <-- removed self
        return [MockSegment()], type('Info', (), {"text": "Hi"})()

    import utils.asr_local as asr_local
    monkeypatch.setattr(asr_local.model, "transcribe", mock_transcribe)

    from utils.asr_local import transcribe_local
    result = transcribe_local(b"fake audio bytes")

    assert result["text"] == "Hi"
