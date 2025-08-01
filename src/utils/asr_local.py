from faster_whisper import WhisperModel

# Load model once globally (adjust model size if needed)
model = WhisperModel("base")  # or "base" as preferred

def transcribe_local(audio_bytes: bytes, language: str | None = None) -> dict:
    # Save bytes temporarily or use from buffer
    import io
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        # Run transcription
        segments, info = model.transcribe(tmp.name, language=language, word_timestamps=True)

        # Build output in OpenAI verbose_json style
        result = {
            "text": info.text,
            "segments": [],
        }

        for i, segment in enumerate(segments):
            seg_dict = {
                "id": i,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word
                    } for w in segment.words
                ]
            }
            result["segments"].append(seg_dict)

        return result
