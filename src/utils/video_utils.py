import os
from src.utils.asr_local import transcribe_local

def get_whisper_srt(f_name, audio_bytes):
    whisper_provider = os.getenv("WHISPER_PROVIDER", "auto")

    if whisper_provider == "openai":
        # Keep existing OpenAI transcription logic
        return transcribe_openai(audio_bytes)  # assuming this exists

    elif whisper_provider == "faster-whisper":
        return transcribe_local(audio_bytes)

    elif whisper_provider == "auto":
        try:
            return transcribe_openai(audio_bytes)
        except Exception as e:
            print(f"OpenAI transcription failed: {e}. Falling back to faster-whisper.")
            return transcribe_local(audio_bytes)
