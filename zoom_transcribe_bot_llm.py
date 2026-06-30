import io
import wave
import sounddevice as sd
import numpy as np
import queue
import time
import os
from datetime import datetime
from openai import OpenAI

# ========================= CONFIG =========================
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_SEC = 8          # Transcription chunk size

# Choose your LLM provider: "gpt-4o" or "grok"
LLM_PROVIDER = "gpt-4o"         # Change to "grok" if preferred

# Set to a specific device index for virtual audio (e.g. VB-Cable, Blackhole).
# Run: python -c "import sounddevice as sd; print(sd.query_devices())" to list devices.
DEVICE_INDEX = 13                # Stereo Mix - captures all Windows/Zoom system audio

# API Keys (use environment variables for security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Dedicated OpenAI client for Whisper transcription (always required)
transcription_client = OpenAI(api_key=OPENAI_API_KEY)

# LLM client (may differ from transcription client)
if LLM_PROVIDER == "grok":
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )
    LLM_MODEL = "grok-4"  # or "grok-3" depending on availability
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    LLM_MODEL = "gpt-4o"

# Trading keywords (simple fallback)
TRADING_KEYWORDS = {
    "buy": ["bullish", "long", "buy signal", "upside", "breakout"],
    "sell": ["bearish", "short", "sell", "downside", "resistance"],
    "alert": ["fed", "earnings", "breaking", "surprise", "volatility"]
}

audio_queue = queue.Queue()

# Log file for this session
LOG_FILE = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
# ========================================================


def log(message: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def numpy_to_wav_bytes(audio_data: np.ndarray) -> bytes:
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.flatten().tobytes())
    byte_io.seek(0)
    return byte_io.read()


def transcribe_chunk(audio_data: np.ndarray):
    wav_bytes = numpy_to_wav_bytes(audio_data)
    try:
        transcript = transcription_client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", wav_bytes, "audio/wav"),
            language="en",
            response_format="text"
        )
        return transcript.strip()
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""


def analyze_with_llm(transcript: str):
    if not transcript:
        return None

    system_prompt = """You are an expert trading analyst specializing in real-time market commentary.
Analyze the provided transcript excerpt from a trading call/webinar.
Focus on:
- Key market insights, sentiment, and potential impact on major assets (stocks, indices, crypto).
- Specific trading signals: bullish/bearish bias, entry/exit ideas, risk factors.
- Actionable summary: Buy/Sell/Hold recommendations with reasoning.
Keep response concise (max 150 words) and structured."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript excerpt: {transcript}"}
            ],
            temperature=0.3,
            max_completion_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Analysis Error: {e}")
        return None


def trading_logic(transcript: str):
    if not transcript:
        return

    lower = transcript.lower()
    timestamp = datetime.now().strftime("%H:%M:%S")

    entry = f"\n[{timestamp}] TRANSCRIPT: {transcript}"
    print(entry)
    log(entry)

    # Quick keyword scan
    signals = []
    for action, keywords in TRADING_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            signals.append(action.upper())

    if signals:
        sig_line = f"QUICK SIGNALS: {', '.join(signals)}"
        print(f"[!] {sig_line}")
        log(sig_line)

    # LLM Deep Analysis
    print("[~] Running LLM trade analysis...")
    analysis = analyze_with_llm(transcript)

    if analysis:
        analysis_entry = f"LLM ANALYSIS:\n{analysis}\n"
        print(f"[+] {analysis_entry}")
        log(analysis_entry)


def main():
    print(f"[*] Zoom Trading Bot with {LLM_PROVIDER.upper()} Reasoning Layer Started")
    print(f"Model: {LLM_MODEL} | Chunk: {CHUNK_DURATION_SEC}s | Log: {LOG_FILE}\n")

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY is required (used for Whisper transcription).")
        return
    if LLM_PROVIDER == "grok" and not XAI_API_KEY:
        print("❌ XAI_API_KEY is required when LLM_PROVIDER is 'grok'.")
        return

    if DEVICE_INDEX is not None:
        print(f"Audio device: {sd.query_devices(DEVICE_INDEX)['name']}")
    else:
        print("Audio device: system default (set DEVICE_INDEX to capture Zoom audio)")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION_SEC),
        device=DEVICE_INDEX
    )

    with stream:
        buffer = np.array([], dtype=np.int16)

        while True:
            try:
                chunk = audio_queue.get(timeout=1)
                buffer = np.append(buffer, chunk)

                if len(buffer) >= SAMPLE_RATE * CHUNK_DURATION_SEC:
                    text = transcribe_chunk(buffer)

                    if text:
                        trading_logic(text)

                    buffer = np.array([], dtype=np.int16)

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                print("\nStopping bot...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)


if __name__ == "__main__":
    main()
