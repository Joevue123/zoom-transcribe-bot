import io
import wave
import json
import threading
import pyaudiowpatch as pyaudio
import numpy as np
import queue
import time
import os
from datetime import datetime
from faster_whisper import WhisperModel
from openai import OpenAI

# ========================= CONFIG =========================
SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 8          # Transcription chunk size

# Ollama (local) for LLM analysis
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
LLM_MODEL = "llama3.2:1b"

# faster-whisper runs locally — no API key needed
# Model options: "tiny", "base", "small", "medium", "large-v3"
WHISPER_MODEL_SIZE = "tiny"

# Trading keywords (simple fallback)
TRADING_KEYWORDS = {
    "buy": ["bullish", "long", "buy signal", "upside", "breakout"],
    "sell": ["bearish", "short", "sell", "downside", "resistance"],
    "alert": ["fed", "earnings", "breaking", "surprise", "volatility"]
}

audio_queue = queue.Queue()
LOG_FILE = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
SIGNALS_FILE = "signals.json"
_signals_lock = threading.Lock()
# ========================================================


def init_signals_file():
    data = {"last_updated": None, "counts": {"BUY": 0, "SELL": 0, "ALERT": 0}, "events": []}
    with open(SIGNALS_FILE, "w") as f:
        json.dump(data, f)


def write_signal_event(transcript: str, signals: list):
    with _signals_lock:
        try:
            with open(SIGNALS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {"last_updated": None, "counts": {"BUY": 0, "SELL": 0, "ALERT": 0}, "events": []}

        timestamp = datetime.now().strftime("%H:%M:%S")
        event = {"id": len(data["events"]) + 1, "timestamp": timestamp, "transcript": transcript, "signals": signals}
        data["events"].insert(0, event)
        data["events"] = data["events"][:100]  # keep last 100
        data["last_updated"] = timestamp
        for s in signals:
            data["counts"][s] = data["counts"].get(s, 0) + 1

        with open(SIGNALS_FILE, "w") as f:
            json.dump(data, f)


def log(message: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def numpy_to_wav_bytes(audio_data: np.ndarray) -> bytes:
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.flatten().tobytes())
    byte_io.seek(0)
    return byte_io.read()


def transcribe_chunk(whisper_model: WhisperModel, audio_data: np.ndarray) -> str:
    wav_bytes = numpy_to_wav_bytes(audio_data)
    try:
        audio_file = io.BytesIO(wav_bytes)
        segments, _ = whisper_model.transcribe(audio_file, language="en")
        return " ".join(s.text.strip() for s in segments).strip()
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
            max_tokens=300
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

    signals = []
    for action, keywords in TRADING_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            signals.append(action.upper())

    if signals:
        sig_line = f"QUICK SIGNALS: {', '.join(signals)}"
        print(f"[!] {sig_line}")
        log(sig_line)

    write_signal_event(transcript, signals)

    # LLM analysis disabled — re-enable when upgrading hardware


def main():
    init_signals_file()
    print(f"[*] Zoom Trading Bot | Whisper ({WHISPER_MODEL_SIZE}, local) + Keyword Signals")
    print(f"Chunk: {CHUNK_DURATION_SEC}s | Log: {LOG_FILE}\n")

    print("[*] Loading Whisper model (first run downloads it)...")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print("[*] Whisper model ready.\n")

    p = pyaudio.PyAudio()

    loopback_device = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get("isLoopbackDevice") and dev["maxInputChannels"] > 0:
            loopback_device = dev
            loopback_idx = i
            break

    if loopback_device is None:
        print("No WASAPI loopback device found.")
        p.terminate()
        return

    print(f"Audio device: {loopback_device['name']} (WASAPI loopback)")

    dev_channels = int(loopback_device["maxInputChannels"])
    src_rate = int(loopback_device["defaultSampleRate"])
    buffer = np.array([], dtype=np.int16)

    def pyaudio_callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=dev_channels,
        rate=src_rate,
        frames_per_buffer=1024,
        input=True,
        input_device_index=loopback_idx,
        stream_callback=pyaudio_callback
    )

    print("[*] Listening... (Ctrl+C to stop)\n")
    stream.start_stream()

    try:
        while stream.is_active():
            try:
                raw = audio_queue.get(timeout=1)
                chunk = np.frombuffer(raw, dtype=np.int16)

                if dev_channels > 1:
                    chunk = chunk.reshape(-1, dev_channels).mean(axis=1).astype(np.int16)

                if src_rate != SAMPLE_RATE:
                    new_len = int(len(chunk) * SAMPLE_RATE / src_rate)
                    chunk = np.interp(
                        np.linspace(0, len(chunk), new_len),
                        np.arange(len(chunk)),
                        chunk
                    ).astype(np.int16)

                buffer = np.append(buffer, chunk)

                if len(buffer) >= SAMPLE_RATE * CHUNK_DURATION_SEC:
                    text = transcribe_chunk(whisper_model, buffer[:SAMPLE_RATE * CHUNK_DURATION_SEC])
                    if text:
                        trading_logic(text)
                    buffer = buffer[SAMPLE_RATE * CHUNK_DURATION_SEC:]

            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\nStopping bot...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
