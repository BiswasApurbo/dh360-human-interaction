#!/usr/bin/env python3
"""
Low-latency Soniox streamer using blocking microphone reads.

Refactor notes:
- One utterance per WebSocket connection
- WS is closed and reopened after dispatch
- No STT hold / silence injection
- Deterministic lifecycle
"""

import contextlib
import json
import os
import platform
import re
import subprocess
import threading
import time
from collections.abc import Callable

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore

try:
    import soxr as pysoxr  # type: ignore
except ImportError:
    pysoxr = None  # type: ignore

try:
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect
except ImportError:
    connect = None  # type: ignore
    WebSocketException = Exception  # type: ignore

# -----------------------
# Config
# -----------------------
SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY", "").strip()

MIC_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
MIC_CHANNELS = 1

CHUNK_MS = 30
MIC_CHUNK_SAMPLES = int(MIC_SAMPLE_RATE * CHUNK_MS / 1000)
OUT_CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_MS / 1000)

SONIOX_MODEL = "stt-rt-v3"
LANGUAGE_HINTS = ["bn"]
ENABLE_ENDPOINT_DETECTION = True

# -----------------------
# Callbacks
# -----------------------
_live_callback: Callable[[str], None] | None = None

# -----------------------
# Thread control
# -----------------------
_stop_event = threading.Event()
_dispatch_event = threading.Event()
_hold_event = threading.Event()
_hold_event.set()


# -----------------------
# Soniox config
# -----------------------
def make_soniox_config(api_key: str | None = None):
    resolved_key = (api_key or os.getenv("SONIOX_API_KEY", "") or SONIOX_API_KEY).strip()
    if not resolved_key:
        raise ValueError("SONIOX_API_KEY environment variable is required.")
    return {
        "api_key": resolved_key,
        "model": SONIOX_MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": TARGET_SAMPLE_RATE,
        "num_channels": MIC_CHANNELS,
        "enable_endpoint_detection": ENABLE_ENDPOINT_DETECTION,
        "language_hints": LANGUAGE_HINTS,
    }


# -----------------------
# Device helpers
# -----------------------
# def get_wordforum_usb_index() -> Optional[int]:
#     if platform.system() != "Linux":
#         return None
#     try:
#         result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True)
#         for line in result.stdout.splitlines():
#             print(line, "\n")
#             match = re.search(r"card (\d+): .*(USB Audio [USB Audio])", line)
#             if match:
#                 print("Found: USB Audio Input Device at ", int(match.group(1)) )
#                 return int(match.group(1))
#         print("Did not find USB Audio Input Device")
#     except Exception:
#         print("Did not find USB Audio Input Device")
#     return None


def get_wordforum_usb_index() -> int | None:
    if platform.system() != "Linux":
        return None
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True)

        for line in result.stdout.splitlines():
            # Match card number if USB Audio appears anywhere in line
            match = re.search(r"card (\d+):.*USB Audio", line, re.IGNORECASE)

            if match:
                card_index = int(match.group(1))
                print("Found USB Audio Input Device at card", card_index)
                return card_index

        # Secondary check: any USB capture device
        for line in result.stdout.splitlines():
            if "card" in line.lower() and "usb" in line.lower():
                match = re.search(r"card (\d+):", line, re.IGNORECASE)
                if match:
                    card_index = int(match.group(1))
                    print("Found USB capture card at", card_index)
                    return card_index

        print("Did not find USB Audio Input Device")

    except Exception as e:
        print("Error detecting device:", e)

    return None


def has_capture_hardware() -> bool:
    """Return True if at least one audio capture hardware device is detected."""
    if platform.system() != "Linux":
        return True
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True)
        lines = [ln for ln in result.stdout.splitlines() if "card" in ln.lower()]
        return len(lines) > 0
    except Exception:
        return True


def alsa_card_to_sounddevice_index(alsa_card: int) -> int | None:
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name = dev.get("name", "")
            if f"hw:{alsa_card}," in name:
                return i
    except Exception:
        pass
    return None


# -----------------------
# Resampling
# -----------------------
def resample_chunk(
    in_int16: np.ndarray,
    in_rate: int = MIC_SAMPLE_RATE,
    out_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    if in_rate == out_rate:
        return in_int16
    if pysoxr is not None:
        x = in_int16.astype(np.float32) / 32768.0
        res = pysoxr.resample(x, in_rate, out_rate)
        out = np.clip(res * 32767.0, -32768, 32767).astype(np.int16)
        if out.size != OUT_CHUNK_SAMPLES:
            if out.size > OUT_CHUNK_SAMPLES:
                out = out[:OUT_CHUNK_SAMPLES]
            else:
                out = np.pad(out, (0, OUT_CHUNK_SAMPLES - out.size))
        return out

    # Pure numpy fallback (decimation or linear interpolation) without pysoxr dependency
    if np is not None:
        if in_rate % out_rate == 0:
            factor = in_rate // out_rate
            out = in_int16[::factor]
        else:
            num_out = int(round(len(in_int16) * out_rate / in_rate))
            indices = np.linspace(0, len(in_int16) - 1, num_out)
            out = np.interp(indices, np.arange(len(in_int16)), in_int16).astype(np.int16)
        if out.size != OUT_CHUNK_SAMPLES:
            if out.size > OUT_CHUNK_SAMPLES:
                out = out[:OUT_CHUNK_SAMPLES]
            else:
                out = np.pad(out, (0, OUT_CHUNK_SAMPLES - out.size))
        return out

    return in_int16


# -----------------------
# Token rendering
# -----------------------
def render_tokens(final_tokens, non_final_tokens) -> str:
    return "".join(t["text"] for t in final_tokens + non_final_tokens if t.get("text"))


# -----------------------
# WebSocket receiver
# -----------------------
def ws_receiver(
    ws,
    result_container,
    silence_timeout: float = 6.0,
    post_speech_timeout: float = 2.0,
    max_duration: float = 15.0,
):
    final_tokens = []
    latest_text = ""
    after_end = False
    empty_frames = 0
    start_time = time.time()
    last_speech_time = None

    try:
        while not _stop_event.is_set():
            now = time.time()
            # 1. Check max overall duration
            if now - start_time >= max_duration:
                result_container["text"] = latest_text
                _dispatch_event.set()
                _stop_event.set()
                break

            # 2. Check silence timeout (no speech detected)
            if last_speech_time is None and (now - start_time >= silence_timeout):
                result_container["text"] = ""
                _dispatch_event.set()
                _stop_event.set()
                break

            # 3. Check post-speech silence timeout (speech ended)
            if last_speech_time is not None and (now - last_speech_time >= post_speech_timeout):
                result_container["text"] = latest_text
                _dispatch_event.set()
                _stop_event.set()
                break

            try:
                raw_msg = ws.recv(timeout=0.3)
            except TimeoutError:
                continue
            except Exception:
                break

            msg = json.loads(raw_msg)
            tokens = msg.get("tokens", [])
            non_final = []

            for t in tokens:
                if t.get("is_final"):
                    final_tokens.append(t)
                else:
                    non_final.append(t)

            text = render_tokens(final_tokens, non_final)
            clean_text = text.replace("<end>", "").strip()
            if clean_text:
                latest_text = clean_text
                last_speech_time = time.time()

            # Update live callback if set (optional, for UI/Printing)
            if _live_callback and clean_text:
                _live_callback(clean_text)

            if "<end>" in text and not after_end:
                after_end = True
                empty_frames = 0

            if after_end:
                empty_frames = empty_frames + 1 if not tokens else 0

            if (after_end and empty_frames >= 1) or msg.get("finished", False):
                result_container["text"] = latest_text
                _dispatch_event.set()
                _stop_event.set()
                break

    except Exception:
        pass
    finally:
        if not result_container.get("text") and latest_text:
            result_container["text"] = latest_text
        _dispatch_event.set()


# -----------------------
# Blocking mic loop
# -----------------------
def mic_producer_loop(ws, device_index=None, audio_queue=None, max_duration: float = 16.0):
    start_time = time.time()
    if audio_queue is not None:
        try:
            while not _stop_event.is_set() and not _dispatch_event.is_set():
                if time.time() - start_time >= max_duration:
                    break
                try:
                    chunk = audio_queue.get(timeout=0.2)
                except Exception:
                    continue
                if chunk is None:
                    break

                if not _hold_event.is_set():
                    silent = np.zeros(OUT_CHUNK_SAMPLES, dtype=np.int16) if np is not None else b"\x00" * (OUT_CHUNK_SAMPLES * 2)
                    ws.send(silent.tobytes() if hasattr(silent, "tobytes") else silent)
                    continue

                if isinstance(chunk, (bytes, bytearray)):
                    ws.send(chunk)
                elif np is not None and isinstance(chunk, np.ndarray):
                    ws.send(chunk.astype(np.int16).tobytes())
        finally:
            with contextlib.suppress(Exception):
                ws.send("")
        return

    stream = sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        blocksize=MIC_CHUNK_SAMPLES,
        device=device_index,
    )

    stream.start()
    try:
        while not _stop_event.is_set() and not _dispatch_event.is_set():
            if time.time() - start_time >= max_duration:
                break
            # Always read to keep buffer clear and timing correct
            audio, _ = stream.read(MIC_CHUNK_SAMPLES)

            # Check hold state
            if not _hold_event.is_set():
                # Send silence to keep connection alive but ignore input
                silent = np.zeros(OUT_CHUNK_SAMPLES, dtype=np.int16) if np is not None else b"\x00" * (OUT_CHUNK_SAMPLES * 2)
                ws.send(silent.tobytes() if hasattr(silent, "tobytes") else silent)
                continue

            out = resample_chunk(np.asarray(audio, dtype=np.int16).flatten())
            ws.send(out.tobytes())
    finally:
        with contextlib.suppress(Exception):
            ws.send("")
        stream.stop()
        stream.close()


# -----------------------
# Public API
# -----------------------
def hold_stt():
    _hold_event.clear()


def release_stt():
    _hold_event.set()


def get_stt_hold():
    return not _hold_event.is_set()


def listen_once(live_callback=None, audio_queue=None, api_key: str | None = None) -> str:
    """
    Connects to Soniox, listens for one utterance, and returns the final text.
    Blocks until the utterance is complete or cleanly times out on silence.
    """

    _stop_event.clear()
    _dispatch_event.clear()
    _hold_event.wait()
    global _live_callback
    _live_callback = live_callback

    alsa = get_wordforum_usb_index()
    device_index = alsa_card_to_sounddevice_index(alsa) if alsa is not None else None

    result_container = {"text": ""}

    try:
        with connect(SONIOX_WS_URL, timeout=20) as ws:
            ws.send(json.dumps(make_soniox_config(api_key=api_key)))

            recv_thread = threading.Thread(
                target=ws_receiver,
                args=(ws, result_container),
                daemon=True,
                name="KBCON-SonioxReceiver",
            )
            recv_thread.start()

            mic_producer_loop(ws, device_index=device_index, audio_queue=audio_queue)

            _dispatch_event.wait(timeout=2.0)
            _stop_event.set()
            recv_thread.join(timeout=2.0)

    except WebSocketException as e:
        print("WS error:", e)
    finally:
        _stop_event.set()
        _dispatch_event.set()

    return result_container["text"].replace("<end>", "").strip()


def transcribe_audio_with_soniox(
    audio_bytes: bytes,
    mime_type: str = "audio/webm",
    language_hint: str = "bn",
    api_key: str | None = None,
) -> str:
    """Transcribe client-recorded audio bytes (e.g. from browser mic) using Soniox STT."""
    resolved_key = (api_key or os.getenv("SONIOX_API_KEY", "") or SONIOX_API_KEY).strip()
    if not resolved_key:
        # Check backend.env paths
        for p in (
            "/home/eliva/.config/dh360/backend.env",
            os.path.expanduser("~/.config/dh360/backend.env"),
        ):
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        for line in f:
                            if line.startswith("SONIOX_API_KEY="):
                                resolved_key = line.strip().split("=", 1)[1].strip("\"'")
                                break
                except Exception:
                    pass
            if resolved_key:
                break

    if not resolved_key:
        raise ValueError("SONIOX_API_KEY is not configured.")

    if connect is None:
        raise RuntimeError("websockets client is not installed.")

    import shutil

    pcm_bytes = None
    if shutil.which("ffmpeg"):
        try:
            proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    "pipe:0",
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "pipe:1",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, _ = proc.communicate(input=audio_bytes, timeout=12)
            if proc.returncode == 0 and out:
                pcm_bytes = out
        except Exception:
            pass

    lang_hints = [language_hint, "en"] if language_hint != "en" else ["en"]

    if pcm_bytes is not None:
        config = {
            "api_key": resolved_key,
            "model": SONIOX_MODEL,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
            "num_channels": 1,
            "language_hints": lang_hints,
            "enable_endpoint_detection": True,
        }
        payload = pcm_bytes
    else:
        config = {
            "api_key": resolved_key,
            "model": SONIOX_MODEL,
            "audio_format": "auto",
            "language_hints": lang_hints,
            "enable_endpoint_detection": True,
        }
        payload = audio_bytes

    with connect(SONIOX_WS_URL, timeout=12) as ws:
        ws.send(json.dumps(config))
        chunk_size = 2048
        for i in range(0, len(payload), chunk_size):
            ws.send(payload[i : i + chunk_size])
        ws.send("")

        final_tokens = []
        non_final = []
        start_t = time.time()
        while time.time() - start_t < 15.0:
            try:
                raw_msg = ws.recv(timeout=1.5)
            except Exception:
                break
            try:
                data = json.loads(raw_msg)
            except Exception:
                continue
            curr_non_final = []
            for t in data.get("tokens", []):
                if t.get("is_final"):
                    final_tokens.append(t)
                else:
                    curr_non_final.append(t)
            non_final = curr_non_final
            if data.get("finished", False):
                break

        all_tokens = final_tokens + non_final
        return "".join(t.get("text", "") for t in all_tokens).replace("<end>", "").strip()


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    try:
        while True:
            hold_stt()
            time.sleep(5)
            release_stt()
            text = listen_once(live_callback=lambda t: print(f"Live: {t}", end="\r"))
            if text:
                print(f"\nFinal: {text}")
            else:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        _stop_event.set()


def stop_stt():
    _stop_event.set()
