#!/usr/bin/env python3
"""
Low-latency Soniox streamer using blocking microphone reads.

Refactor notes:
- One utterance per WebSocket connection
- WS is closed and reopened after dispatch
- No STT hold / silence injection
- Deterministic lifecycle
"""

import os
import sys
import time
import json
import threading
from typing import Optional, Callable

import subprocess
import re
import platform

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
    from websockets.sync.client import connect
    from websockets.exceptions import WebSocketException
except ImportError:
    connect = None  # type: ignore
    WebSocketException = Exception  # type: ignore

# -----------------------
# Config
# -----------------------
SONIOX_WS_URL       = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_API_KEY      = os.getenv("SONIOX_API_KEY", "").strip()

MIC_SAMPLE_RATE     = 48000
TARGET_SAMPLE_RATE  = 16000
MIC_CHANNELS        = 1

CHUNK_MS            = 30
MIC_CHUNK_SAMPLES   = int(MIC_SAMPLE_RATE * CHUNK_MS / 1000)
OUT_CHUNK_SAMPLES   = int(TARGET_SAMPLE_RATE * CHUNK_MS / 1000)

SONIOX_MODEL                 = "stt-rt-v3"
LANGUAGE_HINTS               = ["bn"]
ENABLE_ENDPOINT_DETECTION    = True

# -----------------------
# Callbacks
# -----------------------
_live_callback: Optional[Callable[[str], None]] = None

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
def make_soniox_config(api_key: Optional[str] = None):
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


def get_wordforum_usb_index() -> Optional[int]:
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

        print("Did not find USB Audio Input Device")

    except Exception as e:
        print("Error detecting device:", e)

    return None

def alsa_card_to_sounddevice_index(alsa_card: int) -> Optional[int]:
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name = dev.get('name', '')
            if f"hw:{alsa_card}," in name:
                return i
    except Exception:
        pass
    return None

# -----------------------
# Resampling
# -----------------------
def resample_chunk(in_int16: np.ndarray) -> np.ndarray:
    x = in_int16.astype(np.float32) / 32768.0
    res = pysoxr.resample(x, MIC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    out = np.clip(res * 32767.0, -32768, 32767).astype(np.int16)

    if out.size != OUT_CHUNK_SAMPLES:
        if out.size > OUT_CHUNK_SAMPLES:
            out = out[:OUT_CHUNK_SAMPLES]
        else:
            out = np.pad(out, (0, OUT_CHUNK_SAMPLES - out.size))
    return out

# -----------------------
# Token rendering
# -----------------------
def render_tokens(final_tokens, non_final_tokens) -> str:
    return "".join(t["text"] for t in final_tokens + non_final_tokens if t.get("text"))

# -----------------------
# WebSocket receiver
# -----------------------
# -----------------------
# WebSocket receiver
# -----------------------
def ws_receiver(ws, result_container):
    final_tokens = []
    after_end = False
    empty_frames = 0

    try:
        while not _stop_event.is_set():
            msg = json.loads(ws.recv())
            tokens = msg.get("tokens", [])
            non_final = []

            for t in tokens:
                if t.get("is_final"):
                    final_tokens.append(t)
                else:
                    non_final.append(t)

            text = render_tokens(final_tokens, non_final)
            
            # Update live callback if set (optional, for UI/Printing)
            if _live_callback:
                _live_callback(text)

            if "<end>" in text and not after_end:
                after_end = True
                empty_frames = 0

            if after_end:
                empty_frames = empty_frames + 1 if not tokens else 0

            if after_end and empty_frames >= 1:
                result_container["text"] = text
                _dispatch_event.set()
                break

    except Exception:
        pass

# -----------------------
# Blocking mic loop
# -----------------------
def mic_producer_loop(ws, device_index):
    stream = sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        blocksize=MIC_CHUNK_SAMPLES,
        device=device_index
    )

    stream.start()
    try:
        while not _stop_event.is_set() and not _dispatch_event.is_set():
            # Always read to keep buffer clear and timing correct
            audio, _ = stream.read(MIC_CHUNK_SAMPLES)
            
            # Check hold state
            if not _hold_event.is_set():
                # Send silence to keep connection alive but ignore input
                silent = np.zeros(OUT_CHUNK_SAMPLES, dtype=np.int16)
                ws.send(silent.tobytes())
                continue

            out = resample_chunk(np.asarray(audio, dtype=np.int16).flatten())
            ws.send(out.tobytes())
    finally:
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

def listen_once(live_callback=None) -> str:
    """
    Connects to Soniox, listens for one utterance, and returns the final text.
    Blocks until the utterance is complete.
    """
    
    # def validate_device():
    #     try:
    #         return open("/proc/device-tree/serial-number").read().strip('\x00') == "10000000656f322e"
    #     except:
    #         return False
    # if not validate_device():
    #     print("DRM validation failed: unauthorized!")
    #     return
    
    _stop_event.clear()
    _hold_event.wait()
    global _live_callback
    _live_callback = live_callback
    
    _dispatch_event.clear()

    alsa = get_wordforum_usb_index()
    device_index = alsa_card_to_sounddevice_index(alsa) if alsa is not None else None

    result_container = {"text": ""}

    try:
        with connect(SONIOX_WS_URL, timeout=20) as ws:
            ws.send(json.dumps(make_soniox_config()))
            print("Listening...")

            recv_thread = threading.Thread(target=ws_receiver, args=(ws, result_container), daemon=True)
            recv_thread.start()

            mic_producer_loop(ws, device_index)

            recv_thread.join(timeout=1.0)
            
    except WebSocketException as e:
        print("WS error:", e)
    
    return result_container["text"].replace("<end>", "").strip()

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

