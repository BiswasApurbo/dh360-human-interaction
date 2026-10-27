import os
import platform
import asyncio
import datetime
import json
import threading

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore

try:
    from google.cloud import texttospeech
    from google.oauth2 import service_account
except ImportError:
    texttospeech = None  # type: ignore
    service_account = None  # type: ignore


# ============================================================
# SERVICE ACCOUNT CREDENTIALS & RESOLUTION
# ============================================================

SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()


def resolve_tts_credentials(credentials_path_or_json: str | None = None):
    """Resolve Google Cloud credentials safely from arguments or environment.

    Hierarchy:
    1. Direct credentials_path_or_json argument (path or JSON string)
    2. GOOGLE_APPLICATION_CREDENTIALS environment variable (file path)
    3. GOOGLE_SERVICE_ACCOUNT_JSON environment variable (JSON string)
    4. Google Application Default Credentials (ADC)
    """
    if service_account is None:
        raise RuntimeError("google-auth / google.oauth2 package is not installed.")

    # 1. Parameter passed
    if credentials_path_or_json:
        stripped = credentials_path_or_json.strip()
        if os.path.isfile(stripped):
            return service_account.Credentials.from_service_account_file(stripped)
        try:
            info = json.loads(stripped)
            return service_account.Credentials.from_service_account_info(info)
        except Exception as e:
            raise ValueError(f"Invalid credentials argument provided: {e}") from None

    # 2. File path from environment
    sa_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if sa_file and os.path.isfile(sa_file):
        return service_account.Credentials.from_service_account_file(sa_file)

    # 3. Raw JSON from environment
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip() or SERVICE_ACCOUNT_JSON
    if sa_json:
        try:
            info = json.loads(sa_json)
            return service_account.Credentials.from_service_account_info(info)
        except Exception as e:
            raise ValueError(f"Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON: {e}") from None

    # 4. Fallback to Application Default Credentials (ADC)
    try:
        import google.auth

        creds, _ = google.auth.default()
        return creds
    except Exception:
        pass

    raise ValueError(
        "Google Cloud Text-to-Speech credentials not configured. "
        "Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON path, "
        "or GOOGLE_SERVICE_ACCOUNT_JSON to the JSON content."
    )


# ============================================================
# AUDIO STREAMER
# ============================================================

class AudioStreamer:
    """
    Non-blocking audio playback.

    Google TTS PCM chunks are placed into a local byte buffer.
    PortAudio reads from that buffer through its callback.

    This avoids sounddevice.OutputStream.write(), which was
    blocking in the ALSA/PortAudio thread and causing the
    30-second interrupt delay.
    """

    def __init__(self, rate=24000, channels=1):
        self.rate = rate
        self.channels = channels
        self.stream = None

        self.audio_buffer = bytearray()
        self.buffer_lock = threading.Lock()

        self._interrupted = False
        self._start_stream()

    def _find_pipewire_device(self):
        devices = sd.query_devices()

        for i, device in enumerate(devices):
            name = str(device["name"]).strip().lower()

            if (
                name == "pipewire"
                and device["max_output_channels"] > 0
            ):
                return i

        if platform.system() != "Linux":
            return None
        return None

    def _start_stream(self):
        self.stop()

        self._interrupted = False

        device = self._find_pipewire_device()

        print(
            f"Using audio device: {device} "
            f"({sd.query_devices(device)['name'] if device is not None else 'System Default'})"
        )

        def callback(outdata, frames, time_info, status):
            bytes_needed = (
                frames
                * self.channels
                * np.dtype(np.int16).itemsize
            )

            if self._interrupted:
                outdata[:] = b"\x00" * bytes_needed
                return

            with self.buffer_lock:
                available = len(self.audio_buffer)

                if available >= bytes_needed:
                    data = bytes(
                        self.audio_buffer[:bytes_needed]
                    )
                    del self.audio_buffer[:bytes_needed]

                else:
                    data = bytes(self.audio_buffer)
                    self.audio_buffer.clear()

            if len(data) < bytes_needed:
                data += b"\x00" * (
                    bytes_needed - len(data)
                )

            outdata[:] = data

        self.stream = sd.RawOutputStream(
            samplerate=self.rate,
            channels=self.channels,
            dtype="int16",
            device=device,
            callback=callback,
            blocksize=0,
        )

        self.stream.start()

    def ensure_started(self):
        if self.stream is None:
            self._start_stream()
            return

        if not self.stream.active:
            self._start_stream()

    def play_chunk(self, chunk):
        """
        Non-blocking.

        This only copies PCM data into the local buffer.
        It never calls stream.write().
        """

        if self._interrupted:
            return

        if not chunk:
            return

        with self.buffer_lock:
            self.audio_buffer.extend(chunk)

    def interrupt(self):
        """
        Immediately stop playback and discard all buffered audio.

        No blocking PortAudio operation is performed here.
        """

        self._interrupted = True

        with self.buffer_lock:
            self.audio_buffer.clear()

        stream = self.stream

        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

    def stop(self):
        self._interrupted = True

        with self.buffer_lock:
            self.audio_buffer.clear()

        stream = self.stream
        self.stream = None

        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass


# ============================================================
# GOOGLE TTS
# ============================================================

class GoogleTTS:

    def __init__(self, credentials_path_or_json: str | None = None):
        if texttospeech is None:
            raise RuntimeError("google-cloud-texttospeech package is not installed.")

        credentials = resolve_tts_credentials(credentials_path_or_json)

        self.client = texttospeech.TextToSpeechClient(
            credentials=credentials
        )

        self.audio_streamer = AudioStreamer()

        self.is_interrupted = False
        self.current_queue = None

    # --------------------------------------------------------
    # INTERRUPT
    # --------------------------------------------------------

    def interrupt(self):
        """
        Immediately interrupt TTS generation/playback.
        """

        print("Interrupt signal received in TTS")

        self.is_interrupted = True

        # Stop audio immediately.
        self.audio_streamer.interrupt()

        # Discard any chunks waiting in asyncio queue.
        queue = self.current_queue

        if queue is not None:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    break

    # --------------------------------------------------------
    # AUDIO PLAYER
    # --------------------------------------------------------

    async def _audio_player_worker(self, audio_queue):
        """
        Feed received PCM chunks into AudioStreamer.

        AudioStreamer.play_chunk() is non-blocking, so this
        coroutine can respond to interrupts immediately.
        """

        chunks_played = 0
        first_chunk_time = None

        try:
            while True:

                if self.is_interrupted:
                    break

                try:
                    chunk = await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=0.01,
                    )

                except asyncio.TimeoutError:
                    continue

                except asyncio.CancelledError:
                    break

                except Exception:
                    break

                if chunk is None:
                    break

                if self.is_interrupted:
                    break

                if first_chunk_time is None:
                    first_chunk_time = datetime.datetime.now()

                    start_t = globals().get(
                        "streaming_start_time",
                        datetime.datetime.now(),
                    )

                    print(
                        "First chunk playback started at: "
                        f"{(first_chunk_time - start_t).total_seconds()} "
                        "seconds"
                    )

                # NON-BLOCKING.
                self.audio_streamer.play_chunk(chunk)

                chunks_played += 1

                try:
                    audio_queue.task_done()
                except Exception:
                    pass

                # Immediately give control back to asyncio.
                await asyncio.sleep(0)

        finally:
            print(
                f"Audio player finished. "
                f"Played {chunks_played} chunks."
            )

    # --------------------------------------------------------
    # STREAM TTS
    # --------------------------------------------------------

    async def stream_text_immediately(
        self,
        text: str,
        voice_name="en-US-Chirp3-HD-Charon",
        language_code="en-US",
    ):
        """
        Real-time streaming Google Cloud TTS.
        """

        global streaming_start_time

        # Reset interrupt state for this turn.
        self.is_interrupted = False

        # Recreate/restart audio if the previous turn stopped it.
        self.audio_streamer.ensure_started()

        audio_queue = asyncio.Queue(
            maxsize=512
        )

        self.current_queue = audio_queue

        player_task = asyncio.create_task(
            self._audio_player_worker(
                audio_queue
            )
        )

        print(
            "Starting Google Cloud TTS streaming"
        )

        print(
            f"Voice: {voice_name}, "
            f"Language: {language_code}"
        )

        print(
            f"Text length: {len(text)} characters"
        )

        if len(text) > 100:
            print(
                f"Streaming text: {text[:100]}..."
            )
        else:
            print(
                f"Streaming text: {text}"
            )

        print()

        streaming_start_time = (
            datetime.datetime.now()
        )

        request_start_time = (
            streaming_start_time
        )

        try:

            # ------------------------------------------------
            # GOOGLE TTS CONFIG
            # ------------------------------------------------

            streaming_config = (
                texttospeech.StreamingSynthesizeConfig(
                    voice=texttospeech.VoiceSelectionParams(
                        name=voice_name,
                        language_code=language_code,
                    )
                )
            )

            config_request = (
                texttospeech.StreamingSynthesizeRequest(
                    streaming_config=streaming_config
                )
            )

            # ------------------------------------------------
            # REQUEST GENERATOR
            # ------------------------------------------------

            def request_generator():
                yield config_request

                yield (
                    texttospeech.StreamingSynthesizeRequest(
                        input=(
                            texttospeech.StreamingSynthesisInput(
                                text=text
                            )
                        )
                    )
                )

            print(
                "Text sent to Google Cloud TTS - "
                "waiting for audio..."
            )

            # ------------------------------------------------
            # GOOGLE STREAM
            # ------------------------------------------------

            streaming_responses = (
                self.client.streaming_synthesize(
                    request_generator()
                )
            )

            first_audio_received = False
            chunks_received = 0

            for response in streaming_responses:

                if self.is_interrupted:
                    print(
                        "TTS Streaming interrupted "
                        "during receiving loop."
                    )
                    break

                if not response.audio_content:
                    continue

                if not first_audio_received:

                    first_audio_received = True

                    first_audio_time = (
                        datetime.datetime.now()
                    )

                    time_to_first_audio = (
                        first_audio_time
                        - request_start_time
                    )

                    print(
                        "Time to first audio: "
                        f"{time_to_first_audio.total_seconds()} "
                        "seconds"
                    )

                if self.is_interrupted:
                    break

                chunks_received += 1

                await audio_queue.put(
                    response.audio_content
                )

                # Give audio worker immediate CPU time.
                await asyncio.sleep(0)

            if self.is_interrupted:

                print(
                    "Audio synthesis interrupted. "
                    f"Received {chunks_received} chunks."
                )

            else:

                print(
                    "Audio synthesis complete. "
                    f"Received {chunks_received} chunks."
                )

        except asyncio.CancelledError:
            self.is_interrupted = True
            raise

        except Exception as e:

            if not self.is_interrupted:
                print(
                    "Error in Google Cloud TTS streaming: "
                    f"{e}"
                )

        finally:

            # ------------------------------------------------
            # INTERRUPTED
            # ------------------------------------------------

            if self.is_interrupted:

                if not player_task.done():
                    player_task.cancel()

                try:
                    await player_task
                except asyncio.CancelledError:
                    pass

            # ------------------------------------------------
            # NORMAL COMPLETION
            # ------------------------------------------------

            else:

                await audio_queue.put(None)

                try:
                    await player_task
                except asyncio.CancelledError:
                    pass

            self.current_queue = None

            completion_time = (
                datetime.datetime.now()
            )

            total_time = (
                completion_time
                - request_start_time
            )

            print(
                "Total streaming time: "
                f"{total_time.total_seconds()} seconds"
            )


# ============================================================
# MAIN
# ============================================================

async def main():

    tts = GoogleTTS()

    large_text = """
The cow (Bos taurus) is arguably the most significant domestic
animal, fundamentally shaping human culture and agriculture across
millennia. As a large, docile herbivore, its unique biology is
centered on its status as a ruminant, possessing a complex,
four-chambered stomach—the rumen, reticulum, omasum, and abomasum.
"""

    large_text = " ".join(
        large_text.split()
    )

    print(
        "=== Google Cloud TTS Real-Time Streaming Test ==="
    )

    print(
        "This should be much faster than Gemini Live API"
    )

    print()

    task = asyncio.create_task(
        tts.stream_text_immediately(
            text=large_text,
            voice_name="en-US-Chirp3-HD-Charon",
            language_code="en-US",
        )
    )

    # Simulate interrupt after 10 seconds.
    # Change to 2 for a 2-second interrupt test.
    await asyncio.sleep(10)

    print(
        "\n!!! Simulating Interrupt !!!\n"
    )

    tts.interrupt()

    await task

    print(
        "\nGoogle Cloud TTS streaming completed!"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print(
            "\nProgram interrupted by user."
        )
