# Real-Time Voice Interaction with GPT-4 (Python CLI Implementation)

Building a real-time voice assistant with OpenAI GPT-4 involves capturing microphone audio, transcribing it to text, sending it to the GPT-4 API, and then converting the response back to speech. Below, we outline the steps and provide code examples, focusing on low latency and a seamless CLI (command-line) experience.

## Overview of the System Architecture

1. **Audio Capture (Microphone)** – Continuously record audio from the user’s microphone in real time.  
2. **Speech-to-Text (Whisper)** – Transcribe the audio stream into text using OpenAI’s Whisper (or a local Whisper model).  
3. **GPT-4 API (ChatCompletion)** – Send the transcribed text as input to the OpenAI GPT-4 model via the `ChatCompletion` API endpoint and receive a text response.  
4. **Text-to-Speech (TTS)** – Convert GPT-4’s text response into audio and play it back to the user.  
5. **Conversation Loop** – Repeat the process, maintaining context across turns for a natural conversational experience.

This pipeline ensures that voice interactions feel natural. *Note:* OpenAI’s new GPT-4o (“Omni”) model is designed to natively accept audio input and produce audio output with very low latency ([Hello GPT-4o | OpenAI](https://openai.com/index/hello-gpt-4o/#:~:text=GPT%E2%80%914o%20,GPT%E2%80%914o%20is)), but as of now we implement these steps manually via Whisper and a TTS engine. (OpenAI’s upcoming *Realtime API* will eventually allow end-to-end speech-to-speech over WebSockets ([Realtime API (Advanced Voice Mode) Python Implementation - API - OpenAI Developer Community](https://community.openai.com/t/realtime-api-advanced-voice-mode-python-implementation/964636#:~:text=The%20Realtime%20API%20enables%20you,as%20well%20as%20function%20calling)), but we’ll use the standard RESTful API here.)

## Capturing Microphone Audio in Real Time

To capture audio from the microphone in a CLI application, you can use libraries like **PyAudio** or **sounddevice**. These libraries interface with the system’s audio hardware (via PortAudio) to get a stream of audio samples.

- **PyAudio:** A popular library for audio I/O. It provides low-level control and uses PortAudio under the hood. PyAudio is widely used in examples and documentation, though some find it less documented lately ([using webrtcvad in realtime application · Issue #29 · wiseman/py-webrtcvad · GitHub](https://github.com/wiseman/py-webrtcvad/issues/29#:~:text=edited)).  
- **sounddevice:** A pure-Python alternative to PyAudio that also uses PortAudio. It offers a simpler API and is well-maintained ([using webrtcvad in realtime application · Issue #29 · wiseman/py-webrtcvad · GitHub](https://github.com/wiseman/py-webrtcvad/issues/29#:~:text=edited)) ([Best Python Tools for Manipulating Audio Data - Deepgram Blog ⚡️ | Deepgram](https://deepgram.com/learn/best-python-audio-manipulation-tools#:~:text=Play%20Audio%20With%20Python)). It can be easier to install (via `pip install sounddevice`) and often more convenient for recording/playing audio.  

For real-time streaming, both libraries let you read audio in small chunks (frames). The chunk size and sample rate determine how frequently you get audio data and impact latency. A common approach is to use a sample rate of **16 kHz** (16000 Hz) and 16-bit mono audio, since Whisper expects 16 kHz mono audio by default ([Generating Subtitles in Real-Time with OpenAI Whisper and PyAudio - Bruno Scheufler](https://brunoscheufler.com/blog/2023-03-12-generating-subtitles-in-real-time-with-openai-whisper-and-pyaudio#:~:text=We%20need%20some%20audio)). Smaller chunk sizes (e.g. 20-30 ms of audio per chunk) yield lower latency but result in more frequent API calls or processing.

Below is an example using **PyAudio** to capture audio from the microphone in real-time. This code reads audio in short chunks and can detect silence to decide when to stop recording a user’s query:

```python
import pyaudio
import numpy as np

# Audio stream config
RATE = 16000       # 16 kHz sample rate for Whisper compatibility
CHANNELS = 1       # mono audio
FORMAT = pyaudio.paInt16  # 16-bit audio
CHUNK = 1024       # buffer size in frames (1024 frames per chunk)

# Initialize PyAudio
audio_interface = pyaudio.PyAudio()
stream = audio_interface.open(format=FORMAT, channels=CHANNELS,
                              rate=RATE, input=True,
                              frames_per_buffer=CHUNK)

print("🎤 Listening... (press Ctrl+C to stop)")

frames = []
silence_threshold = 500  # adjust based on ambient noise
silent_chunks = 0
max_silent_chunks = 10    # stop after ~10 chunks of silence (~0.5s if CHUNK=1024 @16kHz)

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        # Compute audio volume to detect silence
        audio_array = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_array).mean()
        if volume < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        # If we've encountered sufficient consecutive silent chunks, assume end of speech
        if silent_chunks > max_silent_chunks:
            break
except KeyboardInterrupt:
    print("Stopped listening.")
finally:
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()

# Combine captured frames
audio_data = b''.join(frames)
```

In this snippet, we continuously read from the microphone until a period of silence is detected (you can also use a Voice Activity Detection tool like `webrtcvad` for more robust silence detection). We accumulate audio chunks in `frames` and then break out of the loop when the user stops speaking (or on manual interruption). The result is `audio_data`, a bytes object containing the recorded audio.

**Latency tip:** By processing audio in small increments (here ~1024 samples per chunk, ~64ms of audio), you can achieve near-real-time detection of when the user finishes speaking. Lowering `CHUNK` (or using a callback stream) can reduce latency further, but may increase CPU usage. The `sounddevice` library in streaming mode uses roughly 7% CPU on a single core for capturing audio continuousl ([using webrtcvad in realtime application · Issue #29 · wiseman/py-webrtcvad · GitHub](https://github.com/wiseman/py-webrtcvad/issues/29#:~:text=,))】, which is quite efficient.

## Transcribing Audio with Whisper (Speech-to-Text)

Once we have the raw audio bytes from the microphone, the next step is to transcribe it to text. OpenAI’s Whisper model is state-of-the-art for speech recognition and is available via OpenAI’s API (model name `"whisper-1"` for the large-v2 model) or as open-source packages for local inference. 

**Using OpenAI’s Whisper API:** The OpenAI API provides a convenient `transcriptions` endpoint. We can send the recorded audio and get a transcript. This approach is simple but introduces some latency due to the network round-trip and processing time (the Whisper large model is heavy). The benefit is you get high accuracy without setting up local models.

Here's how to call the Whisper API using the `openai` Python library (make sure you have `openai.api_key` set):

```python
import openai
from scipy.io import wavfile

# If our audio_data is in memory (bytes), we may need to write it to a file-like object:
with open("temp.wav", "wb") as f:
    f.write(audio_data)

# Transcribe using OpenAI Whisper API
audio_file = open("temp.wav", "rb")
transcript = openai.Audio.transcribe("whisper-1", file=audio_file)
user_text = transcript['text']
print("🗣️ User said:", user_text)
```

In this example, we wrote the audio to a temporary WAV file for simplicity. The Whisper API expects an audio file (in formats like WAV, MP3, or M4A). The result `transcript` is a dictionary; the transcribed text is in `transcript['text']`. 

*Alternative:* If you want to avoid file I/O for better performance, you can use Python’s `io.BytesIO` to create a file-like object from `audio_data` and pass that to `openai.Audio.transcribe`. Also, when using local Whisper models (e.g., via the `whisper` or `faster-whisper` Python packages), you can pass the raw audio array directly to the model without writing to di ([Real-time Speech Recognition with PyAudio and Faster Whisper (Issue with Temporary Files) · SYSTRAN faster-whisper · Discussion #827 · GitHub](https://github.com/SYSTRAN/faster-whisper/discussions/827#:~:text=Faster%20whisper%20accepts%20both%20buffers,and%20sampled%20at%2016khz))7】. For example, with OpenAI’s `whisper` library:

```python
import whisper
model = whisper.load_model("small")  # load a smaller model for speed
# Convert bytes to numpy array for whisper (16-bit PCM to float32)
audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
result = model.transcribe(audio_np, fp16=False)
user_text = result['text']
```

Using a smaller Whisper model (like `"small"` or `"base"`) can significantly reduce transcription time at the cost of some accuracy. This might be preferable for real-time interaction to minimize delay.

**Note:** Whisper (in both API and local forms) is **not truly streaming** – it processes audio in chunks (the model has a 30-second window) and typically requires the full audio segment to produce a transcripti ([Generating Subtitles in Real-Time with OpenAI Whisper and PyAudio - Bruno Scheufler](https://brunoscheufler.com/blog/2023-03-12-generating-subtitles-in-real-time-with-openai-whisper-and-pyaudio#:~:text=To%20get%20started%2C%20I%20had,by%20recording%20our%20computer%20audio))8】. This means our system will introduce a brief pause while the audio segment is transcribed. Keeping audio segments short (a few seconds) helps keep this delay low.

## Sending the Transcribed Text to GPT-4 (ChatCompletion API)

With the user’s speech converted to text, we can now send it to the GPT-4 model using OpenAI’s ChatCompletion API. We’ll maintain a conversation history to provide context, as GPT-4’s responses can be improved by giving it the previous dialogue turns. The API expects a list of message dictionaries with roles (`"system"`, `"user"`, `"assistant"`).

Here’s how to call the ChatCompletion endpoint with the user’s transcribed text, assuming we keep a list `conversation` of messages:

```python
# Append the new user message to the conversation history
conversation.append({"role": "user", "content": user_text})

# Call GPT-4 ChatCompletion API
response = openai.ChatCompletion.create(
    model="gpt-4",              # or the appropriate GPT-4 model name
    messages=conversation,
    temperature=0.7,            # adjust as needed
    max_tokens=500,             # limit the response length
    stream=False                # we will demonstrate streaming below
)
assistant_reply = response['choices'][0]['message']['content']
conversation.append({"role": "assistant", "content": assistant_reply})

print("🤖 GPT-4:", assistant_reply)
```

This sends the conversation (including the latest user query) to GPT-4 and retrieves the assistant’s reply. We print the text response and also store it in `conversation` for context in future turns. 

**Maintaining Context:** By appending messages to the `conversation` list, each API call includes the recent history, enabling GPT-4 to produce contextual, conversational answers. Be mindful of the token limits (GPT-4 can handle a large context, but very long conversations might need trimming of older turns).

**Streaming Responses:** For a more interactive feel and lower latency on long responses, you can enable streaming. Streaming allows you to start processing the response token-by-token as it’s generated, rather than waiting for the full response. Using the OpenAI Python SDK, set `stream=True` and iterate over the response:

```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=conversation,
    stream=True
)
streamed_reply = ""
for chunk in response:
    if 'choices' in chunk:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            token = delta['content']
            streamed_reply += token
            print(token, end="", flush=True)  # print partial response
print()  # newline after completion
assistant_reply = streamed_reply
conversation.append({"role": "assistant", "content": assistant_reply})
```

In streaming mode, GPT-4’s reply is built up incrementally. We collect the tokens in `streamed_reply`. This is useful for **textual** feedback (e.g., showing a typing effect). If you intend to **speak the response**, you could start feeding these partial tokens into a TTS engine. However, many TTS systems need the full sentence to produce natural speech, so an easier approach is to stream-print for the user to read, and simultaneously or afterward do TTS on the complete reply.

OpenAI’s streaming capability can significantly improve perceived latency: you start getting tokens as soon as they are genera ([Using the ChatGPT streaming API from Python | Simon Willison’s TILs](https://til.simonwillison.net/gpt3/python-chatgpt-streaming-api#:~:text=for%20chunk%20in%20openai.ChatCompletion.create%28%20model%3D%22gpt,%7D%5D%2C%20stream%3DTrue%2C)) ([Using the ChatGPT streaming API from Python | Simon Willison’s TILs](https://til.simonwillison.net/gpt3/python-chatgpt-streaming-api#:~:text=content%20%3D%20chunk%5B,content%2C%20end))31】, which means the assistant can start “speaking” (via TTS) before it has finalized the entire paragraph of response.

## Converting GPT-4’s Response to Audio (Text-to-Speech)

To complete the loop, we take GPT-4’s text reply and output it as spoken audio. There are a few ways to do this in a CLI environment:

- **Offline TTS libraries:** 
  - `pyttsx3` – A cross-platform library that uses the system’s speech engines (e.g., SAPI on Windows or NSSpeechSynthesizer on macOS). It doesn’t require an internet connection and is easy to use.
  - `gTTS` – A Python library for Google Text-to-Speech. It requires internet access (uses Google’s TTS API behind the scenes) and produces an MP3 which you can play with an external player.
- **Online TTS APIs:** Amazon Polly, Google Cloud TTS, Microsoft Azure Cognitive Services, or specialized services like ElevenLabs can produce high-quality voices, but require API keys and introduce network latency.
- **OpenAI GPT-4o voice (future):** GPT-4o can theoretically produce audio output directly as part of the API respo ([Hello GPT-4o | OpenAI](https://openai.com/index/hello-gpt-4o/#:~:text=GPT%E2%80%914o%20,GPT%E2%80%914o%20is))41】, but at the moment of writing, public API access for direct TTS is not available. (OpenAI has *voice samples* for ChatGPT’s voice mode, but not an API endpoint for arbitrary TTS.)

For simplicity, we’ll use **pyttsx3** to generate speech from the assistant’s text. It’s entirely Python-based and works in CLI. First, install it (`pip install pyttsx3`). Then:

```python
import pyttsx3

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)  # speaking rate (words per minute)
tts_engine.setProperty('volume', 1.0)  # volume (0.0 to 1.0)

# Speak the assistant's reply
tts_engine.say(assistant_reply)
tts_engine.runAndWait()
```

This will use the default voice on your system to speak the text. You can adjust the speaking rate or choose a different voice via `tts_engine.setProperty('voice', voice_id)` if desired (platform-specific voice IDs). 

Alternatively, if you prefer `gTTS`:
```python
from gtts import gTTS
from io import BytesIO
import os
tts = gTTS(assistant_reply, lang='en')
tts.save("reply.mp3")
os.system("ffplay -nodisp -autoexit reply.mp3")  # using ffmpeg/ffplay to play audio
```
This will save the speech to an MP3 and play it with `ffplay` (an ffmpeg-based player) on the command line. The trade-off is a dependency on an external tool and a slight delay to fetch the TTS from Google.

**Playing audio output:** If using `pyttsx3`, it plays audio directly via the system’s TTS. If using other methods that produce an audio file, you can play it using Python as well (for example, use `pyaudio` or `sounddevice` to play raw audio, or use a simple library like `simpleaudio` or `playsound`). The `sounddevice` library provides a convenient playback function too, e.g., `sounddevice.play(audio_array, samplerate)`.

## Example: Putting It All Together

Below is a simplified **conversation loop** combining the steps above. The program listens to the microphone, transcribes speech to text, gets a GPT-4 response, and speaks it out. This repeats until the user stops the program:

```python
import openai, pyaudio, numpy as np, pyttsx3

# Initialize audio input and TTS engine
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                frames_per_buffer=1024)
tts_engine = pyttsx3.init()

conversation = [ 
    {"role": "system", "content": "You are a helpful voice assistant."} 
]

print("Start speaking. Press Ctrl+C to exit.\n")
try:
    while True:
        # Record until silence
        frames = []
        silent_chunks = 0
        while True:
            data = stream.read(1024)
            frames.append(data)
            audio_array = np.frombuffer(data, np.int16)
            if np.abs(audio_array).mean() < 500:  # silence threshold
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks > 10:  # ~0.5s of silence
                break

        # Transcribe audio with Whisper API
        audio_bytes = b''.join(frames)
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)
        audio_file = open("temp.wav", "rb")
        transcript = openai.Audio.transcribe("whisper-1", file=audio_file)
        user_text = transcript['text'].strip()
        if user_text == "":
            continue  # ignore if no speech detected
        print(f"🗣️ You: {user_text}")

        # Get GPT-4 response
        conversation.append({"role": "user", "content": user_text})
        response = openai.ChatCompletion.create(model="gpt-4", messages=conversation)
        assistant_reply = response['choices'][0]['message']['content']
        conversation.append({"role": "assistant", "content": assistant_reply})
        print(f"🤖 GPT-4: {assistant_reply}")

        # Speak the response
        tts_engine.say(assistant_reply)
        tts_engine.runAndWait()
        # Loop back to listen for the next user input
except KeyboardInterrupt:
    print("Exiting...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
```

This code ties everything together in a simple loop. In a production scenario, you would want to handle errors, do cleanup of the temp audio file, and perhaps improve silence detection (the above uses a basic amplitude threshold).

## Minimizing Latency and Improving Responsiveness

Real-time voice interaction should feel snappy. Here are some tips to reduce latency:

- **Stream the GPT-4 Response:** Enable `stream=True` in the ChatCompletion request to start receiving GPT-4’s answer token by  ([Using the ChatGPT streaming API from Python | Simon Willison’s TILs](https://til.simonwillison.net/gpt3/python-chatgpt-streaming-api#:~:text=for%20chunk%20in%20openai.ChatCompletion.create%28%20model%3D%22gpt,%7D%5D%2C%20stream%3DTrue%2C))-L22】. You can begin processing or even speaking the answer *while* it’s being generated, cutting down wait time for long responses.
- **Optimize Whisper Transcription:** If using the Whisper API, send audio as soon as the user stops speaking (don’t wait unnecessarily). If using a local model, you can even perform *continuous transcription* on a separate thread while listening. Some advanced implementations use Whisper’s voice activity detection (VAD) to segment speech in real ([whisper-live-transcription/standalone-poc/live-transcribe.py at main · gaborvecsei/whisper-live-transcription · GitHub](https://github.com/gaborvecsei/whisper-live-transcription/blob/main/standalone-poc/live-transcribe.py#:~:text=vad_filter%3DTrue%2C)) ([whisper-live-transcription/standalone-poc/live-transcribe.py at main · gaborvecsei/whisper-live-transcription · GitHub](https://github.com/gaborvecsei/whisper-live-transcription/blob/main/standalone-poc/live-transcribe.py#:~:text=segments%2C%20_%20%3D%20whisper))L626】. You could also use the `webrtcvad` library to detect end-of-speech quickly, as mentioned. Smaller Whisper models or faster ones (like `tiny` or `base`) will transcribe faster, reducing delay at the cost of accuracy.
- **Use Concurrent Processing:** Overlap operations when possible. For example, while GPT-4 is formulating a response, you could pre-initialize the TTS engine or prepare audio output. Similarly, if the TTS engine allows streaming input (some TTS APIs do), feed it partial text as you get streamed tokens.
- **Audio Chunking and VAD:** Capturing audio in short chunks (e.g., 20-100 ms) and using VAD to trigger the transcription as soon as the user finishes talking will minimize dead air. OpenAI’s own GPT-4o demos show response times around ~320ms which approach human-level turn-t ([Hello GPT-4o | OpenAI](https://openai.com/index/hello-gpt-4o/#:~:text=GPT%E2%80%914o%20,GPT%E2%80%914o%20is))L141】. While our pipeline will have more overhead, careful tuning of each component (perhaps transcribing and generating in parallel streams) can make it quite responsive.
- **Library Choices:** Use efficient libraries. PyAudio and sounddevice both are capable of realtime audio – choose the one you are more comfortable with. Some developers prefer sounddevice for its simplicity and maintenance s ([using webrtcvad in realtime application · Issue #29 · wiseman/py-webrtcvad · GitHub](https://github.com/wiseman/py-webrtcvad/issues/29#:~:text=edited))L238】, but PyAudio is also solid. If you’re already using PyAudio successfully, there’s no need to switch. Ensure you run the audio capture and playback in non-blocking modes or separate threads if you plan to do full-duplex audio (speaking while listening, which is advanced). For playing audio, lightweight tools (like using sounddevice’s playback or PyAudio output stream) can be faster than launching an external player. Avoid unnecessary file writes (we used a temp file for clarity, but you can transcribe from memory to save time).

## Choosing Audio Streaming Libraries: PyAudio vs WebRTC vs FFmpeg

**PyAudio** and **sounddevice** are the go-to choices for capturing and playing audio in Python CLI applications, as discussed. They give you direct access to microphone and speaker streams with low latency. 

**WebRTC:** If you were building a web-based or networked application, WebRTC would be relevant. In Python, one might use `aiortc` or similar libraries to handle audio streams over the network. WebRTC excels in scenarios where you need echo cancellation, networking, and real-time communication between remote peers. However, for a local CLI tool, WebRTC is not necessary for capturing local audio. One part of WebRTC that *is* useful is its VAD (Voice Activity Detection) module (available as `webrtcvad` in Python) which can be used alongside PyAudio/sounddevice to detect speech segments in the audio stream.

**FFmpeg:** FFmpeg is a powerful multimedia framework. In this context, FFmpeg can be used to record or play audio via command-line, or to convert audio formats. For example, one could use FFmpeg to directly capture microphone input to a file or pipe (`ffmpeg -f avfoundation -i ":0" output.wav` on Mac, or `-f dshow` on Windows, etc.). However, integrating FFmpeg via system calls or even using `ffmpeg-python` is generally less straightforward for interactive applications. FFmpeg shines when you need to manipulate or convert audio (e.g., trim silence, change format) in a s ([Best Python Tools for Manipulating Audio Data - Deepgram Blog ⚡️ | Deepgram](https://deepgram.com/learn/best-python-audio-manipulation-tools#:~:text=FFMPEG%20is%20a%20well%20known,the%C2%A0sys%C2%A0and%C2%A0subprocess%C2%A0libraries%20that%20are%20native%20to))-L29】, but for real-time capture/playback, PyAudio or sounddevice (which use PortAudio under the hood) are more direct solutions.

In summary, **use PyAudio or sounddevice for direct real-time audio** in a CLI app. You can achieve latency on the order of tens of milliseconds for capturing audio. Incorporate **Whisper** for transcription – either via OpenAI’s API (for accuracy) or a lighter local model (for speed). Send the text to **OpenAI’s GPT-4** via the ChatCompletion API to get a response, and use a **TTS library** to speak the answer. With careful optimization (streaming and parallelism), you can maintain a smooth, back-and-forth conversational experience with minimal lag.

**References:** Real-time Whisper integration and its constr ([Generating Subtitles in Real-Time with OpenAI Whisper and PyAudio - Bruno Scheufler](https://brunoscheufler.com/blog/2023-03-12-generating-subtitles-in-real-time-with-openai-whisper-and-pyaudio#:~:text=To%20get%20started%2C%20I%20had,by%20recording%20our%20computer%20audio))-L38】, streaming responses from OpenAI’ ([Using the ChatGPT streaming API from Python | Simon Willison’s TILs](https://til.simonwillison.net/gpt3/python-chatgpt-streaming-api#:~:text=for%20chunk%20in%20openai.ChatCompletion.create%28%20model%3D%22gpt,%7D%5D%2C%20stream%3DTrue%2C)) ([Using the ChatGPT streaming API from Python | Simon Willison’s TILs](https://til.simonwillison.net/gpt3/python-chatgpt-streaming-api#:~:text=content%20%3D%20chunk%5B,content%2C%20end))-L31】, and OpenAI’s vision for low-latency multi-modal GPT-4o interac ([Hello GPT-4o | OpenAI](https://openai.com/index/hello-gpt-4o/#:~:text=GPT%E2%80%914o%20,GPT%E2%80%914o%20is))L141】 all informed this implementation. Developers have noted the trade-offs between audio libraries (PyAudio vs soundde ([using webrtcvad in realtime application · Issue #29 · wiseman/py-webrtcvad · GitHub](https://github.com/wiseman/py-webrtcvad/issues/29#:~:text=edited))L238】, and OpenAI’s forthcoming Realtime API hints at future improvements with native speech-to-speech su ([Realtime API (Advanced Voice Mode) Python Implementation - API - OpenAI Developer Community](https://community.openai.com/t/realtime-api-advanced-voice-mode-python-implementation/964636#:~:text=The%20Realtime%20API%20enables%20you,as%20well%20as%20function%20calling))-L91】. The solution above uses currently available tools to achieve a comparable voice interface today. Enjoy your voice-enabled GPT-4 assistant! 🎙️🤖

