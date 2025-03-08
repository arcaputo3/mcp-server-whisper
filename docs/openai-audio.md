# Sending Audio Files to OpenAI's GPT-4 ChatCompletion API (GPT-4o)

OpenAI’s **GPT-4o** model (the audio-enabled GPT-4 preview) allows you to include audio in your chat conversations. Below, we address each part of your question with references to official documentation and announcements:

## 1. Direct Audio Input via ChatCompletion API

**Yes – with GPT-4o, you can send audio directly as a user message.** The Chat Completions API was updated (in late 2024) to support audio inputs and outputs via the `gpt-4o-audio-preview` model ([Audio support in the Chat Completions API - Announcements - OpenAI Developer Community](https://community.openai.com/t/audio-support-in-the-chat-completions-api/983234#:~:text=The%20Chat%20Completions%20API%20now,Completions%20API%20lets%20you%3A)). This means you can pass an audio file *without manual transcription*, and the model will process it (using built-in speech recognition) as part of the conversation. In short, GPT-4o can accept audio alongside or instead of text in user messages ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=OpenAI%20promised%20this%20at%20DevDay,easier%20to%20write%20code%20against)). 

> **Note:** This direct audio input **only works with models that support audio**, such as the `gpt-4o-audio-preview` snapshot. Standard GPT-4 models (without the audio feature) do **not** accept raw audio in the ChatCompletion API – in those cases you must transcribe the audio first (see Section 3).

## 2. Required API Call Format (Headers & Parameters)

To send an audio file via the ChatCompletion API, you must construct the request with the proper JSON structure and headers:

- **Endpoint:** `POST https://api.openai.com/v1/chat/completions`  
- **Headers:** 
  - `Authorization: Bearer YOUR_OPENAI_API_KEY`  
  - `Content-Type: application/json`  
- **JSON Body Fields:**  
  - **`model`** – set to the audio-capable model, e.g. `"gpt-4o-audio-preview"` (or a specific dated snapshot like `"gpt-4o-audio-preview-2024-10-01"`).  
  - **`modalities`** – an array indicating the modalities in use. Include `"audio"` here (along with `"text"`) if you plan to send or receive audio. For example: `"modalities": ["text", "audio"]` ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=curl%20https%3A%2F%2Fapi.openai.com%2Fv1%2Fchat%2Fcompletions%20%5C%20,wav)).  
  - **`audio`** – *optional*, used to specify audio output settings. If you want the assistant’s **response** in audio form, include an `audio` object with a voice and format (e.g. `"voice": "alloy", "format": "wav"` for a spoken response) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=curl%20https%3A%2F%2Fapi.openai.com%2Fv1%2Fchat%2Fcompletions%20%5C%20,wav)). This is not required if you only send audio in the input and want a text reply.  
  - **`messages`** – the conversation messages array. Here you include the user message with the audio content. The user message `content` should be an **array** of content parts, where one part is the audio. For an audio file, use an object of type `"input_audio"` containing the base64-encoded audio data and format. For example: 

```json
"messages": [
  {
    "role": "user",
    "content": [
      { "type": "text", "text": "Describe the audio:" },
      { 
        "type": "input_audio",
        "input_audio": {
          "data": "<BASE64_ENCODED_AUDIO>",
          "format": "wav"
        }
      }
    ]
  }
]
``` 

In the above JSON, the user's message consists of a text prompt **and** an audio file. The audio file data is base64-encoded and labeled with its format (wav in this case) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=modalities%3A%20%5B,input_audio%3A)) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=%7B%20type%3A%20,%7D%20%7D%20%5D)). You can also send **only** an audio clip (without additional text) by making the `content` array contain just the `"input_audio"` object ([How to make audio file input with gpt-4o - Microsoft Q&A](https://learn.microsoft.com/en-au/answers/questions/2126353/how-to-make-audio-file-input-with-gpt-4o#:~:text=messages%20%3D%20%5B%7B)). 

**Summary of required syntax:** Use the `gpt-4o-audio-preview` model, include `"audio"` in `modalities`, and embed the audio file in the message content as shown above. Ensure you send the request as JSON with the proper headers. OpenAI’s example (via cURL) illustrates this format, where the request JSON includes `model`, `modalities`, an `audio` config, and the `messages` with the audio content ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=curl%20https%3A%2F%2Fapi.openai.com%2Fv1%2Fchat%2Fcompletions%20%5C%20,wav)) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=,)).

## 3. Whisper Integration (Is Pre-Transcription Required?)

If you **do not** use an audio-enabled model (or if audio input wasn’t supported), then **yes – you must transcribe the audio first** using OpenAI’s Whisper API, and then send the text to the chat API. Prior to GPT-4o’s introduction, the typical workflow was: 

1. **Transcribe audio to text** with the Whisper **transcriptions** endpoint (e.g. `POST /v1/audio/transcriptions` or using `openai.Audio.transcribe()` in Python). This uses the `whisper-1` model under the hood to convert speech to text. The Whisper API supports a variety of audio formats and file lengths (see Section 4) and returns the recognized text. For example, you would send the audio file (MP3, WAV, etc.) to the Whisper endpoint and get back `"text": "Transcribed content..."`. 
2. **Send the transcribed text** as the user message content in `ChatCompletion.create()`.

In summary, if direct audio input is **not** supported in the ChatCompletion call (for example, when using GPT-4 *without* the audio preview), you *must* use Whisper first. OpenAI’s documentation notes that Whisper can handle audio of any length up to **25 MB** in file size ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=audio%3F)), and it supports common file types like MP3, MP4, WAV, etc. ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=What%20file%20formats%20are%20supported%3F)). So you would utilize Whisper to get the text, then pass that text into `ChatCompletion.create()` as you would with any normal text prompt.

*GPT-4o changes this flow* by allowing you to skip the explicit Whisper step – the model will internally transcribe the audio input. But this only applies if you use the special audio-enabled model. For all other cases, Whisper transcription first is required (there’s no way to directly feed audio into the standard GPT-4 ChatCompletion API).

## 4. Supported Audio Formats and Limitations

**Supported Audio File Formats:** The underlying speech-to-text technology (Whisper) accepts a wide range of audio formats. According to OpenAI’s documentation, supported formats include: **M4A, MP3, WebM, MP4, MPGA, WAV, and MPEG** ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=What%20file%20formats%20are%20supported%3F)) ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=)). In practice, the GPT-4o ChatCompletion API has so far explicitly demonstrated using **WAV** and **MP3** for input audio files ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=From%20the%20documentation%20it%20looks,mp3)). (WAV is uncompressed audio, and MP3 is a common compressed format. M4A is also a common container for AAC audio.) It’s safest to use one of these widely supported types (e.g. `wav` or `mp3` in the `"format"` field) unless OpenAI expands support. If your audio is in an unsupported format, you may need to convert it to one of the above (for example, convert OGG to MP3) before sending.

**Size and Duration Restrictions:** OpenAI imposes a size limit on audio files. The file must be **<= 25 MB** when sending to the API ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=audio%3F)) ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=Up%20to%2025MB)). There isn’t a strict *duration* limit given in seconds, because it depends on the format/encoding (25 MB of an MP3 could be several hours of audio, whereas 25 MB WAV might be just a few minutes). The guidance is **“any length of audio is fine as long as the file is under 25 MB.”** ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=audio%3F)) If you have a longer audio, you should break it into smaller chunks or compress it to stay under this size. 

Additionally, keep in mind that extremely long audio (even if under 25 MB) will consume a lot of **audio tokens** when processed by GPT-4o. Audio tokens are billed separately (for example, the preview pricing was about $0.06 per minute of audio input) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=Audio%20inputs%20are%20charged%20at,6%20cents)) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=,costs%20approximately%2024%C2%A2%20per%20minute)). Very large audio inputs could also approach model limits – although audio is handled outside the normal text token context, practical limits (like processing time and cost) will apply. 

**Summary:** Use common audio formats (WAV or MP3 are recommended for GPT-4o) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=From%20the%20documentation%20it%20looks,mp3)). Ensure the file is within size limits (~25 MB). If your audio is longer, you may need to send it in segments. These limits align with the Whisper transcription API’s capabilities, since GPT-4o uses similar tech under the hood.

## 5. Example Python Implementation (Audio as User Input)

Below is an example of how to send an audio file as a user message using Python. This code uses the OpenAI Python library’s `ChatCompletion.create()` and the GPT-4o audio model. It reads an audio file, encodes it to base64, and includes it in the conversation:

```python
import openai, base64

# (1) Load and encode the audio file (e.g., "question.wav")
with open("question.wav", "rb") as audio_file:
    audio_bytes = audio_file.read()
audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

# (2) Construct the message with the audio content
messages = [
    {
        "role": "user",
        "content": [
            # Optionally include a text prompt along with the audio:
            # {"type": "text", "text": "Please translate this audio to English:"},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav"
                }
            }
        ]
    }
]

# (3) Call the ChatCompletion API with GPT-4o model
response = openai.ChatCompletion.create(
    model="gpt-4o-audio-preview-2024-10-01",   # audio-capable model
    modalities=["text"],                      # expecting a text response; use ["text","audio"] if you also want spoken audio reply
    messages=messages
)

# (4) Extract the assistant's reply
assistant_reply = response["choices"][0]["message"]["content"]
print("Assistant response:", assistant_reply)
```

**Explanation:** 

- We open a WAV file (`question.wav`) in binary mode and base64-encode its contents. The encoded string is placed in the message content under an object of type `"input_audio"`. We specify the format as `"wav"`.  
- We send the request with `model="gpt-4o-audio-preview-2024-10-01"`. The `modalities=["text"]` indicates we are providing audio (implicitly) and we only want a text answer. (If we wanted the answer in audio form as well, we would use `modalities:["text","audio"]` and include an `audio={"voice": "...", "format": "..."}` parameter to pick the voice and output format ([Audio support in the Chat Completions API - Announcements - OpenAI Developer Community](https://community.openai.com/t/audio-support-in-the-chat-completions-api/983234#:~:text=completion%20%3D%20client.chat.completions.create%28%20model%3D%27gpt,Then%2C%20in%20a%20slow%2C%20brittish)) ([Audio support in the Chat Completions API - Announcements - OpenAI Developer Community](https://community.openai.com/t/audio-support-in-the-chat-completions-api/983234#:~:text=,out)).)  
- The assistant will process the audio input and respond. In this example, we print the text reply. If we had requested an audio output, the response JSON would include an `"audio": { "data": "<base64_audio>" , ... }` field for the assistant’s message. We could then decode that base64 to save or play the audio answer.

**Important:** Make sure you have the **latest version of OpenAI’s Python SDK** installed. Older versions may not recognize the `modalities` or `input_audio` parameters, resulting in errors ([Audio support in the Chat Completions API - Announcements - OpenAI Developer Community](https://community.openai.com/t/audio-support-in-the-chat-completions-api/983234#:~:text=_j%20%20October%2026%2C%202024%2C,10%3A08pm%20%2011)). Upgrade via `pip install --upgrade openai` to a version that supports GPT-4o. 

Using this approach, you can have a multi-turn conversation where the user’s turns are spoken (audio files) and the assistant’s replies can be text and/or spoken audio. Under the hood, GPT-4o will transcribe the audio (using Whisper) and incorporate it into the conversation, allowing for seamless voice interactions in the chat format ([Audio support in the Chat Completions API - Announcements - OpenAI Developer Community](https://community.openai.com/t/audio-support-in-the-chat-completions-api/983234#:~:text=,Realtime%20API%2C%20you%20can%20use)).

**References:**

- OpenAI Developer Forum – *“Audio support in the Chat Completions API”* (Oct 2024 announcement) ([Audio support in the Chat Completions API - Announcements - OpenAI Developer Community](https://community.openai.com/t/audio-support-in-the-chat-completions-api/983234#:~:text=The%20Chat%20Completions%20API%20now,Completions%20API%20lets%20you%3A)).  
- OpenAI Help Center – Whisper Audio API FAQ (supported formats and size limits) ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=audio%3F)) ([Whisper Audio API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq#:~:text=What%20file%20formats%20are%20supported%3F)).  
- Simon Willison’s Blog – *“Experimenting with audio input and output for the OpenAI Chat Completion API”* (Oct 18, 2024) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=modalities%3A%20%5B,input_audio%3A)) ([Experimenting with audio input and output for the OpenAI Chat Completion API](https://simonwillison.net/2024/Oct/18/openai-audio/#:~:text=From%20the%20documentation%20it%20looks,mp3)).  
- OpenAI Developer Forum – GPT-4o audio Q&A (usage of `input_audio` and `modalities`) ([Audio support in the Chat Completions API - Announcements - OpenAI Developer Community](https://community.openai.com/t/audio-support-in-the-chat-completions-api/983234#:~:text=,out)) ([Audio support in the Chat Completions API - Announcements - OpenAI Developer Community](https://community.openai.com/t/audio-support-in-the-chat-completions-api/983234#:~:text=_j%20%20October%2026%2C%202024%2C,10%3A08pm%20%2011)).