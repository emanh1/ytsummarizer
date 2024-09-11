
# Youtube Summarizer

Summarize youtube videos


## Getting started
### 1. Install Python
https://www.python.org/downloads/

### 2. Install Youtube Summarizer
Download the source files for Youtube Summarizer, then install the dependencies via pip.
```
git clone https://github.com/emanh1/ytsummarizer
cd ytsummarizer
pip install -r requirements.txt
```
Then execute main.py
```
python main.py
```

### 3. Transcribing using video's audio
In the case of the video not having subtitles or automatic subtitles, the program needs [ffmpeg](https://ffmpeg.org) and a [vosk model](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip). Place both inside the running directory. Make sure to place the extracted folder and the ffmpeg binaries inside the working directory.
## Environment Variables

`OPENAI` - OpenAI API key if you plan to use their models to summarize.


## TODO

Add support for other languages.