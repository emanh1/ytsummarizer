import os
import json
import subprocess
import threading
import webvtt
import yt_dlp as youtube_dl
import tkinter as tk
from tkinter import ttk, messagebox
from openai import OpenAI
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))


def download_subtitles(video_url):
    result_text.insert(tk.END, "Downloading Subtitles...\n")
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'skip_download': True,
        'outtmpl': '%(id)s.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def extract_text_from_vtt(vtt_file_path):
    vtt = webvtt.read(vtt_file_path)
    seen=[]
    for i in range(len(vtt)):
        texts = vtt[i].text.split("\n")
        for text in texts:
            if text not in seen:
                seen.append(text)
    
    return " ".join(seen)


def download_audio(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def transcribe_audio(file_path, openai=False):
    if openai:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(file_path, "rb"),
            response_format="text",
        )
        return transcription.text
    else:
        return transcribe_audio_locally(file_path, "vosk-model-small-en-us-0.15")


def transcribe_audio_locally(file_path, model_path):
    if not os.path.exists(model_path):
        raise Exception(f"Model not found at {model_path}")
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)

    with subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            file_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
    ) as process:
        transcription = ""
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = rec.Result()
                transcription += json.loads(result)["text"] + " "

        final_result = rec.FinalResult()
        transcription += json.loads(final_result)["text"]
        print(transcription)
        return transcription


def summarize_text(text, openai=False):
    min_length = int(min_length_entry.get())
    max_length = int(max_length_entry.get())
    length_penalty = float(length_penalty_entry.get())
    
    if openai:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this text: {text}",
                }
            ],
            model="gpt-4o-mini",
        )
        return response.choices[0].message["content"]
    else:
        import torch
        from transformers import BartTokenizer, BartForConditionalGeneration

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result_text.insert(tk.END, f"Using device: {device}\n")
        model.to(device)

        inputs = tokenizer.encode(f"summarize: {text}", return_tensors="pt", truncation=True).to(device)
            

        summary_ids = model.generate(
            inputs, 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=length_penalty, 
            num_beams=4, 
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary




def summarize_youtube_video(video_url, use_openai=False):
    video_id = video_url.split("v=")[-1]
    
    progress_bar["value"] = 20
    root.update_idletasks()
    
    try:
        download_subtitles(video_url)
        text_to_summarize = extract_text_from_vtt(f"{video_id}.en.vtt")
    except FileNotFoundError:
        result_text.insert(tk.END, "Cant get subtitles, trying to get audio...\n")
        download_audio(video_url)
        text_to_summarize = transcribe_audio(
            f"{video_id}.mp3", use_openai
        )
    
    copy_button.config(state=tk.NORMAL)
    copy_button.config(command=lambda: copy_to_clipboard(text_to_summarize))
    progress_bar["value"] = 50
    root.update_idletasks()
    
    summary = summarize_text(text_to_summarize, use_openai)
    print("finished")
    progress_bar["value"] = 100
    return summary


def summarize():
    video_url = entry.get()
    use_openai = openai_var.get()
    if not video_url:
        messagebox.showerror("Input Error", "Please enter the video URL")
        return
    
    progress_bar["value"] = 0
    result_text.delete(1.0, tk.END)
    
    def run_summarization():
        try:
            summary = summarize_youtube_video(video_url, use_openai)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, summary)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    threading.Thread(target=run_summarization).start()


def copy_to_clipboard(transcript):
    root.clipboard_clear()
    root.clipboard_append(transcript)
    messagebox.showinfo("Copy to Clipboard", "Transcript copied to clipboard!")

root = tk.Tk()
root.title("YouTube Video Summarizer")


root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(9, weight=1)


tk.Label(root, text="YouTube URL:", font=("Arial", 12)).grid(
    row=0, column=0, padx=10, pady=(10, 5), sticky="w"
)
entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.grid(row=0, column=1, padx=10, pady=(10, 5), sticky="ew")
entry.insert(0, "https://www.youtube.com/watch?v=Sv1PL7IyPIg")


openai_var = tk.BooleanVar()
openai_checkbox = tk.Checkbutton(
    root, text="Use OpenAI services to summarize", variable=openai_var, font=("Arial", 12)
)
openai_checkbox.grid(
    row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w"
)


min_length_label = tk.Label(root, text="Min Length:", font=("Arial", 12))
min_length_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
min_length_entry = tk.Entry(root, width=10, font=("Arial", 12))
min_length_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
min_length_entry.insert(0, "100")


max_length_label = tk.Label(root, text="Max Length:", font=("Arial", 12))
max_length_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
max_length_entry = tk.Entry(root, width=10, font=("Arial", 12))
max_length_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
max_length_entry.insert(0, "500")


length_penalty_label = tk.Label(root, text="Length Penalty:", font=("Arial", 12))
length_penalty_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
length_penalty_entry = tk.Entry(root, width=10, font=("Arial", 12))
length_penalty_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")
length_penalty_entry.insert(0, "1.0")


button_frame = tk.Frame(root)
button_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)

summarize_button = tk.Button(button_frame, text="Summarize", command=summarize, font=("Arial", 12))
summarize_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

copy_button = tk.Button(button_frame, text="Copy Transcript", state=tk.DISABLED, command=copy_to_clipboard, font=("Arial", 12))
copy_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")


progress_bar = ttk.Progressbar(
    root, orient="horizontal", length=400, mode="determinate"
)
progress_bar.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")


result_text = tk.Text(root, wrap=tk.WORD, height=15, width=80, font=("Arial", 12))
result_text.grid(row=7, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

root.mainloop()



