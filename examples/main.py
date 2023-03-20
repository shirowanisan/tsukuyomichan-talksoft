import os
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from scipy.io.wavfile import write
import numpy as np
import simpleaudio as sa

#非同期でインポート
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thread")
def talkinit():
    from tsukuyomichan_talksoft import onnx_talksoft
    return onnx_talksoft.TsukuyomichanTalksoft(model_version='v.1.2.0')
future = pool.submit(talkinit)
# config
MAX_WAV_VALUE = 25000
fs = 24000
wav = []
seed = 0
text = ''
voice = ''
talksoft = "ぬるぽ"

# GUI
root = tk.Tk()
root.geometry("550x500")
root.title('シロワニさんのつくよみちゃんトークソフト')

textbox_label = tk.Label(text='テキスト')
textbox_label.grid(row=0, column=0, padx=30, pady=15)

textbox = tk.Entry(width=50)
textbox.grid(row=1, column=0, padx=30, pady=15)

seed_label = tk.Label(text='seed')
seed_label.grid(row=2, column=0, padx=30, pady=15)

seed_box = tk.Entry(width=5)
seed_box.grid(row=3, column=0, padx=10, pady=15)

load_label = tk.Label(text='何も作ってません')
load_label.grid(row=4, column=0, padx=30, pady=15)


def make_tsukuyomichan_voice():
    global talksoft
    if talksoft == "ぬるぽ":
        talksoft = future.result()
        pool.shutdown()
    global text, seed, wav
    text = textbox.get()
    seed = int(seed_box.get())

    wav = talksoft.generate_voice(text, seed)
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
    load_label['text'] = f"「{text}」「{seed}」で作りました"


def play_voice():
    sa.play_buffer(wav, 1, 2, fs)


def save_voice():
    os.makedirs('output', exist_ok=True)
    write(f"output/{text}_{seed}.wav", rate=fs ,data=wav)


run_button = tk.Button(text='作る', command=make_tsukuyomichan_voice)
run_button.grid(row=5, column=0, padx=30, pady=15)

run_button = tk.Button(text='聞く', command=play_voice)
run_button.grid(row=6, column=0, padx=10, pady=15)

run_button = tk.Button(text='保存', command=save_voice)
run_button.grid(row=7, column=0, padx=10, pady=15)

root.mainloop()
