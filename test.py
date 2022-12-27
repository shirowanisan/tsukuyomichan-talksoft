import numpy as np
MAX_WAV_VALUE = 32768.0
fs = 24000
import simpleaudio as sa
import time
#from python_talksoft import TsukuyomichanTalksoft
from tsukuyomichan_talksoft import TsukuyomichanTalksoft
talksoft = TsukuyomichanTalksoft(model_version='v.1.0.0')
start = time.perf_counter()
wav = talksoft.generate_voice("やぁ",0)
print("End:"+str(time.perf_counter()-start))
wav = wav * MAX_WAV_VALUE
wav = wav.astype(np.int16)
sa.play_buffer(wav, 1, 2, fs)