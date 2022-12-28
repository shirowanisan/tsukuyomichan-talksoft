import numpy as np
MAX_WAV_VALUE = 32768.0
fs = 24000
import time
#from python_talksoft import TsukuyomichanTalksoft
from tsukuyomichan_talksoft import TsukuyomichanTalksoft
talksoft = TsukuyomichanTalksoft(model_version='v.1.0.0')
def test_say_01():
    start = time.perf_counter()
    wav = talksoft.generate_voice("やぁ")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
def test_say_02():
    start = time.perf_counter()
    wav = talksoft.generate_voice("こんにちは")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
def test_say_03():
    start = time.perf_counter()
    wav = talksoft.generate_voice("やったね")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
def test_say_04():
    start = time.perf_counter()
    wav = talksoft.generate_voice("そうだね")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
def test_say_05():
    start = time.perf_counter()
    wav = talksoft.generate_voice("違うよ")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
