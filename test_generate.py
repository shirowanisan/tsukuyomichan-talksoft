import numpy as np
MAX_WAV_VALUE = 32768.0
fs = 24000
import time
from onnx_talksoft import TsukuyomichanTalksoft as onnxtalk
onnxtalk1 = onnxtalk(model_version='v.1.0.0')
onnxtalk2 = onnxtalk(model_version='v.1.1.0')
onnxtalk3 = onnxtalk(model_version='v.1.2.0')
from tsukuyomichan_talksoft import TsukuyomichanTalksoft as torchtalk
torchtalk1 = torchtalk(model_version='v.1.0.0')
torchtalk2 = torchtalk(model_version='v.1.1.0')
torchtalk3 = torchtalk(model_version='v.1.2.0')
def test_onnx_say_01():
    start = time.perf_counter()
    wav = onnxtalk1.generate_voice("やぁ")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
def test_onnx_say_02():
    start = time.perf_counter()
    wav = onnxtalk2.generate_voice("こんにちは")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
def test_onnx_say_03():
    start = time.perf_counter()
    wav = onnxtalk3.generate_voice("やったね")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)

def test_torch_say_01():
    start = time.perf_counter()
    wav = torchtalk1.generate_voice("やぁ")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
def test_torch_say_02():
    start = time.perf_counter()
    wav = torchtalk2.generate_voice("こんにちは")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
def test_torch_say_03():
    start = time.perf_counter()
    wav = torchtalk3.generate_voice("やったね")
    print("End:"+str(time.perf_counter()-start))
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
