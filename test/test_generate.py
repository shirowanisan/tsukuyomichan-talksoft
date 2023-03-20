import numpy as np
import time

from tsukuyomichan_talksoft import onnx_talksoft
#onnxで推論するためのクラスを取得しています.
onnxtalk1 = onnx_talksoft.TsukuyomichanTalksoft(model_version='v.1.0.0')
onnxtalk2 = onnx_talksoft.TsukuyomichanTalksoft(model_version='v.1.1.0')
onnxtalk3 = onnx_talksoft.TsukuyomichanTalksoft(model_version='v.1.2.0')

from tsukuyomichan_talksoft import tsukuyomichan_talksoft
#pytorchで推論するためのクラスを取得しています.
torchtalk1 = tsukuyomichan_talksoft.TsukuyomichanTalksoft(model_version='v.1.0.0')
torchtalk2 = tsukuyomichan_talksoft.TsukuyomichanTalksoft(model_version='v.1.1.0')
torchtalk3 = tsukuyomichan_talksoft.TsukuyomichanTalksoft(model_version='v.1.2.0')

def test_onnx_say_01():
    start = time.perf_counter()  #タイマーの開始
    wav = onnxtalk1.generate_voice("やぁ")  #モデルのバージョン v1.0.0 でonnxでやぁの生成
    print("End:"+str(time.perf_counter()-start))  #タイマーの終了
def test_onnx_say_02():
    start = time.perf_counter()  #タイマーの開始
    wav = onnxtalk2.generate_voice("こんにちは")  #モデルのバージョン v1.1.0 でonnxでこんにちはの生成
    print("End:"+str(time.perf_counter()-start))  #タイマーの終了
def test_onnx_say_03():
    start = time.perf_counter()  #タイマーの開始
    wav = onnxtalk3.generate_voice("やったね")  #モデルのバージョン v1.2.0 でonnxでやったねの生成
    print("End:"+str(time.perf_counter()-start))  #タイマーの終了

def test_torch_say_01():
    start = time.perf_counter()  #タイマーの開始
    wav = torchtalk1.generate_voice("やぁ")  #モデルのバージョン v1.0.0 でpytorchでやぁの生成
    print("End:"+str(time.perf_counter()-start))  #タイマーの終了
def test_torch_say_02():
    start = time.perf_counter()  #タイマーの開始
    wav = torchtalk2.generate_voice("こんにちは")  #モデルのバージョン v1.1.0 でpytorchでこんにちはの生成
    print("End:"+str(time.perf_counter()-start))  #タイマーの終了
def test_torch_say_03():
    start = time.perf_counter()  #タイマーの開始
    wav = torchtalk3.generate_voice("やったね")  #モデルのバージョン v1.2.0 でpytorchでやったねの生成
    print("End:"+str(time.perf_counter()-start))  #タイマーの終了