<p align="center">
 <img src="https://user-images.githubusercontent.com/111404437/210527746-2dbeb692-6f4a-41f7-b639-e7f7c19ba539.png">
</p>
<h2 align="center">ONNXで実行するから早い</h2>

「シロワニさんのつくよみちゃんトークソフト」は、シロワニさんがフリー素材キャラクター「つくよみちゃん」の無料公開音声データを使用して作成したシロワニさんの自作トークソフトです。

テキストを入力すれば、つくよみちゃんの声質で読み上げ音声を出力します。ボイスロイドやゆっくりのようなものと説明した方がイメージしやすいかもしれません。

そして、「シロワニさんのつくよみちゃんトークソフト」のモデルのロードを非同期にして起動を早くしただけのものがこのレポジトリです。<br>
しかし、[COEIROINK](https://coeiroink.com/)さんの方が早い気がしてきました。<br>
このブランチはCPUで「やぁ」を7秒程度で生成することができます<br>
ほぼほぼONNXに移行しました<br>
何もしなくてもモデルを用意してくれるようにしました<br>
たぶん「シロワニさんのつくよみちゃんトークソフト」より早いです(自分調べ)
# 利用規約・免責事項

こちらを必ずご確認の上、ご使用ください。

https://shirowanisan.com/tyc-talksoft

# Google Colabでの使用

ColabだとGUIがないので高速化ができないのでありません

# ローカルでGUIでの使用（エンジニア向け）

## 動作環境

| OS      | 動作 |
| ------- | ---------------------------------------------------------- |
| Windows | ⭕ 「Windows 11」で動作を確認しました|
| Mac     | ⭕ (多分動きます) |
| Linux   | ⭕️ (多分動きます) |

pythonは3.7系で動作確認しました。

GPUを使って計算する場合は、PC内でcudaの設定をして、インストールするpytorchをgpu用のものにかえてください。

## 環境構築

CMakeをインストールしてからPathを通してから以下のコードを実行してください。
CMakeをインストールした場合GCCなどのコンパイラが必要です。
```bash
$ git clone https://github.com/FanaticPond3462/tsukuyomichan-talksoft
$ pip install -r requirements.txt
```

## 起動

```bash
$ python main.py
```

## 使い方

- テキスト欄にテキストを入力
- seed欄にseed値を入力
- 「作る」ボタンを押し、作成完了の表示を待つ
- 「聞く」ボタンを押すと、作成した音声が再生される
- 「保存」を押すと「./output」フォルダの下にwavファイルが保存される

## 開発者向け
### Pythonモジュールとしてインストール
```bash
pip install git+https://github.com/FanaticPond3462/tsukuyomichan-talksoft
```
### 💬 使用方法
以下のpythonのコードでonnxの推論エンジンで合成することができます。
```python:app.py
import numpy as np
import simpleaudio as sa
from onnx_talksoft import TsukuyomichanTalksoft
MAX_WAV_VALUE = 32768.0
fs = 24000
talksoft = TsukuyomichanTalksoft(model_version='v.1.0.0')
wav = talksoft.generate_voice("こんにちは",0)
wav = wav * MAX_WAV_VALUE
wav = wav.astype(np.int16)
sa.play_buffer(wav, 1, 2, fs)
```
pytorchの推論エンジンで合成するには以下のpythonのコードで合成することができます。
```python:app.py
import numpy as np
import simpleaudio as sa
from tsukuyomichan_talksoft import TsukuyomichanTalksoft
MAX_WAV_VALUE = 32768.0
fs = 24000
talksoft = TsukuyomichanTalksoft(model_version='v.1.0.0')
wav = talksoft.generate_voice("こんにちは",0)
wav = wav * MAX_WAV_VALUE
wav = wav.astype(np.int16)
sa.play_buffer(wav, 1, 2, fs)
```
> **Note**
>```python
>　talksoft = TsukuyomichanTalksoft()
>　talksoft.generate_voice("こんにちは")  
>```
> でも行けます
# クレジット表記

本コンテンツは「シロワニさんのつくよみちゃんトークソフト」のソースコードを使用しています。

■シロワニさんのつくよみちゃんトークソフト
https://shirowanisan.com/tyc-talksoft
© shirowanisan

こちらのコードでは、フリー素材キャラクター「つくよみちゃん」の無料公開音声データから作られた音声合成モデルを使用する可能性があります。

■つくよみちゃんコーパス（CV.夢前黎）

https://tyc.rei-yumesaki.net/material/corpus/

© Rei Yumesaki
