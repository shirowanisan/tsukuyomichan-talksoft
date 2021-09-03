# シロワニさんのつくよみちゃんトークソフト

「シロワニさんのつくよみちゃんトークソフト」は、私（シロワニさん）がフリー素材キャラクター「つくよみちゃん」の無料公開音声データを使用して作成した自作トークソフトです。

テキストを入力すれば、つくよみちゃんの声質で読み上げ音声を出力します。ボイスロイドやゆっくりのようなものと説明した方がイメージしやすいかもしれません。

# 利用規約・免責事項

こちらを必ずご確認の上、ご使用ください。

https://shirowanisan.com/tyc-talksoft

# Google Colabでの使用

◆落ち着いた読み上げ（従来版）

[つくよみちゃんトークソフト-v.1.0.0-GoogleColab](https://colab.research.google.com/drive/1VX1pPK-A5KHcUnpBz__IYXzVR-93ECan?usp=sharing)

◆感情的な読み上げ（声が明るい時と暗い時があります）

[つくよみちゃんトークソフト-v.1.1.0-GoogleColab](https://colab.research.google.com/drive/1x8T1FE_Gt3baJetEperSYhkVOvEBhX1p?usp=sharing)

◆感情的な読み上げ（イントネーション改善版）

[つくよみちゃんトークソフト-v.1.2.0-GoogleColab](https://colab.research.google.com/drive/1zYzc4qJF_sTp8Vt51wI718sgMckfy85e?usp=sharing)

# ローカルでGUIでの使用（エンジニア向け）

## 動作環境

| OS      | 動作 |
| ------- | ---------------------------------------------------------- |
| Windows | ⭕️（試していませんが、たぶん動くと思います。） |
| Mac     | ⭕️（Intel MacのBig Sur バージョン11.2.3でのみ試しました。） |
| Linux   | ⭕️（試していませんが、たぶん動くと思います。） |

pythonは3.8.9で動作確認しました。

## 環境構築

```bash
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

# クレジット表記

こちらのコードでは、フリー素材キャラクター「つくよみちゃん」の無料公開音声データから作られた音声合成モデルを使用する可能性があります。

■つくよみちゃんコーパス（CV.夢前黎）

https://tyc.rei-yumesaki.net/material/corpus/

© Rei Yumesaki
