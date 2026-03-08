# pyannote で話者分離を使うための設定手順

GUI で「話者分離エンジン」を **pyannote（HFトークン要）** にし、精度の高い話者分離を使うための手順です。

---

## 1. パッケージのインストール

**重要**: pyannote.audio **4.x** は **torch 2.8 以上**を要求します。いま **torch 2.5.1+cu121** など 2.8 未満を使っている場合は、次のどちらかで合わせてください。

### 方法A: いまの PyTorch のまま使う（手軽）

pyannote.audio を **3.x** に固定し、torch 2.5 と共存させます。

```powershell
pip install "pyannote.audio>=3,<4"
```

- すでに `pip install pyannote.audio torchaudio` で 4.x が入ってしまった場合は、上記のあとに `pip install "pyannote.audio>=3,<4"` で 3.x にダウングレードできます。
- このアプリのコードは 3.x / 4.x どちらでも動作する想定です。

### 方法B: PyTorch を 2.8 に上げて pyannote 4.x を使う

torch / torchaudio を 2.8 に揃えてから、pyannote.audio 4.x を入れます。

```powershell
pip install --upgrade torch torchaudio
pip install pyannote.audio torchaudio
```

- GPU を使う場合は、PyTorch 公式の [Get Started](https://pytorch.org/get-started/locally/) で OS・CUDA を選び、表示される `--index-url` 付きのコマンドで **torch 2.8 系の CUDA 版**を入れてから、その上で `pip install pyannote.audio` してください。
- 2.8 の CUDA 用 wheel がまだない環境では、方法A（pyannote 3.x）を推奨します。

---

- このアプリでは WAV を **torchaudio** で読み、pyannote に渡すため **torchcodec は不要**です（`AudioDecoder is not defined` を避けています）。

---

## 2. Hugging Face トークンの作成

1. https://huggingface.co/settings/tokens を開く
2. **New token** で作成し、**Read** 権限を付与
3. 発行された `hf_...` をコピーし、アプリの「HFトークン」欄に貼り付ける

---

## 3. gated モデルへのアクセス許可（必須）

pyannote は複数の「gated」モデルを参照します。**それぞれのページで利用条件に同意しないと 403/401 になります。**

ブラウザで **ログインした状態** のまま、次のページを開き、表示に従って **Agree / Accept / Request access** を押してください。

| モデル | URL |
|--------|-----|
| speaker-diarization-3.1（メイン） | https://huggingface.co/pyannote/speaker-diarization-3.1 |
| segmentation-3.0（内部で使用） | https://huggingface.co/pyannote/segmentation-3.0 |
| speaker-diarization-community-1（内部で使用） | https://huggingface.co/pyannote/speaker-diarization-community-1 |

- 会社名・Web サイトなどの入力欄があれば、空にせず記入する
- 同意後、反映まで数分かかることがあります

---

## 4. アプリでの設定

1. 「話者分離エンジン」で **pyannote（HFトークン要）** を選択
2. 「HFトークン」に 2 で作ったトークンを貼り付け
3. 「話者分離」は **常に音声話者分離** または **自動** のまま実行

---

## 5. 速度について

- **pyannote は CPU だと数分〜十分かかることがあります。** 実用上は次のいずれかを推奨します。
  - **速度重視**: 話者分離エンジンに **「ローカル（diarize）」** を選ぶ（HF 不要・かなり短時間で完了）
  - **pyannote のまま短縮**: **GPU (CUDA)** を使うと、アプリが自動でバッチサイズを拡大し処理が短縮されます。手順は **PYANNOTE_GPU_PLAN.md** を参照
- `config.py` で `PYANNOTE_SEGMENTATION_BATCH_SIZE` / `PYANNOTE_EMBEDDING_BATCH_SIZE` を指定すると、バッチサイズを固定できます（GPU 時は 16 前後が目安）。

---

## トラブルシュート

| 症状・エラー | 対処 |
|--------------|------|
| **pyannote-audio 4.0.4 requires torch>=2.8.0, but you have torch 2.5.1** | 上記「1. パッケージのインストール」の **方法A** を使い、`pip install "pyannote.audio>=3,<4"` で pyannote を 3.x に揃える（いまの torch 2.5 のまま利用可能）。 |
| **エラーメッセージが出ずにアプリが落ちる**（pyannote 実行時） | `torchcodec` の DLL 読み込みで落ちていることがあります。**`pip uninstall torchcodec`** を実行してから再起動。本アプリは音声を torchaudio で読むため torchcodec は不要です。 |
| `AudioDecoder is not defined` | 本アプリは torchaudio で音声を読むため通常は発生しません。念のため `pip install torchaudio` を実行 |
| `403` / `401` / `not in the authorized list` | 上記 3 つのモデルすべてで Agree したか確認。トークンは「同意したアカウント」で作成したものか確認 |
| `cublas64_12.dll` など CUDA エラー | CUDA 12.x をインストールするか、pyannote は CPU でも動作します（遅くなります） |
| **Could not load symbol cudnnGetLibConfig** のあとアプリが落ちる | cuDNN が未導入またはバージョン不一致です。**`config.py` で `PYANNOTE_FORCE_CPU = True`** にすると pyannote を CPU で動かし落ちなくなります（デフォルトで True にしています）。GPU を使う場合は cuDNN を PyTorch のバージョンに合わせて導入してください。 |

エラーが出た場合は、`%APPDATA%\TeamsTranscriber\last_error.txt` にトレースバックが保存されます。
