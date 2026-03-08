# pyannote で GPU を使うためのプラン

話者分離エンジン「pyannote（HFトークン要）」を **GPU (CUDA)** で動かすための手順と方針です。

---

## GPU で安定して動かすには（要点）

1. **PyTorch は CUDA 用の公式 wheel で入れる**  
   `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` のように **index-url で cu121 を指定**すると、多くの環境で cuDNN も同梱されたビルドが入り、GPU が安定しやすいです。
2. **NVIDIA ドライバを新しめに**  
   [NVIDIA ドライバ](https://www.nvidia.com/Download/index.aspx) を入れ、CUDA 12.x 対応のバージョン（例: R525 以降）にしておく。
3. **アプリの挙動**  
   GPU 利用に失敗した場合（cuDNN のシンボルエラーなど）は、**自動で CPU に切り替えて再実行**するため、アプリは落ちずに完了します。そのときはログに「GPU 実行でエラーがあったため CPU で再実行します」と出ます。  
   どうしても GPU で通したい場合は、上記 1・2 を確認するか、`config.py` の `PYANNOTE_FORCE_CPU = False` のまま公式の CUDA 用 PyTorch だけを使うようにしてください。

---

## 1. 現状

- **アプリ側**: `diarizer.py` で `torch.cuda.is_available()` が True のとき `pipeline.to(torch.device("cuda"))` を実行し、ログに「GPU (CUDA) を使用」と出します。
- **問題になりがちな点**:
  - PyTorch が **CPU 用** のまま入っている（`pip install torch` のみで CUDA 版になっていない）
  - NVIDIA ドライバや **CUDA ランタイムのバージョン** が PyTorch の CUDA ビルドと合っていない
  - **faster-whisper** は別の CUDA 依存（cublas64_12.dll 等）を持ち、pyannote と「どちらかだけ動く」ことがある

---

## 2. 前提条件（ハードウェア・OS）

| 項目 | 内容 |
|------|------|
| GPU | NVIDIA GPU（CUDA 対応） |
| OS | Windows 10/11（64bit） |
| ドライバ | [NVIDIA ドライバ](https://www.nvidia.com/Download/index.aspx) をインストールし、使用したい CUDA バージョンに対応したドライバ以上にする（例: CUDA 12.1 なら R525 以降を推奨） |

---

## 3. プラン概要

1. **CUDA 対応 PyTorch を入れる**  
   - デフォルトの `pip install torch` は CPU 版のことが多いため、**CUDA 用の index-url を指定**してインストールする。
2. **pyannote / torchaudio はそのまま**  
   - `pip install pyannote.audio torchaudio` は PyTorch に合わせて動く。先に CUDA 版 PyTorch を入れておく。
3. **動作確認**  
   - Python で `import torch; print(torch.cuda.is_available())` が `True`、かつアプリで話者分離実行時にログに「GPU (CUDA) を使用」と出ることを確認する。
4. **（任意）CPU 強制オプション**  
   - 環境によっては「pyannote だけ CPU」にしたい場合があるため、設定や GUI で **pyannote 用に CPU を強制**する項目を追加する案。

---

## 4. 手順（推奨順）

### 4.1 仮想環境を用意する

```powershell
cd TeamsTranscriber
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4.2 既存の torch を外す（CUDA 版で揃え直す場合）

```powershell
pip uninstall -y torch torchvision torchaudio
pip cache purge
```

### 4.3 CUDA 12.1 用 PyTorch を入れる

- 多くの環境で扱いやすい **CUDA 12.1** 用:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- 公式: [PyTorch Get Started](https://pytorch.org/get-started/locally/) で OS=Windows, Package=pip, CUDA=12.1 を選ぶと同様のコマンドが表示されます。
- **CUDA 11.8** に合わせる場合（ドライバが古い等）:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4.4 アプリ用パッケージを入れる

```powershell
pip install pyannote.audio torchaudio
pip install faster-whisper
# その他 requirements.txt があれば
pip install -r requirements.txt
```

- `torchaudio` は 4.3 で既に入っている場合はスキップ可。**PyTorch と同じ index-url（cu121/cu118）で揃える**と安全です。

### 4.5 動作確認

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

- `CUDA: True` かつデバイス名が表示されれば、pyannote も同じ CUDA を使用します。
- アプリで「話者分離エンジン: pyannote」を選び、話者分離を実行してログに「GPU (CUDA) を使用」と出れば GPU で動いています。

---

## 5. バージョン対応の目安

| 環境 | PyTorch (pip) | 備考 |
|------|----------------|------|
| CUDA 12.x 系 | `cu121` 推奨 | ドライバ R525 以降など |
| CUDA 11.x 系 | `cu118` | 古いドライバ・環境用 |
| GPU なし / 安定優先 | 通常の `pip install torch`（CPU） | pyannote は CPU でも動作（遅い） |

- **faster-whisper** は内部で別の CUDA ライブラリを参照することがあり、`cublas64_12.dll` 不足で落ちる場合は「Whisper は CPU にフォールバック」する実装になっています。pyannote は **PyTorch の CUDA** だけ見るため、PyTorch を cu121 で入れれば pyannote は GPU を使えます。

---

## 6. （任意）pyannote を CPU に固定するオプション

- 理由の例: 同じ GPU を Whisper と pyannote で奪い合いたくない、または特定環境で pyannote だけ GPU で不安定な場合。
- **実装案**:
  - `config.py` に `PYANNOTE_DEVICE: "cuda" | "cpu" | "auto"` を追加する。
  - `diarizer.py` の `_diarize_with_pyannote` 内で、`device = config の値 or (cuda if available else cpu)` として `pipeline.to(device)` する。
  - GUI の「詳細設定」などに「話者分離で使うデバイス: 自動 / GPU / CPU」を追加し、設定に保存して `diarizer` に渡す。

---

## 7. トラブルシュート（GPU まわり）

| 症状 | 対処 |
|------|------|
| **Could not load symbol cudnnGetLibConfig** のあと落ちる / GPU で動かない | アプリは **自動で CPU に切り替えて再実行**するため落ちません。GPU を通したい場合は、**4.3 の CUDA 用 index-url（cu121 等）** で PyTorch を入れ直す（多くの公式 wheel に cuDNN 同梱）。ドライバを R525 以降に更新しても改善することがあります。 |
| `torch.cuda.is_available()` が False | 上記のとおり **CUDA 用 index-url** で PyTorch を再インストール。ドライバと CUDA バージョンの対応を確認。 |
| `cublas64_12.dll` などで落ちる | 多くは **faster-whisper** 側。Whisper は CPU フォールバック済み。pyannote 単体なら PyTorch の CUDA が使われているか 4.5 で確認。 |
| ログに「CPU を使用」と出る | PyTorch が CPU 版の可能性。`pip show torch` で `cu121` 等が含まれるか確認し、必要なら 4.2–4.3 をやり直す。 |
| 複数 GPU | 現状は `torch.device("cuda")` で 0 番のみ。複数 GPU 対応は pyannote の `pipeline.to(device)` を `cuda:1` などに変える拡張で対応可能。 |

---

## 8. まとめ

- **やること**: CUDA 対応 PyTorch を `--index-url .../cu121`（または cu118）で入れ、その上で `pyannote.audio` と `torchaudio` をインストールする。
- **確認**: `torch.cuda.is_available()` が True かつ、アプリで話者分離時に「GPU (CUDA) を使用」と出る。
- **任意**: 設定で「pyannote だけ CPU」を選べるようにする拡張を検討する。

これに従えば、pyannote で GPU を使うための道筋が揃います。
