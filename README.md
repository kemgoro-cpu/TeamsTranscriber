# Teams Transcriber

Teams 会議録画（MP4）と文字起こしファイル（.vtt / .docx）から、Whisper による高精度テキストと話者情報付きの文字起こしを生成するデスクトップアプリです。

**バージョン**: 1.0.0

---

## 必要な環境

- **Python 3.10 以上**
- **ffmpeg**（PATH にインストール済み） — [https://ffmpeg.org/](https://ffmpeg.org/)

---

## インストール

```powershell
cd TeamsTranscriber
pip install -r requirements.txt
```

話者分離を**ローカル（diarize）**で使う場合（オプション）:

```powershell
pip install diarize
```

話者分離を**pyannote**で使う場合（オプション）:

- [PYANNOTE_SETUP.md](PYANNOTE_SETUP.md) の手順に従い、`torch` と `pyannote.audio` をインストールし、Hugging Face トークンとモデル利用同意を済ませてください。

---

## 起動方法

```powershell
python main.py
```

GUI が開いたら、MP4 ファイルと文字起こしファイル（.vtt または .docx）を指定し、Whisper モデルや話者分離オプションを選んで「開始」を押します。

- **操作**: 文字起こし開始は **F5** キーでも実行できます。

---

## 話者分離の選択

| モード | 説明 |
|--------|------|
| **自動** | Teams の文字起こしが単一話者のときは音声から話者分離、複数話者のときは Teams の話者情報を使用 |
| **常に音声話者分離** | 常に音声から話者分離（diarize または pyannote）を使用 |
| **使用しない** | Teams の話者・時間情報のみ使用（話者分離は行わない） |

- **ローカル（diarize）**: Hugging Face 不要・短時間。話者登録で名前を付ける場合は HF トークンが必要です。
- **pyannote（HFトークン要）**: 高精度。Hugging Face のトークンとモデル利用同意が必要です。詳しくは [PYANNOTE_SETUP.md](PYANNOTE_SETUP.md) を参照してください。HF トークンは環境変数 `HF_TOKEN` で指定するか、`pip install keyring` で OS の credential store に保存できます（いずれもオプション）。

処理を短時間で完了させたい場合は、[処理を短時間で完了させるには.md](処理を短時間で完了させるには.md) を参照してください。

---

## 実行ファイル化（任意）

PyInstaller で exe 化する場合の例です。ffmpeg は exe に同梱されないため、利用者側で PATH にインストールするか、配布時に別途案内してください。

```powershell
pip install pyinstaller
pyinstaller --onefile --windowed --name TeamsTranscriber main.py
```

生成された exe は `dist/` に出力されます。依存 DLL や Python ランタイムは `--onefile` でまとまりますが、faster-whisper のモデルは初回実行時にダウンロードされます。

---

## 免責事項

- 本ソフトの利用は**自己責任**でお願いします。
- Microsoft Teams および Hugging Face の利用規約に従ってご利用ください。
- 本ソフトは Microsoft および Hugging Face とは無関係です。

---

## サードパーティ・クレジット

本ソフトは以下のオープンソース等を利用しています。

| 名前 | 用途 | ライセンス等 |
|------|------|--------------|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 文字起こし | MIT |
| [python-docx](https://github.com/python-openxml/python-docx) | .docx 解析 | MIT |
| [ffmpeg](https://ffmpeg.org/) | 音声抽出 | LGPL/GPL |
| [diarize](https://github.com/kemgoro-cpu/diarize) | 話者分離（オプション） | 各リポジトリを参照 |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | 話者分離（オプション） | MIT |

---

## ライセンス

本プロジェクトは [LICENSE](LICENSE) に記載のとおりです。
