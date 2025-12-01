# Image Evaluation Tool

信号復元（画像再構成）の精度を評価・比較するための Streamlit ベースの GUI ツールです。
Ground Truth (Reference) と提案手法 (Method) の画像をアップロードすることで、様々な指標や可視化を用いて詳細な比較分析を行うことができます。

## 主な機能

### 1. 定量評価指標 (Metrics)
以下の指標を自動計算し、サイドバーに表示します。
*   **PSNR (Peak Signal-to-Noise Ratio)**: ピーク信号対雑音比
*   **SSIM (Structural Similarity)**: 構造的類似性
*   **GMSD (Gradient Magnitude Similarity Deviation)**: 勾配の類似性に基づく画質評価指標
*   **NIQE (Natural Image Quality Evaluator)**: 参照画像不要 (No-Reference) の自然画質評価指標。再構成特有の不自然さを評価します (要 `pyiqa`)。

### 2. 多角的な可視化 (Views)
タブ切り替えにより、様々な視点から画像を評価できます。
*   **Spatial (Standard)**: 再構成画像と誤差マップ (Error Map) を表示。誤差マップはダウンロード可能です。
*   **Line Profile 📈**: 指定したライン上の画素値をプロットし、エッジの鋭さやアーティファクトを確認できます (Interactive Plotly)。
*   **Sobel Edge**: Sobel フィルタによるエッジ検出結果と、その誤差を表示します。
*   **GMS Map**: 局所的な勾配類似性 (GMS) マップを表示します。
*   **Frequency (FFT)**: フーリエ変換による周波数スペクトルを表示し、周波数領域での誤差を確認できます。
*   **Histogram**: 輝度ヒストグラムと残差 (Residual) の分布を表示し、統計的なバイアスやノイズを分析します。
*   **Image Slider**: スライダーを使って Reference と Method を重ねて比較できます。

### 3. Global Zoom (ROI Analysis)
サイドバーの **"Enable ROI Zoom"** を有効にすると、全てのビューが選択した関心領域 (ROI) にフォーカスします。
*   微細なテクスチャやエッジの再現性を詳細に確認できます。
*   FFT やヒストグラムも ROI 内のデータに基づいて再計算されます。

### 4. その他の機能
*   **デフォルト画像**: 画像をアップロードせずに起動した場合、自動的にサンプル画像を読み込み、即座に動作確認が可能です。
*   **スマートな比較表**:
    *   各指標の良化方向（↑/↓）を明示します。
    *   最も優れた値（Best Metric）を**太字・緑色**でハイライト表示します。

## インストール方法

Python 3.12 以上が推奨されます。

### 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

または `uv` を使用する場合:

```bash
uv sync
```

## 使い方

1. アプリケーションの起動:
   ```bash
   streamlit run app.py
   ```
   または `uv` を使用する場合:
   ```bash
   uv run streamlit run app.py
   ```

2. **画像のアップロード**:
   *   サイドバーの **"1. Reference"** に Ground Truth 画像をアップロードします。
   *   サイドバーの **"2. Methods"** に比較したい手法の画像をアップロードします（複数可）。

3. **評価・分析**:
   *   メインエリアに画像と評価結果が表示されます。
   *   タブを切り替えて詳細な分析を行います。
   *   サイドバーの **"Analysis Settings"** で ROI (Crop) の位置やサイズ、ラインプロファイルの位置を調整します。

## 設定 (Configuration)

`app_config.toml` を編集することで、デフォルトの挙動や表示するタブをカスタマイズできます。

```toml
[sidebar.toggles]
profile = { label = "Enable Line Profile 📈", default = true }
sobel = { label = "Enable Sobel Edge", default = false }
...
```

## ファイル構成

*   `app.py`: アプリケーションのエントリーポイント。メインの UI ロジック。
*   `views.py`: 各タブの描画ロジック (Strategy Pattern)。
*   `utils.py`: 画像処理、指標計算、プロット用のヘルパー関数。
*   `config.py`: 定数定義。
*   `app_config.toml`: アプリケーション設定ファイル。

## License

MIT License
