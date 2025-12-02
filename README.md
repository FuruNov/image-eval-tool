# Image Evaluation Tool

信号復元（画像再構成）の精度を評価・比較するための Streamlit ベースの GUI ツールです。
Ground Truth (Reference) と提案手法 (Method) の画像をアップロードすることで、様々な指標や可視化を用いて詳細な比較分析を行うことができます。

## 主な機能

### 1. 定量評価指標 (Metrics)
以下の指標を自動計算し、サイドバーに表示します。
*   **PSNR (Peak Signal-to-Noise Ratio)**: ピーク信号対雑音比
*   **SSIM (Structural Similarity)**: 構造的類似性
*   **GMSD (Gradient Magnitude Similarity Deviation)**: 勾配の類似性に基づく画質評価指標
*   **No-Reference Metrics (Local Only)**: `pyiqa` が利用可能な環境では、以下の指標が利用可能です。
    *   **NIQE**: 自然画像統計 (NSS) に基づく指標。学習データ（人間のスコア）を使用せず、画像の「自然さ」からの逸脱を測定します。再構成アーティファクトの検出に有効です。
    *   **MANIQA**: Vision Transformer と Attention 機構を用いた深層学習ベースの指標。人間の主観評価スコア（MOS）と高い相関を持ちます。
    *   **MUSIQ**: マルチスケール Transformer を用いた指標。異なる解像度やアスペクト比の画像に対してロバストな評価が可能です。
    *   **CLIPIQA**: 大規模言語画像モデル CLIP を用いたゼロショット指標。"Good photo" / "Bad photo" のテキストプロンプトとの類似度に基づいてスコアを算出します。

### 2. 多角的な可視化 (Views)
タブ切り替えにより、様々な視点から画像を評価できます。
*   **Spatial (Standard)**: 再構成画像と誤差マップ (Error Map) を表示。
*   **Flicker Test (Blink)**: Reference と Method を交互に表示するアニメーション (GIF) で、微細な違いや位置ズレを視覚的に検出します。
*   **CIELAB ΔE Map**: 人間の視覚特性に近い CIELAB 色空間における色差 (ΔE) をヒートマップで表示します。
*   **ROI Check**: 全体像（ROI枠付き）と拡大画像（Zoomed ROI）を並べて表示し、コンテキストと詳細を同時に確認できます。
*   **Line Profile 📈**: 指定したライン上の画素値をプロットし、エッジの鋭さやアーティファクトを確認できます (Interactive Plotly)。
*   **Sobel Edge**: Sobel フィルタによるエッジ検出結果と、その誤差を表示します。
*   **GMS Map**: 局所的な勾配類似性 (GMS) マップを表示します。
*   **Frequency (FFT)**: フーリエ変換による周波数スペクトルを表示し、周波数領域での誤差を確認できます。
*   **Histogram**: 輝度ヒストグラムと残差 (Residual) の分布を表示し、統計的なバイアスやノイズを分析します。
*   **Log-Gradient Histogram**: 勾配強度の対数分布をヒストグラムで比較し、テクスチャの再現性を評価します。
*   **Image Slider**: スライダーを使って Reference と Method を重ねて比較できます。

### 3. Global Zoom (ROI Analysis)
サイドバーの **"Enable ROI Zoom"** を有効にすると、全てのビューが選択した関心領域 (ROI) にフォーカスします。
*   微細なテクスチャやエッジの再現性を詳細に確認できます。
*   FFT やヒストグラムも ROI 内のデータに基づいて再計算されます。

### 4. その他の機能
*   **レイアウト調整**: サイドバーの **"Reference Column Ratio"** スライダーで、Reference 画像の表示幅を調整できます。
*   **デバッグモード**: **"Use Default Images (Debug)"** を有効にすると、アップロード画像を無視してデフォルトのサンプル画像（Blur, Noise, JPEG）を読み込みます。
*   **デフォルト画像**: Blur, Noise に加え、JPEG 圧縮アーティファクトを含むサンプル画像が利用可能です。
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
   *   サイドバーの **"Zoom ROI Settings"** で ROI (Crop) の位置やサイズ、**"Line Profile Settings"** でラインプロファイルの位置を調整します。

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
