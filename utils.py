import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import image_similarity_measures.quality_metrics as ism

def compute_snr(ref, dist):
    """Signal-to-Noise Ratio (SNR) の計算"""
    # 0除算回避
    noise_power = np.sum((ref - dist) ** 2)
    if noise_power == 0:
        return float('inf')
    
    signal_power = np.sum(ref ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

@st.cache_data
def compute_fsim(ref, dist):
    """Feature Similarity Index Measure (FSIM) の計算"""
    # image-similarity-measures は (H, W, C) または (H, W) を期待
    # 0-1 float32 から 0-255 uint8 に変換が必要な場合があるが、
    # ライブラリの実装を確認すると内部で処理されることが多い。
    # ただし、安全のため 0-255 uint8 に変換して渡すのが一般的。
    
    ref_uint8 = (ref * 255).astype(np.uint8)
    dist_uint8 = (dist * 255).astype(np.uint8)
    
    # FSIM計算 (少し時間がかかる場合がある)
    return ism.fsim(ref_uint8, dist_uint8)

@st.cache_data
def compute_gms(ref, dist, c=0.0026):
    """Gradient Magnitude Similarity (GMS) の計算"""
    # 勾配計算 (Prewittフィルタを使用するのが一般的だが、Sobelでも可)
    # ここでは既存のcompute_gradient_magnitudeを流用せず、
    # GMSDの論文に従いPrewittフィルタで実装する
    
    # グレースケール化
    if ref.ndim == 3: ref_gray = np.mean(ref, axis=2)
    else: ref_gray = ref
        
    if dist.ndim == 3: dist_gray = np.mean(dist, axis=2)
    else: dist_gray = dist
    
    # cv2.filter2DでCV_64Fに出力するため、入力をfloat64にキャスト
    ref_gray = ref_gray.astype(np.float64)
    dist_gray = dist_gray.astype(np.float64)
    
    # Prewittフィルタ
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3
    
    # 勾配強度
    gm_ref_x = cv2.filter2D(ref_gray, cv2.CV_64F, kernel_x)
    gm_ref_y = cv2.filter2D(ref_gray, cv2.CV_64F, kernel_y)
    gm_ref = np.sqrt(gm_ref_x**2 + gm_ref_y**2)
    
    gm_dist_x = cv2.filter2D(dist_gray, cv2.CV_64F, kernel_x)
    gm_dist_y = cv2.filter2D(dist_gray, cv2.CV_64F, kernel_y)
    gm_dist = np.sqrt(gm_dist_x**2 + gm_dist_y**2)
    
    # GMS Map
    gms_map = (2 * gm_ref * gm_dist + c) / (gm_ref**2 + gm_dist**2 + c)
    
    # GMSD (Deviation) = std(GMS Map)
    gmsd = np.std(gms_map)
    
    return gms_map, gmsd

def load_image(uploaded_file):
    """画像読み込み & 正規化 (0-1 float32)"""
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file).convert('RGB')
    return np.array(image).astype(np.float32) / 255.0

def to_grayscale(img):
    """RGBを簡易グレースケールに変換 (平均法)"""
    if img.ndim == 3:
        return np.mean(img, axis=2)
    return img

@st.cache_data
def compute_gradient_magnitude(img):
    """Sobelフィルタによるエッジ強度マップの計算"""
    # グレースケール化
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
        
    # X, Y方向の微分 (CV_64Fで負の値も保持して計算精度を確保)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 勾配強度 (Magnitude)
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # 視認性向上のため正規化・強調 (0-1にクリップ)
    # 通常の勾配は値が小さいので、少しゲイン(x2.0~x4.0)をかけると見やすい
    vis_magnitude = np.clip(magnitude * 4.0, 0, 1.0)
    
    return vis_magnitude

@st.cache_data
def compute_fft_spectrum(img):
    """対数振幅スペクトルの計算 (グレースケール変換後)"""
    img_gray = to_grayscale(img)
        
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    # log(1 + abs) で可視化、20はスケーリング係数（dB的表現）
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    
    # 表示用に 0-1 正規化
    m_min, m_max = magnitude_spectrum.min(), magnitude_spectrum.max()
    if m_max - m_min == 0:
        return np.zeros_like(magnitude_spectrum)
    return (magnitude_spectrum - m_min) / (m_max - m_min)

def get_crop(img, y, x, size):
    """画像の切り出し (境界チェック付き)"""
    h, w = img.shape[:2]
    y_end = min(y + size, h)
    x_end = min(x + size, w)
    return img[y:y_end, x:x_end]

def draw_roi(img, x, y, size, color):
    """画像にROI枠を描画する"""
    img_with_roi = img.copy()
    # 境界チェックは簡易的に行う（はみ出しはスライスで無視されるため）
    # 上辺
    img_with_roi[y:y+2, x:x+size] = color
    # 下辺
    img_with_roi[y+size-2:y+size, x:x+size] = color
    # 左辺
    img_with_roi[y:y+size, x:x+2] = color
    # 右辺
    img_with_roi[y:y+size, x+size-2:x+size] = color
    return img_with_roi

def plot_line_profile(ref_img, dist_img, y_index, method_name):
    """指定行の輝度プロファイルをPlotlyでプロット"""
    # グレースケール化して1次元データを取得
    ref_line = to_grayscale(ref_img)[y_index, :]
    dist_line = to_grayscale(dist_img)[y_index, :]
    x_axis = np.arange(len(ref_line))

    fig = go.Figure()
    
    # Reference: グレー、破線
    fig.add_trace(go.Scatter(
        x=x_axis, y=ref_line,
        mode='lines',
        name='Reference',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.6
    ))
    
    # Method: 赤、実線
    fig.add_trace(go.Scatter(
        x=x_axis, y=dist_line,
        mode='lines',
        name=method_name,
        line=dict(color='red', width=1.5),
        opacity=0.9
    ))
    
    fig.update_layout(
        title=dict(text=f"Line Profile (Y={y_index})", font=dict(size=14)),
        xaxis_title="Pixel X-Coordinate",
        yaxis_title="Intensity (Gray)",
        yaxis=dict(range=[-0.05, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=False,
        hovermode="x unified" # ホバー時にX座標の値をまとめて表示
    )
    
    return fig

def plot_histograms(ref_img, dist_img, method_name):
    """
    強度ヒストグラムと残差ヒストグラムをPlotlyでプロット
    """
    from plotly.subplots import make_subplots
    
    # グレースケール化 & 1次元化
    ref_gray = to_grayscale(ref_img).flatten()
    dist_gray = to_grayscale(dist_img).flatten()
    
    # 残差 (Ref - Dist)
    residual = ref_gray - dist_gray
    
    # 統計量
    res_mean = np.mean(residual)
    res_std = np.std(residual)
    from scipy.stats import kurtosis
    res_kurt = kurtosis(residual)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Intensity Histogram", "Residual Distribution (Ref - Dist)"),
        vertical_spacing=0.15
    )
    
    # 1. Intensity Histogram
    # Reference
    fig.add_trace(go.Histogram(
        x=ref_gray,
        name='Reference',
        marker_color='gray',
        opacity=0.6,
        nbinsx=100,
        histnorm='probability density'
    ), row=1, col=1)
    
    # Method
    fig.add_trace(go.Histogram(
        x=dist_gray,
        name=method_name,
        marker_color='red',
        opacity=0.6,
        nbinsx=100,
        histnorm='probability density'
    ), row=1, col=1)
    
    # 2. Residual Histogram
    fig.add_trace(go.Histogram(
        x=residual,
        name='Residual',
        marker_color='blue',
        opacity=0.7,
        nbinsx=100,
        histnorm='probability density'
    ), row=2, col=1)
    
    # レイアウト調整
    fig.update_layout(
        height=600,
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.update_xaxes(title_text="Pixel Intensity", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Residual Value", range=[-0.5, 0.5], row=2, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    
    stats = {
        "Mean Bias": res_mean,
        "Std Dev": res_std,
        "Kurtosis": res_kurt
    }
    
    return fig, stats
