import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import image_similarity_measures.quality_metrics as ism
import io
import pathlib
try:
    import torch
    import pyiqa
    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False

@st.cache_resource
def get_pyiqa_model(metric_name):
    if not HAS_PYIQA:
        return None
        
    # device='cpu' for compatibility, can be 'cuda' if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        # For MUSIQ, we might need to specify a variant, but default 'musiq' usually works (musiq-spaq)
        model = pyiqa.create_metric(metric_name, device=device)
        return model
    except Exception as e:
        st.error(f"Failed to load {metric_name} model: {e}")
        return None

def compute_pyiqa_metric(img, metric_name):
    """PyIQAを用いた汎用的な指標計算"""
    model = get_pyiqa_model(metric_name)
    if model is None:
        return float('nan')
    
    # img is (H, W, C) or (H, W), float 0-1
    # pyiqa expects (B, C, H, W), float 0-1
    
    # Convert to tensor
    if img.ndim == 2:
        img = img[:, :, None] # (H, W, 1)
        
    # (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        try:
            score = model(img_tensor)
            return score.item()
        except Exception as e:
            # Some models might fail with specific image sizes
            print(f"Error computing {metric_name}: {e}")
            return float('nan')

def compute_niqe(img):
    return compute_pyiqa_metric(img, 'niqe')

def compute_maniqa(img):
    return compute_pyiqa_metric(img, 'maniqa')

def compute_musiq(img):
    # musiq often expects specific input size or handling, but pyiqa wraps it.
    # Default musiq is usually trained on KonIQ-10k or SPAQ
    return compute_pyiqa_metric(img, 'musiq')

def compute_clipiqa(img):
    # clipiqa or clipiqa+
    return compute_pyiqa_metric(img, 'clipiqa')

def compute_piqe(img):
    return compute_pyiqa_metric(img, 'piqe')


@st.cache_data
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
def compute_gradient_kurtosis(img):
    """
    勾配の尖度 (Kurtosis) を計算する
    高いほどスパース（平坦な領域が多く、エッジが鋭い）
    """
    from scipy.stats import kurtosis
    
    # 勾配強度計算 (Sobel)
    # compute_gradient_magnitude は視認性用に正規化されているため、
    # 生の勾配を再計算する方が統計的には正確だが、ここでは簡易的に既存関数を利用するか、
    # あるいは内部で再計算する。
    # ここでは生の値を使いたいので再計算する。
    
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
        
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Fisher's definition (normal ==> 0.0)
    k = kurtosis(magnitude.flatten(), fisher=True)
    return k

@st.cache_data
def compute_raw_gradient_magnitude(img):
    """生の勾配強度を計算する (Sobel)"""
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
        
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    return magnitude

def plot_log_gradient_histogram(ref_img, dist_img, method_name):
    """
    勾配強度の対数ヒストグラムをプロットする
    横軸: log10(Gradient Magnitude + epsilon)
    縦軸: log10(Frequency)
    """
    epsilon = 1e-6
    
    # Compute gradients
    grad_ref = compute_raw_gradient_magnitude(ref_img).flatten()
    grad_dist = compute_raw_gradient_magnitude(dist_img).flatten()
    
    # Log transformation
    log_grad_ref = np.log10(grad_ref + epsilon)
    log_grad_dist = np.log10(grad_dist + epsilon)
    
    # Compute Wasserstein Distance (Earth Mover's Distance)
    wd = compute_emd(ref_img, dist_img)
    
    fig = go.Figure()
    
    # Reference
    fig.add_trace(go.Histogram(
        x=log_grad_ref,
        name='Reference',
        marker_color='gray',
        opacity=0.6,
        nbinsx=100,
        histnorm='probability' # Normalize to frequency
    ))
    
    # Method
    fig.add_trace(go.Histogram(
        x=log_grad_dist,
        name=method_name,
        marker_color='red',
        opacity=0.6,
        nbinsx=100,
        histnorm='probability'
    ))
    
    fig.update_layout(
        title="Log-Gradient Histogram",
        xaxis_title="Log10(Gradient Magnitude)",
        yaxis_title="Probability (Log Scale)",
        yaxis_type="log", # Log scale for Y-axis
        barmode='overlay',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    
    return fig

@st.cache_data
def compute_emd(ref_img, dist_img):
    """
    勾配分布間の Wasserstein Distance (EMD) を計算する
    """
    # Compute gradients
    grad_ref = compute_raw_gradient_magnitude(ref_img).flatten()
    grad_dist = compute_raw_gradient_magnitude(dist_img).flatten()
    
    # Use raw gradients (not log) for physical interpretation of edge strength differences
    from scipy.stats import wasserstein_distance
    wd = wasserstein_distance(grad_ref, grad_dist)
    return wd

@st.cache_data
def compute_gradient_gini(img):
    """
    勾配のジニ係数 (Gini Index) を計算する
    1.0に近いほどスパース
    """
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
        
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Flatten and sort
    array = np.sort(magnitude.flatten())
    # Remove zeros to avoid issues? Gini includes zeros usually.
    # But if all zero (blank image), gini is 0.
    
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    
    if np.sum(array) == 0:
        return 0.0
        
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

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

@st.cache_data
def load_image_from_bytes(file_bytes):
    """
    バイト列から画像を読み込み、正規化(0-1 float)して返す (キャッシュ対応)
    """
    import io
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image).astype(np.float32) / 255.0

def load_image(image_file):
    """
    画像ファイルを読み込む (Wrapper)
    """
    import pathlib
    if isinstance(image_file, (str, pathlib.Path)):
        with open(image_file, "rb") as f:
            return load_image_from_bytes(f.read())
    else:
        # UploadedFile
        return load_image_from_bytes(image_file.getvalue())

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

def draw_roi(img, x, y, size, color, thickness=4):
    """画像にROI枠を描画する"""
    img_with_roi = img.copy()
    # 境界チェックは簡易的に行う（はみ出しはスライスで無視されるため）
    # 上辺
    img_with_roi[y:y+thickness, x:x+size] = color
    # 下辺
    img_with_roi[y+size-thickness:y+size, x:x+size] = color
    # 左辺
    img_with_roi[y:y+size, x:x+thickness] = color
    # 右辺
    img_with_roi[y:y+size, x+size-thickness:x+size] = color
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
        line=dict(color='gray', width=1, dash='solid'),
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

@st.cache_data
def compute_delta_e(ref, dist):
    """
    CIELAB色空間における色差 (Delta E) を計算する
    ref, dist: RGB画像 (0-255, uint8) or (0.0-1.0, float)
    Returns: delta_e_map (H, W), mean_delta_e
    """
    # Ensure images are uint8 for cv2 conversion
    if ref.dtype != np.uint8:
        ref = (ref * 255).astype(np.uint8)
    if dist.dtype != np.uint8:
        dist = (dist * 255).astype(np.uint8)
        
    # Convert to LAB
    # cv2.cvtColor returns LAB in specific ranges: L [0, 255], A [0, 255], B [0, 255] for uint8
    # Ideally we want standard CIELAB ranges: L [0, 100], a [-128, 127], b [-128, 127]
    # But for simple Delta E, Euclidean distance in the same space is what matters.
    # However, standard Delta E uses L*a*b* coordinates.
    # skimage.color.rgb2lab is better if available, but let's stick to cv2 for minimal dependencies if possible.
    # Actually, cv2.cvtColor(..., cv2.COLOR_RGB2Lab) for uint8 scales:
    # L <- L * 255/100, a <- a + 128, b <- b + 128
    # So we should convert back to float LAB to get correct Delta E magnitude if we want standard units.
    
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2Lab).astype(np.float32)
    dist_lab = cv2.cvtColor(dist, cv2.COLOR_RGB2Lab).astype(np.float32)
    
    # Unscale to standard CIELAB
    # L: 0..255 -> 0..100
    ref_lab[:,:,0] = ref_lab[:,:,0] * (100.0/255.0)
    dist_lab[:,:,0] = dist_lab[:,:,0] * (100.0/255.0)
    
    # a, b: 0..255 -> -128..127
    ref_lab[:,:,1] = ref_lab[:,:,1] - 128.0
    dist_lab[:,:,1] = dist_lab[:,:,1] - 128.0
    ref_lab[:,:,2] = ref_lab[:,:,2] - 128.0
    dist_lab[:,:,2] = dist_lab[:,:,2] - 128.0
    
    # Calculate Euclidean distance
    diff = ref_lab - dist_lab
    delta_e_map = np.sqrt(np.sum(diff**2, axis=2))
    
    return delta_e_map, np.mean(delta_e_map)

def create_flicker_gif(ref, dist, duration=500):
    """
    ReferenceとMethodを交互に表示するGIFを作成する
    ref, dist: RGB numpy array
    duration: ms per frame
    Returns: bytes of GIF
    """
    if ref.dtype != np.uint8:
        ref = (ref * 255).astype(np.uint8)
    if dist.dtype != np.uint8:
        dist = (dist * 255).astype(np.uint8)
        
    img1 = Image.fromarray(ref)
    img2 = Image.fromarray(dist)
    
    import io
    bio = io.BytesIO()
    
    # Save as GIF
    img1.save(
        bio,
        format='GIF',
        save_all=True,
        append_images=[img2],
        duration=duration,
        loop=0,
        optimize=False
    )
    
    return bio.getvalue()
