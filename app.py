import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import peak_signal_noise_ratio as psnr
import tomllib
from streamlit_image_comparison import image_comparison
from PIL import Image
import io

# Import from custom modules
import config
from utils import (
    load_image,
    to_grayscale,
    compute_gradient_magnitude,
    compute_fft_spectrum,
    get_crop,
    get_crop,
    draw_roi,
    plot_line_profile,
    plot_histograms,
    compute_snr,
    compute_fsim,
    compute_gms,
    compute_niqe,
    compute_maniqa,
    compute_musiq,
    compute_clipiqa,
    HAS_PYIQA
)
from views import (
    SpatialView,
    ProfileView,
    SobelView,
    SliderView,
    GMSView,
    FFTView,
    HistogramView,
    ROIView,
    FlickerView,
    DeltaEView
)

# Load configuration
with open("app_config.toml", "rb") as f:
    app_config = tomllib.load(f)

# ページ設定
st.set_page_config(layout=app_config["page"]["layout"], page_title=app_config["page"]["page_title"])

# --- Main Application ---
def main():
    st.title(app_config["page"]["page_title"])
    
    # --- Sidebar Inputs ---
    # 1. Reference Input
    st.sidebar.subheader("1. Reference")
    ref_file = st.sidebar.file_uploader("Upload GT", type=['png', 'jpg', 'jpeg', 'tif', 'bmp'])
    
    # 2. Methods Input
    st.sidebar.subheader("2. Methods")
    method_files = st.sidebar.file_uploader("Upload Methods", type=['png', 'jpg', 'jpeg', 'tif', 'bmp'], accept_multiple_files=True)

    st.sidebar.divider()

    # --- Sidebar Settings ---
    st.sidebar.subheader("Metrics")
    metrics_toggles = {}
    for key, settings in app_config["sidebar"]["metrics"].items():
        # Skip PyIQA metrics if not available
        pyiqa_metrics = ["niqe", "maniqa", "musiq", "clipiqa"]
        if key in pyiqa_metrics and not HAS_PYIQA:
            metrics_toggles[key] = False
            continue
            
        metrics_toggles[key] = st.sidebar.checkbox(settings["label"], value=settings["default"])

    st.sidebar.subheader("Comparison Settings")
    # 機能トグル
    features = {}
    for key, settings in app_config["sidebar"]["toggles"].items():
        features[key] = st.sidebar.checkbox(settings["label"], value=settings["default"])

    # --- Strategies Initialization ---
    strategies = {
        app_config["tabs"]["spatial"]: SpatialView(),
        app_config["tabs"]["profile"]: ProfileView(),
        app_config["tabs"]["sobel"]: SobelView(),
        app_config["tabs"]["gms"]: GMSView(),
        app_config["tabs"]["fft"]: FFTView(),
        app_config["tabs"]["hist"]: HistogramView(),
        app_config["tabs"]["slider"]: SliderView(),
        app_config["tabs"]["roi"]: ROIView(),
        app_config["tabs"]["flicker"]: FlickerView(),
        app_config["tabs"]["delta_e"]: DeltaEView()
    }
    
    # --- Metric Functions ---
    metric_funcs = {
        "snr": compute_snr,
        "psnr": lambda r, d: psnr(r, d, data_range=1.0),
        "ssim": lambda r, d: ssim(r, d, data_range=1.0, channel_axis=-1),
        "fsim": compute_fsim,
        "gmsd": lambda r, d: compute_gms(r, d)[1],
        "niqe": lambda r, d: compute_niqe(d), # NIQE only needs distorted image
        "maniqa": lambda r, d: compute_maniqa(d),
        "musiq": lambda r, d: compute_musiq(d),
        "clipiqa": lambda r, d: compute_clipiqa(d)
    }

    # --- Default Images Logic ---
    import pathlib
    
    # If no user input, try to load defaults
    if not ref_file:
        default_ref_path = pathlib.Path("assets/reference.png")
        if default_ref_path.exists():
            ref_file = default_ref_path
            
    if not method_files:
        default_method_paths = [
            pathlib.Path("assets/method_blur.png"),
            pathlib.Path("assets/method_noise.png")
        ]
        # Filter existing files
        existing_defaults = [p for p in default_method_paths if p.exists()]
        if existing_defaults:
            method_files = existing_defaults

    if ref_file:
        ref_img = load_image(ref_file)
        h, w = ref_img.shape[:2]
        
        # プレビュー用の画像コピー
        ref_preview = ref_img.copy()
        
        # --- ズーム設定 ---
        crop_y, crop_x, crop_size = 0, 0, 100
        # Show ROI settings if Zoom is enabled OR ROI Check view is enabled
        if features["zoom"] or features.get("roi", False):
            st.sidebar.divider()
            st.sidebar.subheader("🔍 Zoom ROI Settings")
            crop_size = st.sidebar.slider("Box Size", 32, min(h, w)//2, 100)
            crop_x = st.sidebar.slider("X Position", 0, w - crop_size, w//2 - crop_size//2)
            crop_y = st.sidebar.slider("Y Position", 0, h - crop_size, h//2 - crop_size//2)

            # GTプレビュー上にROI枠(赤)を描画
            ref_preview[crop_y:crop_y+2, crop_x:crop_x+crop_size] = config.CV_COLOR_RED
            ref_preview[crop_y+crop_size-2:crop_y+crop_size, crop_x:crop_x+crop_size] = config.CV_COLOR_RED
            ref_preview[crop_y:crop_y+crop_size, crop_x:crop_x+2] = config.CV_COLOR_RED
            ref_preview[crop_y:crop_y+crop_size, crop_x+crop_size-2:crop_x+crop_size] = config.CV_COLOR_RED
            
        # --- 【追加】ラインプロファイル設定 ---
        profile_y = h // 2
        if features["profile"]:
            st.sidebar.subheader("📈 Line Profile Settings")
            # 行選択スライダー
            profile_y = st.sidebar.slider("Y Coordinate (Row)", 0, h-1, h//2)
            
            # GTプレビュー上に対象行の線(赤)を描画
            # 線の太さを2pxにして見やすくする
            target_row_start = max(0, profile_y - 1)
            target_row_end = min(h, profile_y + 1)
            ref_preview[target_row_start:target_row_end, :] = config.CV_COLOR_RED

        # サイドバーにプレビュー表示
        st.sidebar.image(ref_preview, caption="Reference Preview (with ROI/Line indicators)", width="stretch")


        # --- Main Comparison Area ---
        if method_files:
            st.subheader("Comparison Results")
            
            # タブ設定：有効な機能に応じて動的に追加
            tabs = [app_config["tabs"]["spatial"]]
            if features["roi"]: tabs.append(app_config["tabs"]["roi"])
            if features["flicker"]: tabs.append(app_config["tabs"]["flicker"])
            if features["delta_e"]: tabs.append(app_config["tabs"]["delta_e"])
            if features["profile"]: tabs.append(app_config["tabs"]["profile"])
            if features["sobel"]: tabs.append(app_config["tabs"]["sobel"])
            if features["gms"]: tabs.append(app_config["tabs"]["gms"])
            if features["fft"]: tabs.append(app_config["tabs"]["fft"])
            if features["hist"]: tabs.append(app_config["tabs"]["hist"])
            if features["slider"]: tabs.append(app_config["tabs"]["slider"])
            
            active_tab = st.radio("Select View Mode", tabs, horizontal=True)
            current_strategy = strategies[active_tab]
            
            # 結果格納用
            results = []
            cols = st.columns(len(method_files) + 1)
            
            # Context for strategies
            context = {
                'crop_y': crop_y, 'crop_x': crop_x, 'crop_size': crop_size,
                'profile_y': profile_y,
                'zoom_enabled': features["zoom"]
            }
            
            # --- Reference Column (Leftmost) ---
            with cols[0]:
                st.markdown("**Reference**")
                current_strategy.render_reference(st, ref_img, context)
                # Calculate NR metrics for Reference
                nr_metrics = ["niqe", "maniqa", "musiq", "clipiqa"]
                ref_captions = []
                for metric in nr_metrics:
                    if metrics_toggles.get(metric, False):
                        # Dynamic call to compute_{metric}
                        # But we have them imported as functions.
                        # Map string to function
                        func_map = {
                            "niqe": compute_niqe,
                            "maniqa": compute_maniqa,
                            "musiq": compute_musiq,
                            "clipiqa": compute_clipiqa
                        }
                        val = func_map[metric](ref_img)
                        ref_captions.append(f"{metric.upper()}: {val:.4f}")
                
                if ref_captions:
                    st.caption(" / ".join(ref_captions))

            # --- Methods Columns ---
            for idx, m_file in enumerate(method_files):
                dist_img = load_image(m_file)
                
                # サイズチェック
                if ref_img.shape != dist_img.shape:
                    cols[idx+1].error(f"Size mismatch: {m_file.name}")
                    continue
                
                # 指標計算
                metrics = {}
                metrics["Method"] = m_file.name
                
                for key, enabled in metrics_toggles.items():
                    if enabled and key in metric_funcs:
                        # Metric names in config are lowercase (snr, psnr...), but display keys are uppercase
                        metrics[key.upper()] = metric_funcs[key](ref_img, dist_img)
                
                results.append(metrics)
                
                with cols[idx+1]:
                    display_name = m_file.name
                    if len(display_name) > 20:
                        display_name = display_name[:17] + "..."
                    st.markdown(f"**{display_name}**", help=m_file.name)
                    
                    context['method_name'] = m_file.name
                    current_strategy.render_method(st, ref_img, dist_img, context)

                    caption_parts = []
                    # Order matters for display
                    caption_parts = []
                    # Order matters for display
                    display_order = ["SNR", "PSNR", "SSIM", "FSIM", "GMSD", "NIQE", "MANIQA", "MUSIQ", "CLIPIQA"]
                    for key in display_order:
                        if key in metrics:
                            val = metrics[key]
                            fmt = ".2f" if key in ["SNR", "PSNR"] else ".4f"
                            caption_parts.append(f"{key}: {val:{fmt}}")
                    
                    st.caption(" / ".join(caption_parts))

            # CSVダウンロード
            if len(results) > 0:
                st.divider()
                df_results = pd.DataFrame(results).set_index("Method")
                
                # Rename columns to include direction
                rename_map = {
                    "SNR": "SNR (↑)",
                    "PSNR": "PSNR (↑)",
                    "SSIM": "SSIM (↑)",
                    "FSIM": "FSIM (↑)",
                    "GMSD": "GMSD (↓)",
                    "NIQE": "NIQE (↓)",
                    "MANIQA": "MANIQA (↑)",
                    "MUSIQ": "MUSIQ (↑)",
                    "CLIPIQA": "CLIPIQA (↑)"
                }
                # Only rename columns that exist
                final_rename = {k: v for k, v in rename_map.items() if k in df_results.columns}
                df_results = df_results.rename(columns=final_rename)
                
                # Apply styling
                styler = df_results.style.format("{:.4f}")
                
                # Highlight Max for (↑) columns
                max_cols = [c for c in df_results.columns if "(↑)" in c]
                if max_cols:
                    styler = styler.highlight_max(subset=max_cols, axis=0, props="font-weight: bold; color: green;")
                
                # Highlight Min for (↓) columns
                min_cols = [c for c in df_results.columns if "(↓)" in c]
                if min_cols:
                    styler = styler.highlight_min(subset=min_cols, axis=0, props="font-weight: bold; color: green;")

                st.dataframe(styler, width='stretch')

if __name__ == "__main__":
    main()