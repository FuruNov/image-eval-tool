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
    compute_piqe,
    compute_gradient_kurtosis,
    compute_gradient_gini,
    compute_emd,
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
    FlickerView,
    DeltaEView,
    LogGradientView
)
from state_manager import StateManager
import os
import shutil

CACHE_DIR = ".cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def cache_uploaded_file(uploaded_file, prefix):
    """Save uploaded file to cache and return path."""
    file_path = os.path.join(CACHE_DIR, f"{prefix}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def cleanup_cache(active_paths):
    """Delete files in CACHE_DIR that are not in active_paths."""
    if not os.path.exists(CACHE_DIR):
        return
        
    # Prepare set of active absolute paths
    active_abs_paths = set()
    for p in active_paths:
        if p:
            try:
                active_abs_paths.add(os.path.abspath(p))
            except:
                pass
    
    # Iterate over cache directory
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(file_path):
            if os.path.abspath(file_path) not in active_abs_paths:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting orphan {file_path}: {e}")

# Load configuration
with open("app_config.toml", "rb") as f:
    app_config = tomllib.load(f)

# ページ設定
st.set_page_config(layout=app_config["page"]["layout"], page_title=app_config["page"]["page_title"])

# --- Main Application ---
def main():
    st.title(app_config["page"]["page_title"])
    
    # Initialize State Manager
    enable_persistence = app_config.get("local", {}).get("enable_state_persistence", False)
    state_manager = StateManager(enabled=enable_persistence)
    
    # Callback for state saving
    def on_change_handler(key):
        state_manager.save_state(key)
    
    # --- Sidebar Inputs ---
    # 1. Reference Input
    st.sidebar.subheader("1. Reference")
    
    # Load current state
    current_ref_path = state_manager.get_value("last_ref_path", None)
    
    # Uploader
    ref_file_upload = st.sidebar.file_uploader("Upload GT", type=['png', 'jpg', 'jpeg', 'tif', 'bmp'], key="ref_uploader")
    
    if ref_file_upload:
        # New upload: Replace current
        ref_path = cache_uploaded_file(ref_file_upload, "ref")
        if ref_path != current_ref_path:
            current_ref_path = ref_path
            state_manager.set_value("last_ref_path", current_ref_path)
            # Rerun to clear uploader state visually if needed, but Streamlit handles it.
            # However, to avoid re-processing on every script run if uploader is not cleared:
            # We can't easily clear uploader. We rely on the check that ref_path is what we have.
    
    # Display current reference
    ref_file = None
    if current_ref_path and os.path.exists(current_ref_path):
        ref_file = current_ref_path
        st.sidebar.success(f"Current GT: {os.path.basename(current_ref_path)}")
        if st.sidebar.button("Clear GT", key="clear_ref"):
            state_manager.set_value("last_ref_path", None)
            st.rerun()
    else:
        st.sidebar.warning("No GT image loaded.")

    # 2. Methods Input
    st.sidebar.subheader("2. Methods")
    
    # Load current state
    current_method_paths = state_manager.get_value("last_method_paths", [])
    
    # Uploader
    method_files_upload = st.sidebar.file_uploader("Add Methods", type=['png', 'jpg', 'jpeg', 'tif', 'bmp'], accept_multiple_files=True, key="method_uploader")
    
    if method_files_upload:
        # Initialize prev state if needed
        if "method_uploader_prev" not in st.session_state:
            st.session_state["method_uploader_prev"] = set()
            
        # Identify new files by name
        current_upload_names = set(f.name for f in method_files_upload)
        new_files = [f for f in method_files_upload if f.name not in st.session_state["method_uploader_prev"]]
        
        if new_files:
            for mf in new_files:
                path = cache_uploaded_file(mf, "method")
                if path not in current_method_paths:
                    current_method_paths.append(path)
            
            state_manager.set_value("last_method_paths", current_method_paths)
        
        # Update prev state
        st.session_state["method_uploader_prev"] = current_upload_names
    else:
        st.session_state["method_uploader_prev"] = set()
    
    # Manage Methods List
    method_files = []
    if current_method_paths:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Cached Methods:**")
        
        # Remove invalid paths
        valid_paths = [p for p in current_method_paths if os.path.exists(p)]
        if len(valid_paths) != len(current_method_paths):
            current_method_paths = valid_paths
            state_manager.set_value("last_method_paths", current_method_paths)
            
        # Display list with remove buttons
        # Display list with rename inputs, visibility toggles, and remove buttons
        paths_to_remove = []
        
        # Load hidden state
        hidden_methods = set(state_manager.get_value("hidden_methods", []))
        
        # Use a copy to iterate safely while modifying
        for i, p in enumerate(current_method_paths):
            col1, col2, col3 = st.sidebar.columns([0.7, 0.15, 0.15])
            
            dirname, filename = os.path.split(p)
            name, ext = os.path.splitext(filename)
            
            # Rename Input
            new_name = col1.text_input(
                "Rename", 
                value=name, 
                key=f"rename_{i}_{filename}", 
                label_visibility="collapsed"
            )
            
            # Handle Rename
            if new_name != name:
                new_filename = new_name + ext
                new_path = os.path.join(dirname, new_filename)
                
                if not os.path.exists(new_path):
                    try:
                        os.rename(p, new_path)
                        # Update list safely
                        current_method_paths[i] = new_path
                        state_manager.set_value("last_method_paths", current_method_paths)
                        
                        # Update hidden state if needed
                        if p in hidden_methods:
                            hidden_methods.remove(p)
                            hidden_methods.add(new_path)
                            state_manager.set_value("hidden_methods", list(hidden_methods))
                            
                        st.rerun()
                    except OSError as e:
                        st.sidebar.error(f"Rename failed: {e}")
                else:
                    st.sidebar.error(f"Name '{new_filename}' already exists.")
            
            # Visibility Toggle
            is_hidden = p in hidden_methods
            vis_icon = "👀" if is_hidden else "🚫"
            vis_help = "Click to Show" if is_hidden else "Click to Hide"
            if col2.button(vis_icon, key=f"vis_{i}_{filename}", help=vis_help, type="tertiary"):
                if is_hidden:
                    hidden_methods.remove(p)
                else:
                    hidden_methods.add(p)
                state_manager.set_value("hidden_methods", list(hidden_methods))
                st.rerun()

            # Remove Button
            if col3.button("🗑", key=f"rem_{i}_{filename}", help="Remove", type="tertiary"):
                paths_to_remove.append(current_method_paths[i]) # Use current path
        
        if paths_to_remove:
            for p in paths_to_remove:
                if p in current_method_paths:
                    current_method_paths.remove(p)
                # Cleanup hidden state
                if p in hidden_methods:
                    hidden_methods.remove(p)
            
            state_manager.set_value("last_method_paths", current_method_paths)
            state_manager.set_value("hidden_methods", list(hidden_methods))
            st.rerun()
            
        # Filter visible methods for main display
        method_files = [p for p in current_method_paths if p not in hidden_methods]
    else:
        st.sidebar.info("No method images loaded.")

    st.sidebar.divider()

    # --- Sidebar Settings ---
    st.sidebar.subheader("Metrics")
    metrics_toggles = {}
    for key, settings in app_config["sidebar"]["metrics"].items():
        # Skip PyIQA metrics if not available
        pyiqa_metrics = app_config.get("metrics", {}).get("pyiqa", {}).get("metrics", ["niqe", "maniqa", "musiq", "clipiqa", "piqe"])
        if key in pyiqa_metrics and not HAS_PYIQA:
            metrics_toggles[key] = False
            continue
            
        key_name = f"metric_{key}"
        default_val = state_manager.get_value(key_name, settings["default"])
        metrics_toggles[key] = st.sidebar.checkbox(
            settings["label"], 
            value=default_val,
            key=key_name,
            on_change=on_change_handler,
            args=(key_name,)
        )

    st.sidebar.subheader("Comparison Settings")
    # 機能トグル
    features = {}
    features = {}
    for key, settings in app_config["sidebar"]["toggles"].items():
        key_name = f"feature_{key}"
        default_val = state_manager.get_value(key_name, settings["default"])
        features[key] = st.sidebar.checkbox(
            settings["label"], 
            value=default_val,
            key=key_name,
            on_change=on_change_handler,
            args=(key_name,)
        )

    st.sidebar.subheader("Layout Settings")
    ref_col_ratio = st.sidebar.slider(
        "Reference Column Ratio", 
        min_value=0.5, 
        max_value=3.0, 
        value=state_manager.get_value("ref_col_ratio", 1.0),
        step=0.1,
        key="ref_col_ratio",
        on_change=on_change_handler,
        args=("ref_col_ratio",),
        help="Adjust the width of the Reference column relative to Method columns."
    )

    # --- Strategies Initialization ---
    strategies = {
        app_config["tabs"]["spatial"]: SpatialView(),
        app_config["tabs"]["profile"]: ProfileView(),
        app_config["tabs"]["sobel"]: SobelView(),
        app_config["tabs"]["gms"]: GMSView(),
        app_config["tabs"]["fft"]: FFTView(),
        app_config["tabs"]["hist"]: HistogramView(),
        app_config["tabs"]["roi"]: ROIView(),
        app_config["tabs"]["flicker"]: FlickerView(),
        app_config["tabs"]["delta_e"]: DeltaEView(),
        app_config["tabs"]["log_grad"]: LogGradientView()
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
        "clipiqa": lambda r, d: compute_clipiqa(d),
        "piqe": lambda r, d: compute_piqe(d),
        "grad_kurtosis": lambda r, d: compute_gradient_kurtosis(d),
        "grad_gini": lambda r, d: compute_gradient_gini(d),
        "emd": compute_emd
    }

    # --- Default Images Logic ---
    import pathlib
    
    # Debug Toggle
    use_defaults = st.sidebar.checkbox("Use Default Images (Debug)", value=False, key="debug_defaults")
    
    if use_defaults:
        # Force defaults
        default_ref_path = pathlib.Path(app_config.get("defaults", {}).get("reference", "assets/reference.png"))
        if default_ref_path.exists():
            ref_file = default_ref_path
            
        default_method_paths = [
            pathlib.Path(p) for p in app_config.get("defaults", {}).get("methods", ["assets/method_blur.png", "assets/method_noise.png"])
        ]
        existing_defaults = [p for p in default_method_paths if p.exists()]
        if existing_defaults:
            method_files = [str(p) for p in existing_defaults]
    else:
        # Normal logic: Use uploaded/cached if available, else fallback to defaults if empty
        if not ref_file:
            default_ref_path = pathlib.Path(app_config.get("defaults", {}).get("reference", "assets/reference.png"))
            if default_ref_path.exists():
                ref_file = default_ref_path
                
        if not method_files:
            default_method_paths = [
                pathlib.Path(p) for p in app_config.get("defaults", {}).get("methods", ["assets/method_blur.png", "assets/method_noise.png"])
            ]
            existing_defaults = [p for p in default_method_paths if p.exists()]
            if existing_defaults:
                method_files = [str(p) for p in existing_defaults]

    # Cleanup unused cache files
    active_cache_files = []
    if ref_file: active_cache_files.append(ref_file)
    
    # Protect current user uploads even if using defaults
    if 'current_ref_path' in locals() and current_ref_path:
        active_cache_files.append(current_ref_path)
    
    # Use current_method_paths (all loaded files) for cleanup protection, 
    # not method_files (which only contains visible ones)
    if 'current_method_paths' in locals() and current_method_paths:
         active_cache_files.extend(current_method_paths)
    elif method_files:
         active_cache_files.extend(method_files)
         
    cleanup_cache(active_cache_files)

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
            
            # ROI Sliders with persistence
            crop_size = st.sidebar.slider(
                "Box Size", 32, min(h, w)//2, 
                state_manager.get_value("crop_size", 100),
                key="crop_size", on_change=on_change_handler, args=("crop_size",)
            )
            crop_x = st.sidebar.slider(
                "X Position", 0, w - crop_size, 
                state_manager.get_value("crop_x", w//2 - crop_size//2),
                key="crop_x", on_change=on_change_handler, args=("crop_x",)
            )
            crop_y = st.sidebar.slider(
                "Y Position", 0, h - crop_size, 
                state_manager.get_value("crop_y", h//2 - crop_size//2),
                key="crop_y", on_change=on_change_handler, args=("crop_y",)
            )

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
            profile_y = st.sidebar.slider(
                "Y Coordinate (Row)", 0, h-1, 
                state_manager.get_value("profile_y", h//2),
                key="profile_y", on_change=on_change_handler, args=("profile_y",)
            )
            
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
            if features["log_grad"]: tabs.append(app_config["tabs"]["log_grad"])
            if features["slider"]: tabs.append(app_config["tabs"]["slider"])
            
            # Restore previous selection if possible
            saved_view = state_manager.get_value("view_mode", tabs[0])
            default_index = 0
            if saved_view in tabs:
                default_index = tabs.index(saved_view)
            
            active_tab = st.radio(
                "Select View Mode", 
                tabs, 
                index=default_index,
                horizontal=True,
                key="view_mode",
                on_change=on_change_handler,
                args=("view_mode",)
            )
            current_strategy = strategies[active_tab]
            
            # 結果格納用
            results = []
            
            # Log-Grad view does not need reference column
            show_ref_col = active_tab != app_config["tabs"]["log_grad"]
            
            if show_ref_col:
                # [ref_ratio, 1, 1, ...]
                col_spec = [ref_col_ratio] + [1.0] * len(method_files)
            else:
                # [1, 1, ...]
                col_spec = [1.0] * len(method_files)
                
            cols = st.columns(col_spec)
            
            # Context for strategies
            context = {
                'crop_y': crop_y, 'crop_x': crop_x, 'crop_size': crop_size,
                'profile_y': profile_y,
                'zoom_enabled': features["zoom"],
                'log_scale_hist': features.get("log_scale_hist", True)
            }
            
            # --- Reference Metrics Calculation (Always run) ---
            ref_metrics_dict = {"Method": "Reference"}
            nr_metrics = app_config.get("metrics", {}).get("display", {}).get("nr_metrics", ["niqe", "maniqa", "musiq", "clipiqa", "piqe", "grad_kurtosis", "grad_gini"])
            
            func_map = {
                "niqe": compute_niqe,
                "maniqa": compute_maniqa,
                "musiq": compute_musiq,
                "clipiqa": compute_clipiqa,
                "piqe": compute_piqe,
                "grad_kurtosis": compute_gradient_kurtosis,
                "grad_gini": compute_gradient_gini
            }
            
            ref_captions = []
            for metric in nr_metrics:
                if metrics_toggles.get(metric, False):
                    val = func_map[metric](ref_img)
                    ref_metrics_dict[metric.upper()] = val
                    ref_captions.append(f"{metric.upper()}: {val:.4f}")
            
            # Add Reference metrics to results first
            results.append(ref_metrics_dict)

            # --- Reference Column (Leftmost) ---
            if show_ref_col:
                with cols[0]:
                    st.markdown("**Reference**")
                    current_strategy.render_reference(st, ref_img, context)
                    # if ref_captions:
                    #     st.caption(" / ".join(ref_captions))

            # --- Methods Columns ---
            for idx, m_file_path in enumerate(method_files):
                # m_file_path is now a string path
                dist_img = load_image(m_file_path)
                m_name = os.path.basename(m_file_path)
                
                col_idx = idx + (1 if show_ref_col else 0)
                
                # サイズチェック
                if ref_img.shape != dist_img.shape:
                    cols[col_idx].error(f"Size mismatch: {m_name}")
                    continue
                
                # 指標計算
                metrics = {}
                metrics["Method"] = m_name
                
                for key, enabled in metrics_toggles.items():
                    if enabled and key in metric_funcs:
                        # Metric names in config are lowercase (snr, psnr...), but display keys are uppercase
                        metrics[key.upper()] = metric_funcs[key](ref_img, dist_img)
                
                results.append(metrics)
                
                with cols[col_idx]:
                    display_name = m_name
                    if len(display_name) > 20:
                        display_name = display_name[:17] + "..."
                    st.markdown(f"**{display_name}**", help=m_name)
                    
                    context['method_name'] = m_name
                    current_strategy.render_method(st, ref_img, dist_img, context)

                    # Captions removed as per user request (table is sufficient)
                    # caption_parts = []
                    # display_order = app_config.get("metrics", {}).get("display", {}).get("order", [])
                    # for key in display_order:
                    #     if key in metrics:
                    #         val = metrics[key]
                    #         fmt = ".2f" if key in ["SNR", "PSNR"] else ".4f"
                    #         caption_parts.append(f"{key}: {val:{fmt}}")
                    
                    # st.caption(" / ".join(caption_parts))

            # CSVダウンロード
            if len(results) > 0:
                st.divider()
                df_results = pd.DataFrame(results).set_index("Method")
                
                # Reorder columns based on config
                display_order = app_config.get("metrics", {}).get("display", {}).get("order", [])
                # Filter order to only include columns that exist in df_results
                final_order = [col for col in display_order if col in df_results.columns]
                # Add any remaining columns that are not in display_order
                remaining_cols = [col for col in df_results.columns if col not in final_order]
                df_results = df_results[final_order + remaining_cols]
                
                # Rename columns to include direction
                rename_map = app_config.get("metrics", {}).get("rename", {})
                # Only rename columns that exist
                final_rename = {k: v for k, v in rename_map.items() if k in df_results.columns}
                df_results = df_results.rename(columns=final_rename)
                
                # Apply styling
                styler = df_results.style.format("{:.4f}")
                
                def highlight_rank(s, is_higher_better=True):
                    # Exclude Reference from ranking
                    s_ranked = s.drop("Reference", errors='ignore')
                    
                    # Sort to find ranks
                    # method='min' assigns the same rank to ties (e.g. 1, 1, 3)
                    # ascending=False means Higher is Rank 1
                    ranks = s_ranked.rank(method='min', ascending=not is_higher_better)
                    
                    # Reindex to match original s (Reference will be NaN)
                    ranks = ranks.reindex(s.index)
                    
                    colors = app_config.get("metrics", {}).get("colors", {})
                    gold_style = f"font-weight: bold; color: {colors.get('gold_text', '#856404')}; background-color: {colors.get('gold_bg', '#fff3cd')};"
                    silver_style = f"color: {colors.get('silver_text', '#495057')}; background-color: {colors.get('silver_bg', '#e2e3e5')};"
                    bronze_style = f"color: {colors.get('bronze_text', '#854018')}; background-color: {colors.get('bronze_bg', '#ffe5d0')};"

                    styles = []
                    for r in ranks:
                        if pd.isna(r):
                            styles.append("") # No style for Reference
                        elif r == 1:
                            styles.append(gold_style)
                        elif r == 2:
                            styles.append(silver_style)
                        elif r == 3:
                            styles.append(bronze_style)
                        else:
                            styles.append("")
                    return styles

                # Highlight (↑) columns (Higher is better)
                max_cols = [c for c in df_results.columns if "(↑)" in c]
                if max_cols:
                    styler = styler.apply(highlight_rank, subset=max_cols, axis=0, is_higher_better=True)
                
                # Highlight (↓) columns (Lower is better)
                min_cols = [c for c in df_results.columns if "(↓)" in c]
                if min_cols:
                    styler = styler.apply(highlight_rank, subset=min_cols, axis=0, is_higher_better=False)

                st.dataframe(styler, width='stretch')
                
                # Legend
                colors = app_config.get("metrics", {}).get("colors", {})
                st.markdown(f"""
                <div style="display: flex; gap: 10px; font-size: 0.8em; color: #555;">
                    <span><span style="background-color: {colors.get('gold_bg', '#fff3cd')}; color: {colors.get('gold_text', '#856404')}; padding: 2px 6px; border-radius: 4px; font-weight: bold;">1st</span> Gold</span>
                    <span><span style="background-color: {colors.get('silver_bg', '#e2e3e5')}; color: {colors.get('silver_text', '#495057')}; padding: 2px 6px; border-radius: 4px;">2nd</span> Silver</span>
                    <span><span style="background-color: {colors.get('bronze_bg', '#ffe5d0')}; color: {colors.get('bronze_text', '#854018')}; padding: 2px 6px; border-radius: 4px;">3rd</span> Bronze</span>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()