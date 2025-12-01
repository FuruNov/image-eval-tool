import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison
from PIL import Image


import config
from utils import (
    to_grayscale,
    compute_gradient_magnitude,
    compute_fft_spectrum,
    get_crop,
    draw_roi,
    plot_line_profile,
    plot_histograms,
    compute_gms,
    compute_delta_e,
    create_flicker_gif
)

# --- Strategy Pattern for Views ---
class ViewStrategy:
    def render_reference(self, container, ref_img, context):
        raise NotImplementedError
    
    def render_method(self, container, ref_img, dist_img, context):
        raise NotImplementedError

class SpatialView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        if context['zoom_enabled']:
            ref_img = get_crop(ref_img, context['crop_y'], context['crop_x'], context['crop_size'])
        container.image(ref_img, width="stretch", caption="Ground Truth")
        # Reference Error Map (Black)
        error_map = np.zeros_like(ref_img)
        container.image(error_map, width="stretch", caption="Error Map (Ref)")

    def render_method(self, container, ref_img, dist_img, context):
        if context['zoom_enabled']:
            ref_img = get_crop(ref_img, context['crop_y'], context['crop_x'], context['crop_size'])
            dist_img = get_crop(dist_img, context['crop_y'], context['crop_x'], context['crop_size'])
            
        diff = np.abs(ref_img - dist_img)
        diff_gray = to_grayscale(diff)
        cmap_name = 'viridis'
        heatmap = plt.get_cmap(cmap_name)(np.clip(diff_gray * 5.0, 0, 1))
        container.image(dist_img, width="stretch", caption="Restored")
        container.image(heatmap, width="stretch", caption=f"Error Map ({cmap_name})")
        
class ProfileView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        h = ref_img.shape[0]
        
        # Handle Zoom
        if context['zoom_enabled']:
            crop_y, crop_x, crop_size = context['crop_y'], context['crop_x'], context['crop_size']
            ref_img = get_crop(ref_img, crop_y, crop_x, crop_size)
            # Adjust profile_y relative to crop
            rel_y = context['profile_y'] - crop_y
            if 0 <= rel_y < crop_size:
                ref_with_line = ref_img.copy()
                ref_with_line[max(0, rel_y-1):min(crop_size, rel_y+1), :] = config.CV_COLOR_RED
                container.image(ref_with_line, width="stretch", caption=f"Zoomed GT with Line (Y={context['profile_y']})")
            else:
                container.image(ref_img, width="stretch", caption="Zoomed GT (Line outside)")
        else:
            ref_with_line = ref_img.copy()
            ref_with_line[max(0, context['profile_y']-1):min(h, context['profile_y']+1), :] = config.CV_COLOR_RED
            container.image(ref_with_line, width="stretch", caption=f"GT with Line (Y={context['profile_y']})")

    def render_method(self, container, ref_img, dist_img, context):
        h = dist_img.shape[0]
        
        # Handle Zoom
        if context['zoom_enabled']:
            crop_y, crop_x, crop_size = context['crop_y'], context['crop_x'], context['crop_size']
            ref_img_crop = get_crop(ref_img, crop_y, crop_x, crop_size)
            dist_img_crop = get_crop(dist_img, crop_y, crop_x, crop_size)
            
            rel_y = context['profile_y'] - crop_y
            if 0 <= rel_y < crop_size:
                dist_with_line = dist_img_crop.copy()
                dist_with_line[max(0, rel_y-1):min(crop_size, rel_y+1), :] = config.CV_COLOR_RED
                container.image(dist_with_line, width="stretch", caption=f"Zoomed Image with Line (Y={context['profile_y']})")
                
                # Plot profile for the cropped region
                fig = plot_line_profile(ref_img_crop, dist_img_crop, rel_y, context['method_name'])
                container.plotly_chart(fig, width='stretch', key=f"profile_zoom_{context.get('method_name', 'method')}")
            else:
                container.image(dist_img_crop, width="stretch", caption="Zoomed Image (Line outside)")
                container.warning("Line Profile is outside the Zoomed Region.")
        else:
            dist_with_line = dist_img.copy()
            dist_with_line[max(0, context['profile_y']-1):min(h, context['profile_y']+1), :] = config.CV_COLOR_RED
            container.image(dist_with_line, width="stretch", caption=f"Image with Profile Line (Y={context['profile_y']})")
            fig = plot_line_profile(ref_img, dist_img, context['profile_y'], context['method_name'])
            container.plotly_chart(fig, width='stretch', key=f"profile_{context.get('method_name', 'method')}")

class SobelView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        ref_sobel = compute_gradient_magnitude(ref_img)
        if context['zoom_enabled']:
            ref_sobel = get_crop(ref_sobel, context['crop_y'], context['crop_x'], context['crop_size'])
        container.image(ref_sobel, width="stretch", caption="GT Sobel Edge", clamp=True)

    def render_method(self, container, ref_img, dist_img, context):
        ref_sobel = compute_gradient_magnitude(ref_img)
        dist_sobel = compute_gradient_magnitude(dist_img)
        
        if context['zoom_enabled']:
            ref_sobel = get_crop(ref_sobel, context['crop_y'], context['crop_x'], context['crop_size'])
            dist_sobel = get_crop(dist_sobel, context['crop_y'], context['crop_x'], context['crop_size'])
            
        container.image(dist_sobel, width="stretch", caption="Sobel Edge Map", clamp=True)
        diff_sobel = np.abs(ref_sobel - dist_sobel)
        container.image(np.clip(diff_sobel * 2.0, 0, 1), width="stretch", caption="Edge Error (x2.0)", clamp=True)

class SliderView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        container.info("Slider View is active. Compare Reference vs Method in the Method column.")

    def render_method(self, container, ref_img, dist_img, context):
        if context['zoom_enabled']:
            ref_img = get_crop(ref_img, context['crop_y'], context['crop_x'], context['crop_size'])
            dist_img = get_crop(dist_img, context['crop_y'], context['crop_x'], context['crop_size'])

        # image_comparison expects images as numpy arrays (0-255 uint8) or PIL images
        # Our images are float32 0-1. Convert to uint8.
        ref_uint8 = (ref_img * 255).astype(np.uint8)
        dist_uint8 = (dist_img * 255).astype(np.uint8)
        
        image_comparison(
            img1=ref_uint8,
            img2=dist_uint8,
            label1="Reference",
            label2=context.get('method_name', 'Method'),
            width=700, # Adjust as needed
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
        container.markdown('<div style="margin-top: -20px;"></div>', unsafe_allow_html=True)

class GMSView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        ref_gms_map, _ = compute_gms(ref_img, ref_img)
        if context['zoom_enabled']:
            ref_gms_map = get_crop(ref_gms_map, context['crop_y'], context['crop_x'], context['crop_size'])
        container.image(ref_gms_map, width="stretch", caption="GT GMS Map (Self=1.0)", clamp=True)

    def render_method(self, container, ref_img, dist_img, context):
        gms_map, _ = compute_gms(ref_img, dist_img)
        
        if context['zoom_enabled']:
            gms_map = get_crop(gms_map, context['crop_y'], context['crop_x'], context['crop_size'])
            
        gms_vis = plt.get_cmap('viridis')(gms_map)[:,:,:3]
        container.image(gms_vis, width="stretch", caption="GMS Map (Yellow=High Sim)")
        
        # Download Button for GMS Map
        gms_vis_uint8 = (gms_vis * 255).astype(np.uint8)
        gms_vis_pil = Image.fromarray(gms_vis_uint8)
        buf = io.BytesIO()
        gms_vis_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        container.download_button(
            label="Download GMS Map",
            data=byte_im,
            file_name=f"gms_map_{context.get('method_name', 'method')}.png",
            mime="image/png",
            key=f"dl_gms_{context.get('method_name', 'method')}"
        )

class FFTView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        if context['zoom_enabled']:
            ref_img = get_crop(ref_img, context['crop_y'], context['crop_x'], context['crop_size'])
        ref_fft_img = compute_fft_spectrum(ref_img)
        ref_fft_vis = plt.get_cmap('inferno')(ref_fft_img)[:,:,:3]
        container.image(ref_fft_vis, width="stretch", caption="GT Spectrum")

    def render_method(self, container, ref_img, dist_img, context):
        if context['zoom_enabled']:
            dist_img = get_crop(dist_img, context['crop_y'], context['crop_x'], context['crop_size'])
        fft_img = compute_fft_spectrum(dist_img)
        fft_vis = plt.get_cmap('inferno')(fft_img)[:,:,:3]
        container.image(fft_vis, width="stretch", caption="Log-Magnitude Spectrum")

class HistogramView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        container.info("Histogram Analysis. Compare distributions in the Method column.")
        
    def render_method(self, container, ref_img, dist_img, context):
        if context['zoom_enabled']:
            ref_img = get_crop(ref_img, context['crop_y'], context['crop_x'], context['crop_size'])
            dist_img = get_crop(dist_img, context['crop_y'], context['crop_x'], context['crop_size'])
            
        fig, stats = plot_histograms(ref_img, dist_img, context['method_name'])
        container.plotly_chart(fig, width='stretch', key=f"hist_{context.get('method_name', 'method')}")
        
        # Display stats
        with container.expander("Residual Statistics", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Mean Bias", f"{stats['Mean Bias']:.4f}")
            cols[1].metric("Std Dev", f"{stats['Std Dev']:.4f}")
            cols[2].metric("Kurtosis", f"{stats['Kurtosis']:.2f}")

class ROIView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        # 1. Full Image with ROI Box
        # Use draw_roi to draw the box on the full image
        ref_with_box = draw_roi(ref_img, context['crop_x'], context['crop_y'], context['crop_size'], config.CV_COLOR_RED)
        container.image(ref_with_box, width="stretch", caption="Full Image (with ROI Box)")
        
        # 2. Zoomed Crop
        ref_crop = get_crop(ref_img, context['crop_y'], context['crop_x'], context['crop_size'])
        # Resize crop for better visibility if it's too small? 
        # Streamlit handles display size, but pixelated look is desired for zoom.
        # 'clamp=True' handles values outside 0-1, but we are 0-1.
        container.image(ref_crop, width="stretch", caption="Zoomed ROI")

    def render_method(self, container, ref_img, dist_img, context):
        # 1. Full Image with ROI Box
        dist_with_box = draw_roi(dist_img, context['crop_x'], context['crop_y'], context['crop_size'], config.CV_COLOR_RED)
        container.image(dist_with_box, width="stretch", caption="Full Image (with ROI Box)")
        
        # 2. Zoomed Crop
        dist_crop = get_crop(dist_img, context['crop_y'], context['crop_x'], context['crop_size'])
        container.image(dist_crop, width="stretch", caption="Zoomed ROI")

class FlickerView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        # Show static reference
        if context['zoom_enabled']:
            crop_y, crop_x, crop_size = context['crop_y'], context['crop_x'], context['crop_size']
            ref_img = get_crop(ref_img, crop_y, crop_x, crop_size)
        container.image(ref_img, caption="Reference", use_container_width=True)

    def render_method(self, container, ref_img, dist_img, context):
        # Show Flicker GIF
        if context['zoom_enabled']:
            crop_y, crop_x, crop_size = context['crop_y'], context['crop_x'], context['crop_size']
            ref_img = get_crop(ref_img, crop_y, crop_x, crop_size)
            dist_img = get_crop(dist_img, crop_y, crop_x, crop_size)
            
        gif_bytes = create_flicker_gif(ref_img, dist_img, duration=500)
        container.image(gif_bytes, caption="Flicker (Ref ↔ Method)", use_container_width=True)

class DeltaEView(ViewStrategy):
    def render_reference(self, container, ref_img, context):
        if context['zoom_enabled']:
            crop_y, crop_x, crop_size = context['crop_y'], context['crop_x'], context['crop_size']
            ref_img = get_crop(ref_img, crop_y, crop_x, crop_size)
        container.image(ref_img, caption="Reference", use_container_width=True)

    def render_method(self, container, ref_img, dist_img, context):
        if context['zoom_enabled']:
            crop_y, crop_x, crop_size = context['crop_y'], context['crop_x'], context['crop_size']
            ref_img = get_crop(ref_img, crop_y, crop_x, crop_size)
            dist_img = get_crop(dist_img, crop_y, crop_x, crop_size)
            
        delta_e_map, mean_delta_e = compute_delta_e(ref_img, dist_img)
        max_delta_e = np.max(delta_e_map)
        
        # Normalize for visualization (0-255)
        # Use a colormap (e.g., inferno or jet)
        # Normalize to 0-1 range for colormap
        if max_delta_e > 0:
            norm_map = delta_e_map / max_delta_e
        else:
            norm_map = delta_e_map
            
        # Apply colormap (returns RGBA, take RGB)
        # Using 'jet' or 'inferno' to highlight differences
        heatmap = plt.get_cmap('jet')(norm_map)[:, :, :3]
        
        # Display as image
        container.image(heatmap, caption=f"ΔE Map (Mean: {mean_delta_e:.2f}, Max: {max_delta_e:.2f})", use_container_width=True)
