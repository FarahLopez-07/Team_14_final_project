import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd

st.set_page_config(page_title="Live/Dead Cell Counter", layout="wide")

st.title("Fluorescence Cell Counter (DAPI / Live / Dead)")
st.write(
    """
Upload **one fluorescence image** for each stain:

- **Blue** (DAPI, nuclei)
- **Green** (live cells)
- **Red** (dead cells)

Use the sliders to adjust intensity thresholds and minimum cell size.
The app will count objects and estimate live/dead percentages.
"""
)

def load_image(uploaded_file):
    """Load an image from a Streamlit UploadedFile and return as RGB numpy array."""
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    return np.array(img)

def count_cells_from_gray(gray_img, threshold, min_area):
    """
    Threshold a grayscale image and count connected components
    above a given minimum area.
    """
    # Ensure uint8
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

    # Binary threshold
    _, bin_mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )

    # Skip background (label 0), filter by min_area
    areas = stats[1:, cv2.CC_STAT_AREA]
    count = int(np.sum(areas >= min_area))

    return count, bin_mask

def process_single_channel(file, threshold, min_area, channel_name):
    """
    Process a single UploadedFile for a given channel.
    Returns:
      count: int
      row: dict (for table)
      preview: (filename, original_rgb, binary_mask)
    """
    if file is None:
        return 0, None, None

    rgb = load_image(file)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    count, bin_mask = count_cells_from_gray(gray, threshold, min_area)

    row = {
        "channel": channel_name,
        "filename": file.name,
        "count": count,
    }
    preview = (file.name, rgb, bin_mask)

    return count, row, preview

# --- UI: file uploaders ---
st.header("1. Upload one image per stain")

col1, col2, col3 = st.columns(3)

with col1:
    blue_file = st.file_uploader(
        "Blue / DAPI nuclei image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Upload ONE image with DAPI-stained nuclei (blue).",
    )

with col2:
    green_file = st.file_uploader(
        "Green (live cells) image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Upload ONE image with live cells (green).",
    )

with col3:
    red_file = st.file_uploader(
        "Red (dead cells) image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Upload ONE image with dead cells (red).",
    )

# --- Sidebar: thresholds and size filter ---
st.sidebar.header("Segmentation parameters")

blue_thresh = st.sidebar.slider("Blue (DAPI) threshold", 0, 255, 80)
green_thresh = st.sidebar.slider("Green (live) threshold", 0, 255, 80)
red_thresh = st.sidebar.slider("Red (dead) threshold", 0, 255, 80)

min_area = st.sidebar.slider(
    "Minimum cell area (pixels)", min_value=10, max_value=2000, value=50, step=10
)
st.sidebar.write(
    "Increase this if the program overcounts tiny specks; decrease if it misses small cells."
)

run_button = st.button("2. Run analysis")

if run_button:
    if blue_file is None and green_file is None and red_file is None:
        st.warning("Please upload at least one image for any channel.")
    else:
        # Process each channel
        blue_total, blue_row, blue_preview = process_single_channel(
            blue_file, blue_thresh, min_area, "DAPI (nuclei)"
        )
        green_total, green_row, green_preview = process_single_channel(
            green_file, green_thresh, min_area, "Live (green)"
        )
        red_total, red_row, red_preview = process_single_channel(
            red_file, red_thresh, min_area, "Dead (red)"
        )

        rows = []
        for r in [blue_row, green_row, red_row]:
            if r is not None:
                rows.append(r)

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("Per-image counts")
            st.dataframe(df)

        # Total nuclei = blue if available; otherwise fall back to (green + red)
        total_nuclei = blue_total if blue_total > 0 else (green_total + red_total)

        st.subheader("Total counts")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("DAPI (nuclei)", blue_total)
        col_b.metric("Live (green)", green_total)
        col_c.metric("Dead (red)", red_total)
        col_d.metric("Total nuclei used", total_nuclei)

        # Live/Dead percentages
        if total_nuclei > 0:
            live_pct = 100.0 * green_total / total_nuclei
            dead_pct = 100.0 * red_total / total_nuclei

            st.subheader("Live/Dead percentages")

            st.markdown(
                f"""
**Definitions**

- Total nuclei: \( N_{{total}} = {total_nuclei} \)  
- Live cells: \( N_{{live}} = {green_total} \)  
- Dead cells: \( N_{{dead}} = {red_total} \)  

Percentages:
- \( \\text{{Live \%}} = 100 \\times N_{{live}} / N_{{total}} = {live_pct:.2f}\\% \)  
- \( \\text{{Dead \%}} = 100 \\times N_{{dead}} / N_{{total}} = {dead_pct:.2f}\\% \)
"""
            )

            col_l, col_d2 = st.columns(2)
            col_l.metric("Live %", f"{live_pct:.2f}%")
            col_d2.metric("Dead %", f"{dead_pct:.2f}%")
        else:
            st.warning(
                "Total nuclei count is 0. Adjust thresholds or check that the DAPI image is correctly exposed."
            )

        # Preview segmentation
        st.subheader("Segmentation preview (original vs thresholded)")

        def show_preview(title, preview):
            if preview is None:
                return
            fname, rgb, mask = preview
            st.markdown(f"**{title} — {fname}**")
            c1, c2 = st.columns(2)
            with c1:
                st.image(rgb, caption="Original image", use_column_width=True)
            with c2:
                st.image(mask, caption="Binary mask", use_column_width=True)

        show_preview("DAPI (nuclei)", blue_preview)
        show_preview("Live (green)", green_preview)
        show_preview("Dead (red)", red_preview)
else:
    st.info("Upload one image for each stain and click **Run analysis** to start.")
