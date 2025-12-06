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

- **Blue** (DAPI nuclei)
- **Green** (live cells)
- **Red** (dead cells)

Features:
- Automatic scale-bar + number removal  
- Automatic threshold recommendation  
- Automatic minimum-area recommendation  
- Per-stain threshold + pixel size controls  
- Live/dead percentage  
- Segmentation preview  
- Sidebar history  
"""
)

# ------------------ SESSION HISTORY ------------------
if "history" not in st.session_state:
    st.session_state["history"] = []


# ------------------ UTILITIES ------------------
def load_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    return np.array(img)


def remove_scale_bar(rgb):
    """
    Detect and remove:
    1. The scale bar rectangle.
    2. Bright numbers located above OR below the scale bar.
    Numbers usually appear as small bright connected components.
    """
    h, w, _ = rgb.shape

    # ---- Step 1: detect bar in bottom 25% ----
    bottom_start = int(h * 0.75)
    bottom_region = rgb[bottom_start:, :]
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return rgb, None  # no bar found

    # Scale bar = largest bright component
    bar_cnt = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(bar_cnt)

    clean = rgb.copy()

    # ---- Step 2: remove bar ----
    abs_by = bottom_start + by
    clean[abs_by : abs_by + bh, bx : bx + bw] = 0

    # ---- Step 3: remove text ABOVE the bar (up to 50 px) ----
    top_region_y1 = max(0, abs_by - 50)
    text_region_above = rgb[top_region_y1:abs_by, bx:bx + bw]
    gray_above = cv2.cvtColor(text_region_above, cv2.COLOR_RGB2GRAY)
    _, tmask_above = cv2.threshold(gray_above, 180, 255, cv2.THRESH_BINARY)
    tcnts_above, _ = cv2.findContours(tmask_above, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in tcnts_above:
        tx, ty, tw, th = cv2.boundingRect(c)
        if th < 40:  # text is small
            clean[top_region_y1 + ty : top_region_y1 + ty + th,
                  bx + tx : bx + tx + tw] = 0

    # ---- Step 4: remove text BELOW the bar (up to 50 px) ----
    bot_region_y1 = abs_by + bh
    bot_region_y2 = min(h, abs_by + bh + 50)
    text_region_below = rgb[bot_region_y1:bot_region_y2, bx:bx + bw]
    gray_below = cv2.cvtColor(text_region_below, cv2.COLOR_RGB2GRAY)
    _, tmask_below = cv2.threshold(gray_below, 180, 255, cv2.THRESH_BINARY)
    tcnts_below, _ = cv2.findContours(tmask_below, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in tcnts_below:
        tx, ty, tw, th = cv2.boundingRect(c)
        if th < 40:
            clean[bot_region_y1 + ty : bot_region_y1 + ty + th,
                  bx + tx : bx + tx + tw] = 0

    return clean, (bx, abs_by, bw, bh)


def count_cells_from_gray(gray_img, threshold, min_area):
    """Threshold grayscale image and count connected components > min_area."""
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, bin_mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )

    areas = stats[1:, cv2.CC_STAT_AREA]
    count = int(np.sum(areas >= min_area))

    return count, bin_mask


def recommend_threshold(gray_img):
    """Use Otsu thresholding for recommending threshold."""
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    otsu_val, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if otsu_val < 60:
        condition = "Very clean background"
    elif otsu_val < 100:
        condition = "Clean background"
    elif otsu_val < 140:
        condition = "Slightly noisy background"
    else:
        condition = "Bright / uneven background"

    return int(otsu_val), condition


def recommend_min_area(gray_img, threshold):
    """Estimate typical cell size from connected components."""
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return 20, "No clear cells detected — try lowering threshold"

    median_area = np.median(areas)
    recommended = int(max(10, 0.4 * median_area))

    if median_area < 60:
        cond = "Cells appear small"
    elif median_area < 150:
        cond = "Cells appear medium-sized"
    else:
        cond = "Cells appear large"

    return recommended, cond


def process_single_channel(file, threshold, min_area, channel_name):
    if file is None:
        return 0, None, None

    rgb = load_image(file)
    clean_rgb, scale_bar_box = remove_scale_bar(rgb)
    gray = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY)

    count, bin_mask = count_cells_from_gray(gray, threshold, min_area)

    row = {"channel": channel_name, "filename": file.name, "count": count}
    preview = (file.name, rgb, clean_rgb, bin_mask, scale_bar_box)

    return count, row, preview


# ------------------ UPLOAD UI ------------------
st.header("1. Upload images")
col1, col2, col3 = st.columns(3)

def uploader_block(file, label):
    if file:
        img = load_image(file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        rec_t, cond_t = recommend_threshold(gray)
        st.caption(f"{cond_t} — recommended threshold ≈ {rec_t}")

        rec_area, cond_area = recommend_min_area(gray, rec_t)
        st.caption(f"{cond_area} — recommended min area ≈ {rec_area} px")


with col1:
    blue_file = st.file_uploader("Blue / DAPI", type=["png", "jpg", "jpeg", "tif", "tiff"])
    uploader_block(blue_file, "Blue")

with col2:
    green_file = st.file_uploader("Green (Live)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    uploader_block(green_file, "Green")

with col3:
    red_file = st.file_uploader("Red (Dead)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    uploader_block(red_file, "Red")


# ------------------ SIDEBAR ------------------
st.sidebar.header("Segmentation parameters")

st.sidebar.subheader("Thresholds")
blue_thresh = st.sidebar.slider("DAPI threshold", 0, 255, 80)
green_thresh = st.sidebar.slider("Live threshold", 0, 255, 80)
red_thresh = st.sidebar.slider("Dead threshold", 0, 255, 80)

st.sidebar.subheader("Minimum object size (px)")
blue_min_area = st.sidebar.slider("DAPI min area", 10, 2000, 30)
green_min_area = st.sidebar.slider("Live min area", 10, 5000, 100)
red_min_area = st.sidebar.slider("Dead min area", 10, 5000, 100)

run_button = st.button("2. Run analysis")


# ------------------ RUN ANALYSIS ------------------
if run_button:
    if not (blue_file and green_file and red_file):
        st.warning("Upload all 3 images to run analysis.")
    else:
        blue_total, blue_row, blue_preview = process_single_channel(
            blue_file, blue_thresh, blue_min_area, "DAPI nuclei"
        )
        green_total, green_row, green_preview = process_single_channel(
            green_file, green_thresh, green_min_area, "Live cells"
        )
        red_total, red_row, red_preview = process_single_channel(
            red_file, red_thresh, red_min_area, "Dead cells"
        )

        rows = [r for r in [blue_row, green_row, red_row] if r]
        df = pd.DataFrame(rows)

        st.subheader("Per-image counts")
        st.dataframe(df)

        total_nuclei = blue_total if blue_total > 0 else green_total + red_total

        live_pct = 100 * green_total / total_nuclei if total_nuclei else 0
        dead_pct = 100 * red_total / total_nuclei if total_nuclei else 0

        st.subheader("Total counts")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Nuclei (DAPI)", blue_total)
        colB.metric("Live cells", green_total)
        colC.metric("Dead cells", red_total)
        colD.metric("Total nuclei", total_nuclei)

        st.subheader("Live/Dead percentages")
        col1, col2 = st.columns(2)
        col1.metric("Live %", f"{live_pct:.2f}%")
        col2.metric("Dead %", f"{dead_pct:.2f}%")

        # Save history
        st.session_state["history"].insert(
            0,
            {
                "nuclei": total_nuclei,
                "live": green_total,
                "dead": red_total,
                "live_pct": live_pct,
                "dead_pct": dead_pct,
            },
        )

        # ------------------ PREVIEW ------------------
        st.subheader("Segmentation preview")

        def show_preview(title, preview):
            fname, orig, clean, mask, scale_bar = preview

            st.markdown(f"### {title} — {fname}")

            col1, col2, col3 = st.columns(3)
            col1.image(orig, caption="Original", use_column_width=True)
            col2.image(clean, caption="Scale-bar + number removed", use_column_width=True)
            col3.image(mask, caption="Binary mask", use_column_width=True)

            if scale_bar:
                x, y, w, h = scale_bar
                st.caption(f"Scale bar removed at x={x}, y={y}, size={w}×{h}")

        show_preview("DAPI nuclei", blue_preview)
        show_preview("Live cells", green_preview)
        show_preview("Dead cells", red_preview)

else:
    st.info("Upload all 3 images and click **Run analysis**.")


# ------------------ SIDEBAR HISTORY ------------------
st.sidebar.subheader("History")

if len(st.session_state["history"]) == 0:
    st.sidebar.caption("No previous runs.")
else:
    for i, h in enumerate(st.session_state["history"]):
        st.sidebar.write(f"**Run #{i+1}**")
        st.sidebar.write(
            f"Nuclei: {h['nuclei']}  \n"
            f"Live: {h['live']}  \n"
            f"Dead: {h['dead']}  \n"
            f"Live%: {h['live_pct']:.2f}%"
        )
        st.sidebar.markdown("---")
