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
- Automatic threshold & minimum-area recommendations  
- Per-stain controls (threshold + min area, slider + exact input)  
- Live/dead percentage  
- Segmentation preview  
- Sidebar history (most recent run at top)  
"""
)

# ------------------ SESSION STATE ------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

defaults = {
    "blue_thresh": 80,
    "green_thresh": 80,
    "red_thresh": 80,
    "blue_min_area": 40,
    "green_min_area": 150,
    "red_min_area": 150,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ------------------ UTILITIES ------------------
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)


def remove_scale_bar(rgb):
    """Remove scale bar and bright numbers above/below it."""
    h, w, _ = rgb.shape
    bottom_start = int(h * 0.75)
    bottom_region = rgb[bottom_start:, :]
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rgb, None

    bar_cnt = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(bar_cnt)
    clean = rgb.copy()
    abs_by = bottom_start + by

    # remove bar
    clean[abs_by: abs_by + bh, bx: bx + bw] = 0

    def remove_text(y1, y2):
        region = rgb[y1:y2, bx:bx + bw]
        gray_r = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        _, tmask = cv2.threshold(gray_r, 180, 255, cv2.THRESH_BINARY)
        tcnts, _ = cv2.findContours(tmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in tcnts:
            tx, ty, tw, th = cv2.boundingRect(c)
            if th < 40:
                clean[y1 + ty: y1 + ty + th, bx + tx: bx + tx + tw] = 0

    # above
    remove_text(max(0, abs_by - 50), abs_by)
    # below
    remove_text(abs_by + bh, min(h, abs_by + bh + 50))

    return clean, (bx, abs_by, bw, bh)


def count_cells_from_gray(gray_img, threshold, min_area):
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, bin_mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    count = int(np.sum(areas >= min_area))

    return count, bin_mask


def recommend_threshold(gray_img):
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
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return 20, "No clear cells — try lower threshold"

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
    rgb = load_image(file)
    clean_rgb, bar_box = remove_scale_bar(rgb)
    gray = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY)
    count, mask = count_cells_from_gray(gray, threshold, min_area)
    row = {"channel": channel_name, "filename": file.name, "count": count}
    preview = (file.name, rgb, clean_rgb, mask, bar_box)
    return count, row, preview


def show_recommendations(file):
    img = load_image(file)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t_rec, t_cond = recommend_threshold(gray)
    st.caption(f"{t_cond} — recommended threshold ≈ {t_rec}")
    a_rec, a_cond = recommend_min_area(gray, t_rec)
    st.caption(f"{a_cond} — recommended min area ≈ {a_rec} px")


# ------------------ SYNCED SLIDER + EXACT INPUT ------------------
def synced_param(label, key, minv, maxv, step):
    def slider_changed():
        st.session_state[key] = st.session_state[f"{key}_slider"]

    def input_changed():
        st.session_state[key] = st.session_state[f"{key}_input"]

    colA, colB = st.sidebar.columns([3, 1])

    colA.slider(
        label,
        minv,
        maxv,
        st.session_state[key],
        step=step,
        key=f"{key}_slider",
        on_change=slider_changed,
    )

    colB.number_input(
        " ",
        min_value=minv,
        max_value=maxv,
        value=st.session_state[key],
        step=1,
        key=f"{key}_input",
        on_change=input_changed,
        label_visibility="collapsed",
    )
    colB.caption("Exact")

    return st.session_state[key]


# ------------------ UPLOAD UI ------------------
st.header("1. Upload images")
col1, col2, col3 = st.columns(3)

with col1:
    blue_file = st.file_uploader("Blue / DAPI", ["png", "jpg", "jpeg", "tif", "tiff"])
    if blue_file:
        show_recommendations(blue_file)

with col2:
    green_file = st.file_uploader("Green (Live)", ["png", "jpg", "jpeg", "tif", "tiff"])
    if green_file:
        show_recommendations(green_file)

with col3:
    red_file = st.file_uploader("Red (Dead)", ["png", "jpg", "jpeg", "tif", "tiff"])
    if red_file:
        show_recommendations(red_file)


# ------------------ SIDEBAR CONTROLS ------------------
st.sidebar.header("Segmentation parameters")

st.sidebar.subheader("Thresholds")
blue_thresh = synced_param("DAPI threshold", "blue_thresh", 0, 255, 1)
green_thresh = synced_param("Live threshold", "green_thresh", 0, 255, 1)
red_thresh = synced_param("Dead threshold", "red_thresh", 0, 255, 1)

st.sidebar.subheader("Minimum object size (px)")
blue_min_area = synced_param("DAPI min area", "blue_min_area", 10, 500, 5)
green_min_area = synced_param("Live min area", "green_min_area", 10, 2000, 10)
red_min_area = synced_param("Dead min area", "red_min_area", 10, 2000, 10)

run_button = st.button("2. Run analysis")


# ------------------ RUN ANALYSIS ------------------
if run_button:
    if not (blue_file and green_file and red_file):
        st.warning("Upload all 3 images.")
    else:
        channels = [
            ("DAPI", blue_file, blue_thresh, blue_min_area),
            ("Live", green_file, green_thresh, green_min_area),
            ("Dead", red_file, red_thresh, red_min_area),
        ]

        rows = []
        previews = {}
        counts = {}

        for name, f, th, area in channels:
            total, row, preview = process_single_channel(f, th, area, name)
            rows.append(row)
            previews[name] = preview
            counts[name] = total

        blue_total = counts["DAPI"]
        green_total = counts["Live"]
        red_total = counts["Dead"]

        df = pd.DataFrame(rows)
        st.subheader("Per-image counts")
        st.dataframe(df)

        total_nuclei = blue_total if blue_total > 0 else green_total + red_total
        live_pct = 100 * green_total / total_nuclei if total_nuclei else 0
        dead_pct = 100 * red_total / total_nuclei if total_nuclei else 0

        st.subheader("Totals")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Nuclei", blue_total)
        colB.metric("Live", green_total)
        colC.metric("Dead", red_total)
        colD.metric("Total nuclei", total_nuclei)

        st.subheader("Live/Dead %")
        c1, c2 = st.columns(2)
        c1.metric("Live %", f"{live_pct:.2f}%")
        c2.metric("Dead %", f"{dead_pct:.2f}%")

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

        # -------- preview --------
        st.subheader("Segmentation preview")

        def show_preview(title, preview):
            fname, orig, clean, mask, barbox = preview
            st.markdown(f"### {title} — {fname}")
            c1, c2, c3 = st.columns(3)
            c1.image(orig, caption="Original")
            c2.image(clean, caption="Cleaned (scale bar + numbers removed)")
            c3.image(mask, caption="Binary mask")
            if barbox:
                x, y, w, h = barbox
                st.caption(f"Removed scale bar at x={x}, y={y}, size={w}×{h}")

        show_preview("DAPI", previews["DAPI"])
        show_preview("Live", previews["Live"])
        show_preview("Dead", previews["Dead"])


# ------------------ SIDEBAR HISTORY ------------------
st.sidebar.subheader("History")
if not st.session_state["history"]:
    st.sidebar.caption("No previous runs.")
else:
    for i, h in enumerate(st.session_state["history"]):
        title = f"Run #{i+1} (most recent)" if i == 0 else f"Run #{i+1}"
        st.sidebar.write(f"**{title}**")
        st.sidebar.write(f"Nuclei: {h['nuclei']}")
        st.sidebar.write(f"Live: {h['live']}")
        st.sidebar.write(f"Dead: {h['dead']}")
        st.sidebar.write(f"Live%: {h['live_pct']:.2f}%")
        st.sidebar.write(f"Dead%: {h['dead_pct']:.2f}%")
        st.sidebar.markdown("---")

