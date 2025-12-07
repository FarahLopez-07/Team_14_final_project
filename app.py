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
- Per-stain controls (threshold + min area, slider + exact input)  
- Live/dead percentage  
- Segmentation preview  
- Sidebar history (most recent run at top)  
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
    """
    h, w, _ = rgb.shape

    # detect bar in bottom 25% of image
    bottom_start = int(h * 0.75)
    bottom_region = rgb[bottom_start:, :]
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return rgb, None

    bar_cnt = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(bar_cnt)

    clean = rgb.copy()

    abs_by = bottom_start + by
    clean[abs_by: abs_by + bh, bx: bx + bw] = 0

    # remove text ABOVE (up to 50 px)
    top_region_y1 = max(0, abs_by - 50)
    text_above = rgb[top_region_y1:abs_by, bx:bx + bw]
    gray_above = cv2.cvtColor(text_above, cv2.COLOR_RGB2GRAY)
    _, tmaskA = cv2.threshold(gray_above, 180, 255, cv2.THRESH_BINARY)
    tcntsA, _ = cv2.findContours(tmaskA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in tcntsA:
        tx, ty, tw, th = cv2.boundingRect(c)
        if th < 40:
            clean[top_region_y1 + ty: top_region_y1 + ty + th,
                  bx + tx: bx + tx + tw] = 0

    # remove text BELOW (up to 50 px)
    bot_y1 = abs_by + bh
    bot_y2 = min(h, abs_by + bh + 50)
    text_below = rgb[bot_y1:bot_y2, bx:bx + bw]
    gray_below = cv2.cvtColor(text_below, cv2.COLOR_RGB2GRAY)
    _, tmaskB = cv2.threshold(gray_below, 180, 255, cv2.THRESH_BINARY)
    tcntsB, _ = cv2.findContours(tmaskB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in tcntsB:
        tx, ty, tw, th = cv2.boundingRect(c)
        if th < 40:
            clean[bot_y1 + ty: bot_y1 + ty + th,
                  bx + tx: bx + tx + tw] = 0

    return clean, (bx, abs_by, bw, bh)


def count_cells_from_gray(gray_img, threshold, min_area):
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(
            gray_img, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    _, bin_mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    count = int(np.sum(areas >= min_area))

    return count, bin_mask


def recommend_threshold(gray_img):
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(
            gray_img, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    otsu_val, _ = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

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
        gray_img = cv2.normalize(
            gray_img, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

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


# ------------------ UPLOAD UI ------------------
st.header("1. Upload images")
col1, col2, col3 = st.columns(3)


def show_recommendations(file):
    img = load_image(file)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    t_rec, t_cond = recommend_threshold(gray)
    st.caption(f"{t_cond} — recommended threshold ≈ {t_rec}")

    a_rec, a_cond = recommend_min_area(gray, t_rec)
    st.caption(f"{a_cond} — recommended min area ≈ {a_rec} px")


with col1:
    blue_file = st.file_uploader(
        "Blue / DAPI", ["png", "jpg", "jpeg", "tif", "tiff"]
    )
    if blue_file:
        show_recommendations(blue_file)

with col2:
    green_file = st.file_uploader(
        "Green (Live)", ["png", "jpg", "jpeg", "tif", "tiff"]
    )
    if green_file:
        show_recommendations(green_file)

with col3:
    red_file = st.file_uploader(
        "Red (Dead)", ["png", "jpg", "jpeg", "tif", "tiff"]
    )
    if red_file:
        show_recommendations(red_file)


# ------------------ SYNC CALLBACKS ------------------
# Thresholds
def sync_blue_th_from_slider():
    st.session_state["blue_thresh_input"] = int(
        st.session_state["blue_thresh_slider"]
    )


def sync_blue_th_from_input():
    st.session_state["blue_thresh_slider"] = int(
        st.session_state["blue_thresh_input"]
    )


def sync_green_th_from_slider():
    st.session_state["green_thresh_input"] = int(
        st.session_state["green_thresh_slider"]
    )


def sync_green_th_from_input():
    st.session_state["green_thresh_slider"] = int(
        st.session_state["green_thresh_input"]
    )


def sync_red_th_from_slider():
    st.session_state["red_thresh_input"] = int(
        st.session_state["red_thresh_slider"]
    )


def sync_red_th_from_input():
    st.session_state["red_thresh_slider"] = int(
        st.session_state["red_thresh_input"]
    )


# Min areas
def sync_blue_from_slider():
    st.session_state["blue_min_area_input"] = int(
        st.session_state["blue_min_area_slider"]
    )


def sync_blue_from_input():
    st.session_state["blue_min_area_slider"] = int(
        st.session_state["blue_min_area_input"]
    )


def sync_green_from_slider():
    st.session_state["green_min_area_input"] = int(
        st.session_state["green_min_area_slider"]
    )


def sync_green_from_input():
    st.session_state["green_min_area_slider"] = int(
        st.session_state["green_min_area_input"]
    )


def sync_red_from_slider():
    st.session_state["red_min_area_input"] = int(
        st.session_state["red_min_area_slider"]
    )


def sync_red_from_input():
    st.session_state["red_min_area_slider"] = int(
        st.session_state["red_min_area_input"]
    )


# ------------------ SIDEBAR ------------------
st.sidebar.header("Segmentation parameters")

# Thresholds (slider + exact)
st.sidebar.subheader("Thresholds")

tb1, tb2 = st.sidebar.columns([3, 1])
with tb1:
    st.slider(
        "DAPI threshold (slider)",
        0,
        255,
        80,
        key="blue_thresh_slider",
        on_change=sync_blue_th_from_slider,
    )
with tb2:
    st.number_input(
        " ",
        min_value=0,
        max_value=255,
        key="blue_thresh_input",
        on_change=sync_blue_th_from_input,
        label_visibility="collapsed",
    )
tb2.caption("Exact")

tg1, tg2 = st.sidebar.columns([3, 1])
with tg1:
    st.slider(
        "Live threshold (slider)",
        0,
        255,
        80,
        key="green_thresh_slider",
        on_change=sync_green_th_from_slider,
    )
with tg2:
    st.number_input(
        "  ",
        min_value=0,
        max_value=255,
        key="green_thresh_input",
        on_change=sync_green_th_from_input,
        label_visibility="collapsed",
    )
tg2.caption("Exact")

tr1, tr2 = st.sidebar.columns([3, 1])
with tr1:
    st.slider(
        "Dead threshold (slider)",
        0,
        255,
        80,
        key="red_thresh_slider",
        on_change=sync_red_th_from_slider,
    )
with tr2:
    st.number_input(
        "   ",
        min_value=0,
        max_value=255,
        key="red_thresh_input",
        on_change=sync_red_th_from_input,
        label_visibility="collapsed",
    )
tr2.caption("Exact")

blue_thresh = int(st.session_state.get("blue_thresh_slider", 80))
green_thresh = int(st.session_state.get("green_thresh_slider", 80))
red_thresh = int(st.session_state.get("red_thresh_slider", 80))

# Min areas (slider + exact)
st.sidebar.subheader("Minimum object size (px)")

b_col1, b_col2 = st.sidebar.columns([3, 1])
with b_col1:
    st.slider(
        "DAPI min area (slider)",
        10,
        500,
        40,
        step=5,
        key="blue_min_area_slider",
        on_change=sync_blue_from_slider,
    )
with b_col2:
    st.number_input(
        "    ",
        min_value=1,
        max_value=3000,
        key="blue_min_area_input",
        on_change=sync_blue_from_input,
        label_visibility="collapsed",
    )
b_col2.caption("Exact")

g_col1, g_col2 = st.sidebar.columns([3, 1])
with g_col1:
    st.slider(
        "Live min area (slider)",
        10,
        2000,
        150,
        step=10,
        key="green_min_area_slider",
        on_change=sync_green_from_slider,
    )
with g_col2:
    st.number_input(
        "     ",
        min_value=1,
        max_value=5000,
        key="green_min_area_input",
        on_change=sync_green_from_input,
        label_visibility="collapsed",
    )
g_col2.caption("Exact")

r_col1, r_col2 = st.sidebar.columns([3, 1])
with r_col1:
    st.slider(
        "Dead min area (slider)",
        10,
        2000,
        150,
        step=10,
        key="red_min_area_slider",
        on_change=sync_red_from_slider,
    )
with r_col2:
    st.number_input(
        "      ",
        min_value=1,
        max_value=5000,
        key="red_min_area_input",
        on_change=sync_red_from_input,
        label_visibility="collapsed",
    )
r_col2.caption("Exact")

blue_min_area = int(st.session_state.get("blue_min_area_slider", 40))
green_min_area = int(st.session_state.get("green_min_area_slider", 150))
red_min_area = int(st.session_state.get("red_min_area_slider", 150))

run_button = st.button("2. Run analysis")


# ------------------ RUN ANALYSIS ------------------
if run_button:
    if not (blue_file and green_file and red_file):
        st.warning("Upload all 3 images.")
    else:
        blue_total, blue_row, blue_preview = process_single_channel(
            blue_file, blue_thresh, blue_min_area, "DAPI"
        )
        green_total, green_row, green_preview = process_single_channel(
            green_file, green_thresh, green_min_area, "Live"
        )
        red_total, red_row, red_preview = process_single_channel(
            red_file, red_thresh, red_min_area, "Dead"
        )

        df = pd.DataFrame([blue_row, green_row, red_row])
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
        col1, col2 = st.columns(2)
        col1.metric("Live %", f"{live_pct:.2f}%")
        col2.metric("Dead %", f"{dead_pct:.2f}%")

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

            col1, col2, col3 = st.columns(3)
            col1.image(orig, caption="Original")
            col2.image(clean, caption="Cleaned (scale bar + numbers removed)")
            col3.image(mask, caption="Binary mask")

            if barbox:
                x, y, w, h = barbox
                st.caption(f"Removed scale bar at x={x}, y={y}, size={w}×{h}")

        show_preview("DAPI", blue_preview)
        show_preview("Live", green_preview)
        show_preview("Dead", red_preview)


# ------------------ SIDEBAR HISTORY ------------------
st.sidebar.subheader("History")
if not st.session_state["history"]:
    st.sidebar.caption("No previous runs.")
else:
    for i, h in enumerate(st.session_state["history"]):
        title = "Run #{0} (most recent)".format(i + 1) if i == 0 else f"Run #{i+1}"
        st.sidebar.write(f"**{title}**")
        st.sidebar.write(f"Nuclei: {h['nuclei']}")
        st.sidebar.write(f"Live: {h['live']}")
        st.sidebar.write(f"Dead: {h['dead']}")
        st.sidebar.write(f"Live%: {h['live_pct']:.2f}%")
        st.sidebar.markdown("---")
