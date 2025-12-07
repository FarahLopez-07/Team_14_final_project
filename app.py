import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd

st.set_page_config(page_title="Live/Dead Cell Counter", layout="wide")

st.title("Fluorescence Cell Counter (DAPI / Live / Dead)")
st.write(
    """
Upload one fluorescence image per channel:

- **Blue** = DAPI nuclei  
- **Green** = live cells (Calcein-AM)  
- **Red** = dead cells (EthD-1)

This app provides:
- Automatic scale-bar + number removal  
- Slider + exact threshold/area input (synced only when changed)  
- Recommended values  
- Binary preview  
- Live/dead percentages  
- Run history (newest at top)  
"""
)

# ------------------ DEFAULT STATE ------------------

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

st.session_state.setdefault("history", [])


# ------------------ UTIL FUNCTIONS ------------------

def load_image(uploaded_file):
    return np.array(Image.open(uploaded_file).convert("RGB"))


def remove_scale_bar(rgb):
    h, w, _ = rgb.shape
    bottom = int(h * 0.75)
    region = rgb[bottom:]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rgb, None

    largest = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)
    clean = rgb.copy()
    abs_y = bottom + by
    clean[abs_y:abs_y+bh, bx:bx+bw] = 0

    def remove_text(y1, y2):
        region = rgb[y1:y2, bx:bx + bw]
        gray_r = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        _, mask_r = cv2.threshold(gray_r, 180, 255, cv2.THRESH_BINARY)
        cnts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_r:
            x, y, w_, h_ = cv2.boundingRect(c)
            if h_ < 40:
                clean[y1+y: y1+y+h_, bx+x: bx+x+w_] = 0

    remove_text(max(0, abs_y - 50), abs_y)
    remove_text(abs_y + bh, min(h, abs_y + bh + 50))

    return clean, (bx, abs_y, bw, bh)


def count_cells(gray, thresh, min_area):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    count = int(np.sum(areas >= min_area))
    return count, mask


def recommend_threshold(gray):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    otsu, _ = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if otsu < 60: txt = "Very clean"
    elif otsu < 100: txt = "Clean background"
    elif otsu < 140: txt = "Slightly noisy"
    else: txt = "Uneven background"

    return int(otsu), txt


def recommend_area(gray, thresh):
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return 20, "No detected cells"

    med = np.median(areas)
    rec = int(max(10, med * 0.4))

    if med < 60: txt = "Small cells"
    elif med < 150: txt = "Medium cells"
    else: txt = "Large cells"

    return rec, txt


# ------------------ SMART SYNCED SLIDER + EXACT INPUT ------------------

def synced_param(label, key, minv, maxv, step):

    def slider_changed():
        st.session_state[key] = st.session_state[f"{key}_slider"]

    def input_changed():
        st.session_state[key] = st.session_state[f"{key}_input"]

    colA, colB = st.sidebar.columns([3, 1])

    colA.slider(
        f"{label}",
        minv, maxv,
        st.session_state[key],
        step=step,
        key=f"{key}_slider",
        on_change=slider_changed
    )

    colB.number_input(
        "Exact",
        min_value=minv,
        max_value=maxv,
        value=st.session_state[key],
        step=1,
        key=f"{key}_input",
        on_change=input_changed,
        label_visibility="collapsed"
    )

    return st.session_state[key]


# ------------------ UPLOAD SECTION ------------------

st.header("1. Upload images")
c1, c2, c3 = st.columns(3)

with c1:
    blue_file = st.file_uploader("Blue (DAPI)", ["png","jpg","jpeg","tif","tiff"])
    if blue_file:
        g = cv2.cvtColor(load_image(blue_file), cv2.COLOR_RGB2GRAY)
        t, tc = recommend_threshold(g)
        a, ac = recommend_area(g, t)
        st.caption(f"{tc} — recommended threshold {t}")
        st.caption(f"{ac} — recommended area {a}px")

with c2:
    green_file = st.file_uploader("Green (Live)", ["png","jpg","jpeg","tif","tiff"])
    if green_file:
        g = cv2.cvtColor(load_image(green_file), cv2.COLOR_RGB2GRAY)
        t, tc = recommend_threshold(g)
        a, ac = recommend_area(g, t)
        st.caption(f"{tc} — recommended threshold {t}")
        st.caption(f"{ac} — recommended area {a}px")

with c3:
    red_file = st.file_uploader("Red (Dead)", ["png","jpg","jpeg","tif","tiff"])
    if red_file:
        g = cv2.cvtColor(load_image(red_file), cv2.COLOR_RGB2GRAY)
        t, tc = recommend_threshold(g)
        a, ac = recommend_area(g, t)
        st.caption(f"{tc} — recommended threshold {t}")
        st.caption(f"{ac} — recommended area {a}px")


# ------------------ PARAMETER SIDEBAR ------------------

st.sidebar.header("Segmentation Parameters")

st.sidebar.subheader("Thresholds")
blue_thresh  = synced_param("DAPI threshold", "blue_thresh",  0, 255, 1)
green_thresh = synced_param("Live threshold", "green_thresh", 0, 255, 1)
red_thresh   = synced_param("Dead threshold", "red_thresh",   0, 255, 1)

st.sidebar.subheader("Minimum Object Size (px)")
blue_min_area  = synced_param("DAPI min area", "blue_min_area", 10, 500, 5)
green_min_area = synced_param("Live min area", "green_min_area", 10, 2000, 10)
red_min_area   = synced_param("Dead min area", "red_min_area", 10, 2000, 10)


# ------------------ RUN BUTTON ------------------

run_button = st.button("2. Run analysis")


# ------------------ MAIN ANALYSIS ------------------

if run_button:

    if not (blue_file and green_file and red_file):
        st.warning("Upload all 3 images before running.")
        st.stop()

    def process(file, thresh, area):
        rgb = load_image(file)
        clean, bar = remove_scale_bar(rgb)
        gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        count, mask = count_cells(gray, thresh, area)
        return count, (file.name, rgb, clean, mask, bar)

    blue_count, blue_prev   = process(blue_file,  blue_thresh,  blue_min_area)
    green_count, green_prev = process(green_file, green_thresh, green_min_area)
    red_count, red_prev     = process(red_file,   red_thresh,   red_min_area)

    total_nuclei = blue_count if blue_count > 0 else (green_count + red_count)
    live_pct = 100 * green_count / total_nuclei if total_nuclei else 0
    dead_pct = 100 * red_count / total_nuclei if total_nuclei else 0

    st.session_state["history"].insert(0, dict(
        nuclei=total_nuclei,
        live=green_count,
        dead=red_count,
        live_pct=live_pct,
        dead_pct=dead_pct,
    ))

    st.subheader("Counts")
    st.write(pd.DataFrame([
        {"Channel": "DAPI", "Count": blue_count},
        {"Channel": "Live", "Count": green_count},
        {"Channel": "Dead", "Count": red_count},
    ]))

    st.subheader("Live/Dead Percentages")
    c1, c2 = st.columns(2)
    c1.metric("Live %", f"{live_pct:.2f}%")
    c2.metric("Dead %", f"{dead_pct:.2f}%")

    st.subheader("Segmentation Preview")

    def show(title, data):
        fname, orig, clean, mask, bar = data
        st.markdown(f"### {title} — {fname}")
        a, b, c = st.columns(3)
        a.image(orig, caption="Original")
        b.image(clean, caption="Cleaned")
        c.image(mask, caption="Binary Mask")
        if bar:
            x, y, w, h = bar
            st.caption(f"Scale bar removed at x={x}, y={y}, size {w}×{h}")

    show("DAPI", blue_prev)
    show("Live", green_prev)
    show("Dead", red_prev)


# ------------------ HISTORY ------------------

st.sidebar.subheader("History (Newest First)")
if not st.session_state["history"]:
    st.sidebar.caption("No runs yet.")
else:
    for i, h in enumerate(st.session_state["history"]):
        st.sidebar.write(f"**Run #{i+1}**")
        st.sidebar.write(f"Nuclei: {h['nuclei']}")
        st.sidebar.write(f"Live: {h['live']}")
        st.sidebar.write(f"Dead: {h['dead']}")
        st.sidebar.write(f"Live %: {h['live_pct']:.2f}%")
        st.sidebar.write(f"Dead %: {h['dead_pct']:.2f}%")
        st.sidebar.markdown("---")



