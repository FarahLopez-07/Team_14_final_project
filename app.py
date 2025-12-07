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

- **Blue** = nuclei (DAPI/Hoechst)  
- **Green** = live cells (Calcein-AM)  
- **Red** = dead cells (EthD-1)

This app includes:

- Scale-bar + number removal  
- Threshold & area controls (slider + exact input, always synced)  
- Recommended parameters  
- Live/dead percentages  
- History (most recent at top)  
"""
)

# ------------------ SESSION STATE INIT ------------------

defaults = {
    # Thresholds
    "blue_thresh": 80,
    "green_thresh": 80,
    "red_thresh": 80,

    # Areas
    "blue_min_area": 40,
    "green_min_area": 150,
    "red_min_area": 150,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------ UTILS ------------------

def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def remove_scale_bar(rgb):
    h, w, _ = rgb.shape
    bottom = int(h * 0.75)
    region = rgb[bottom:]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return rgb, None

    largest = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)

    clean = rgb.copy()
    abs_y = bottom + by

    clean[abs_y:abs_y+bh, bx:bx+bw] = 0  # remove bar

    # remove numbers above & below
    # above region
    top_y1 = max(0, abs_y - 50)
    above = rgb[top_y1:abs_y, bx:bx+bw]
    gray_above = cv2.cvtColor(above, cv2.COLOR_RGB2GRAY)
    _, mA = cv2.threshold(gray_above, 180, 255, cv2.THRESH_BINARY)
    cntA, _ = cv2.findContours(mA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cntA:
        tx, ty, tw, th = cv2.boundingRect(c)
        if th < 40:
            clean[top_y1+ty: top_y1+ty+th, bx+tx: bx+tx+tw] = 0

    # below region
    bot_y1 = abs_y + bh
    bot_y2 = min(h, abs_y + bh + 50)
    below = rgb[bot_y1:bot_y2, bx:bx+bw]
    gray_below = cv2.cvtColor(below, cv2.COLOR_RGB2GRAY)
    _, mB = cv2.threshold(gray_below, 180, 255, cv2.THRESH_BINARY)
    cntB, _ = cv2.findContours(mB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cntB:
        tx, ty, tw, th = cv2.boundingRect(c)
        if th < 40:
            clean[bot_y1+ty: bot_y1+ty+th, bx+tx: bx+tx+tw] = 0

    return clean, (bx, abs_y, bw, bh)

def count_cells(gray, thresh, min_area):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    count = int(np.sum(areas >= min_area))

    return count, mask

def recommend_threshold(gray):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if otsu < 60:
        cond = "Very clean background"
    elif otsu < 100:
        cond = "Clean background"
    elif otsu < 140:
        cond = "Slightly noisy background"
    else:
        cond = "Uneven/bright background"

    return int(otsu), cond

def recommend_area(gray, threshold):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return 20, "No clear objects"

    med = np.median(areas)
    rec = int(max(10, 0.4 * med))

    if med < 60:
        cond = "Cells appear small"
    elif med < 150:
        cond = "Cells appear medium-sized"
    else:
        cond = "Cells appear large"

    return rec, cond


# ------------------ UPLOAD SECTION ------------------

st.header("1. Upload images")

c1, c2, c3 = st.columns(3)

with c1:
    blue_file = st.file_uploader("Blue (DAPI)", ["png","jpg","jpeg","tif","tiff"])
    if blue_file:
        g = cv2.cvtColor(load_image(blue_file), cv2.COLOR_RGB2GRAY)
        t, tc = recommend_threshold(g)
        st.caption(f"{tc} — recommended threshold ≈ {t}")
        a, ac = recommend_area(g, t)
        st.caption(f"{ac} — recommended area ≈ {a}px")

with c2:
    green_file = st.file_uploader("Green (Live)", ["png","jpg","jpeg","tif","tiff"])
    if green_file:
        g = cv2.cvtColor(load_image(green_file), cv2.COLOR_RGB2GRAY)
        t, tc = recommend_threshold(g)
        st.caption(f"{tc} — recommended threshold ≈ {t}")
        a, ac = recommend_area(g, t)
        st.caption(f"{ac} — recommended area ≈ {a}px")

with c3:
    red_file = st.file_uploader("Red (Dead)", ["png","jpg","jpeg","tif","tiff"])
    if red_file:
        g = cv2.cvtColor(load_image(red_file), cv2.COLOR_RGB2GRAY)
        t, tc = recommend_threshold(g)
        st.caption(f"{tc} — recommended threshold ≈ {t}")
        a, ac = recommend_area(g, t)
        st.caption(f"{ac} — recommended area ≈ {a}px")


# ------------------ SYNCHRONIZED CONTROLS ------------------

st.sidebar.header("Segmentation Parameters")


### FUNCTION: generic sync for slider & input
def slider_input_pair(label, key_name, minv, maxv, step, sidebar):
    """
    Creates a synced slider + exact number box.
    key_name is the session_state variable that holds the TRUE value.
    """

    with sidebar:
        colA, colB = st.columns([3, 1])

        with colA:
            slider_val = st.slider(
                f"{label} (slider)",
                minv, maxv,
                st.session_state[key_name],
                step=step,
                key=f"{key_name}_slider",
            )

        with colB:
            num_val = st.number_input(
                " ",
                min_value=minv,
                max_value=maxv,
                value=st.session_state[key_name],
                step=1,
                key=f"{key_name}_input",
                label_visibility="collapsed",
            )
            colB.caption("Exact")

        # sync both directions
        if slider_val != st.session_state[key_name]:
            st.session_state[key_name] = slider_val
        if num_val != st.session_state[key_name]:
            st.session_state[key_name] = num_val

    return st.session_state[key_name]


### THRESHOLDS
st.sidebar.subheader("Thresholds")

blue_thresh = slider_input_pair("DAPI threshold", "blue_thresh", 0, 255, 1, st.sidebar)
green_thresh = slider_input_pair("Live threshold", "green_thresh", 0, 255, 1, st.sidebar)
red_thresh = slider_input_pair("Dead threshold", "red_thresh", 0, 255, 1, st.sidebar)

### AREAS
st.sidebar.subheader("Minimum Object Size (pixels)")

blue_min_area = slider_input_pair("DAPI min area", "blue_min_area", 10, 500, 5, st.sidebar)
green_min_area = slider_input_pair("Live min area", "green_min_area", 10, 2000, 10, st.sidebar)
red_min_area = slider_input_pair("Dead min area", "red_min_area", 10, 2000, 10, st.sidebar)


# ------------------ ANALYSIS BUTTON ------------------

run_button = st.button("2. Run analysis")


# ------------------ MAIN ANALYSIS ------------------

if run_button:

    if not (blue_file and green_file and red_file):
        st.warning("Upload all 3 images first.")
        st.stop()

    def process(file, thresh, area, label):
        rgb = load_image(file)
        clean, bar = remove_scale_bar(rgb)
        gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        count, mask = count_cells(gray, thresh, area)
        return count, (file.name, rgb, clean, mask, bar)

    blue_count, blue_prev = process(blue_file, blue_thresh, blue_min_area, "DAPI")
    green_count, green_prev = process(green_file, green_thresh, green_min_area, "Live")
    red_count, red_prev = process(red_file, red_thresh, red_min_area, "Dead")

    total_nuclei = blue_count if blue_count > 0 else (green_count + red_count)

    live_pct = 100 * green_count / total_nuclei if total_nuclei else 0
    dead_pct = 100 * red_count / total_nuclei if total_nuclei else 0

    # history
    st.session_state["history"].insert(
        0,
        dict(
            nuclei=total_nuclei,
            live=green_count,
            dead=red_count,
            live_pct=live_pct,
            dead_pct=dead_pct,
        )
    )

    # results
    st.subheader("Counts")
    st.write(pd.DataFrame([
        dict(Channel="DAPI", Count=blue_count),
        dict(Channel="Live", Count=green_count),
        dict(Channel="Dead", Count=red_count),
    ]))

    st.subheader("Live/Dead Percentages")
    c1, c2 = st.columns(2)
    c1.metric("Live %", f"{live_pct:.2f}%")
    c2.metric("Dead %", f"{dead_pct:.2f}%")

    # previews
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
            st.caption(f"Removed scale bar at: x={x}, y={y}, size={w}×{h}")

    show("DAPI", blue_prev)
    show("Live", green_prev)
    show("Dead", red_prev)


# ------------------ HISTORY ------------------

st.sidebar.subheader("History (Newest First)")
if len(st.session_state["history"]) == 0:
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
