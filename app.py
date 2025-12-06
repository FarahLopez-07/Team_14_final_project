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

App features:
- Automatic **scale-bar removal**
- Threshold recommendations
- Per-stain threshold and minimum pixel size
- Live/dead calculation
- Segmentation preview
- Sidebar **history** of previous runs
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
    Detect and remove a bright scale bar typically located at the bottom.
    Removes the largest bright rectangular object in the lower 20% of the image.
    """
    h, w, _ = rgb.shape
    bottom_region = rgb[int(h * 0.8):, :]  # bottom 20%
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)

    # Threshold to find bright objects
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return rgb, None  # nothing removed

    # Pick largest contour = scale bar
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w2, h2 = cv2.boundingRect(cnt)

    # Create a copy to remove scale bar
    clean = rgb.copy()
    clean[int(h * 0.8) + y : int(h * 0.8) + y + h2, x : x + w2] = 0  # black region

    return clean, (x, int(h * 0.8) + y, w2, h2)


def count_cells_from_gray(gray_img, threshold, min_area):
    """Threshold grayscale image and count connected components > min_area."""
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

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


def process_single_channel(file, threshold, min_area, channel_name):
    if file is None:
        return 0, None, None, None, None

    rgb = load_image(file)

    # ---- Remove scale bar ----
    clean_rgb, scale_bar_box = remove_scale_bar(rgb)

    gray = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY)

    count, bin_mask = count_cells_from_gray(gray, threshold, min_area)

    row = {"channel": channel_name, "filename": file.name, "count": count}

    preview = (file.name, rgb, clean_rgb, bin_mask, scale_bar_box)
    return count, row, preview


# ------------------ UPLOAD UI ------------------
st.header("1. Upload images")
col1, col2, col3 = st.columns(3)

with col1:
    blue_file = st.file_uploader("Blue / DAPI nuclei", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if blue_file:
        img = load_image(blue_file)
        rec, cond = recommend_threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        st.caption(f"{cond} — recommended threshold ≈ {rec}")

with col2:
    green_file = st.file_uploader("Green (live cells)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if green_file:
        img = load_image(green_file)
        rec, cond = recommend_threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        st.caption(f"{cond} — recommended threshold ≈ {rec}")

with col3:
    red_file = st.file_uploader("Red (dead cells)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if red_file:
        img = load_image(red_file)
        rec, cond = recommend_threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        st.caption(f"{cond} — recommended threshold ≈ {rec}")


# ------------------ SIDEBAR SETTINGS ------------------
st.sidebar.header("Segmentation parameters")

st.sidebar.subheader("Intensity thresholds")
blue_thresh = st.sidebar.slider("DAPI threshold", 0, 255, 80)
green_thresh = st.sidebar.slider("Live cell threshold", 0, 255, 80)
red_thresh = st.sidebar.slider("Dead cell threshold", 0, 255, 80)

st.sidebar.subheader("Minimum object size (pixels)")
blue_min_area = st.sidebar.slider("DAPI nuclei min area", 10, 2000, 30)
green_min_area = st.sidebar.slider("Live cell min area", 10, 5000, 100)
red_min_area = st.sidebar.slider("Dead cell min area", 10, 5000, 100)

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
        colD.metric("Total nuclei used", total_nuclei)

        st.subheader("Live/Dead percentages")
        col1, col2 = st.columns(2)
        col1.metric("Live %", f"{live_pct:.2f}%")
        col2.metric("Dead %", f"{dead_pct:.2f}%")

        # Add to history
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
            if preview is None:
                return
            fname, orig, clean, mask, scale_bar = preview

            st.markdown(f"### {title} — {fname}")

            col1, col2, col3 = st.columns(3)
            col1.image(orig, caption="Original image", use_column_width=True)
            col2.image(clean, caption="Scale-bar removed", use_column_width=True)
            col3.image(mask, caption="Binary mask", use_column_width=True)

            if scale_bar:
                x, y, w, h = scale_bar
                st.caption(f"Scale bar removed at location x={x}, y={y}, size={w}×{h}")

        show_preview("DAPI nuclei", blue_preview)
        show_preview("Live cells", green_preview)
        show_preview("Dead cells", red_preview)

else:
    st.info("Upload images and click **Run analysis** to start.")


# ------------------ SIDEBAR HISTORY ------------------
st.sidebar.subheader("History (previous runs)")

if len(st.session_state["history"]) == 0:
    st.sidebar.caption("No previous runs yet.")
else:
    for i, h in enumerate(st.session_state["history"]):
        st.sidebar.write(f"**Run #{i+1}**")
        st.sidebar.write(
            f"Nuclei: {h['nuclei']}, Live: {h['live']}, Dead: {h['dead']}, "
            f"Live%: {h['live_pct']:.2f}%"
        )
        st.sidebar.markdown("---")
