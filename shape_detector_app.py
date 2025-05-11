import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---- Shape detection function ----
def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = float(w) / h
        return "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    elif 10 <= vertices <= 14:
        return "Star"
    elif vertices > 14:
        return "Circle"
    else:
        return "Heart"

# ---- Streamlit UI ----
st.title("🔍 Shape & Area Detection")

uploaded_file = st.file_uploader("Upload an image with shapes", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img = img.copy()

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Skip very small artifacts
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        shape_name = detect_shape(cnt)

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{shape_name} {area:.1f}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    st.subheader("📤 Original Image")
    st.image(original_img, channels="BGR")

    st.subheader("✅ Detected Shapes with Areas")
    st.image(img, channels="BGR")
