import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🔍 Circular Coin Detection Web App (OpenCV)")

# Sidebar instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write("""
- Upload an image containing coins.
- Or click **Use Example Image** to try a demo.
- Then click **Detect Coins**.
""")

# Load example image
def load_example_image():
    try:
        image = Image.open("12sample.jpg")
        return image
    except:
        st.error("❌ Example image '12sample.jpg' not found. Please make sure it's in the same folder.")
        return None

# Coin detection logic
def detect_coins(image):
    img = np.array(image.convert("RGB"))  # ensure correct format
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # reduce noise

    # Adaptive threshold for better detection
    threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_count = 0
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7 and area > 80:  # Filter small noise
            coin_count += 1
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, (0, 255, 0), 2)

    return img, coin_count

# Initialize session state
if "image" not in st.session_state:
    st.session_state.image = None

# Buttons and uploader
uploaded_image = st.file_uploader("📤 Upload your own image...", type=["jpg", "jpeg", "png"])
use_example = st.button("🧪 Use Example Image")

# Store image in session_state
if uploaded_image is not None:
    st.session_state.image = Image.open(uploaded_image)

if use_example:
    st.session_state.image = load_example_image()

# Display the chosen image
if st.session_state.image is not None:
    st.image(st.session_state.image, caption="📌 Image to be processed", use_container_width=True)

    if st.button("🔎 Detect Coins"):
        result, count = detect_coins(st.session_state.image)
        st.image(result, caption="✅ Detection Result", use_container_width=True)

        if count > 0:
            st.success(f"✅ Detected {count} circular object(s) that may be coins.")
        else:
            st.warning("⚠ No circular coins detected. The image might not contain coins or shapes are unclear.")
else:
    st.info("👆 Upload an image or click **Use Example Image** to start.")

# Footer
st.markdown("---")
st.markdown("👨‍💻 *Created by Naps     | Powered by OpenCV and Streamlit*")
