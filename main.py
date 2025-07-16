import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Detection with Viola-Jones (Haar Cascades)")
st.write("""
**Instructions:**  
1. Upload an image containing faces.  
2. Adjust the scaleFactor and minNeighbors parameters to improve detection accuracy.  
3. Choose the color for the rectangle around detected faces.  
4. Click "Detect Faces" to see results.  
5. Optionally, download the image with detected faces.
""")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Parameter sliders
scale_factor = st.slider("Adjust scaleFactor (default 1.1)", min_value=1.01, max_value=2.0, value=1.1, step=0.01)
min_neighbors = st.slider("Adjust minNeighbors (default 5)", min_value=1, max_value=20, value=5, step=1)
rect_color = st.color_picker("Pick rectangle color", "#FF0000")  # default red

if uploaded_file is not None:
    # Read the image via PIL then convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if st.button("Detect Faces"):
        # Detect faces
        faces = face_cascade.detectMultiScale(
            image_cv,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        st.write(f"Found {len(faces)} face(s)")

        # Convert hex color to BGR tuple
        hex_color = rect_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), bgr_color, 2)

        # Convert back to RGB for display
        result_img = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption='Detected Faces', use_column_width=True)

        # Save the image to a temporary file
        save_path = "detected_faces.png"
        cv2.imwrite(save_path, image_cv)

        # Provide a download button for the user
        with open(save_path, "rb") as file:
            btn = st.download_button(
                label="Download Image with Detected Faces",
                data=file,
                file_name="detected_faces.png",
                mime="image/png"
            )
