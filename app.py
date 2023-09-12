# - Importing the dependencies
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# - CSS Styling
st.markdown(
    """
    <style>
    .centered-heading {
        text-align: center;
        padding-bottom: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# - Defining a function to load the YOLO model and return it
@st.cache_data()
def load_model():
    return YOLO('best.pt')

# - Loading the pre-trained YOLOv8l model
model = load_model()

st.markdown("<h1 class='centered-heading'>Vehicle Registration Plate Detection App (YoloV8)</h1>", unsafe_allow_html=True)

# - Uploading multiple images
with st.form("my-form", clear_on_submit=True):
        uploaded_images = st.file_uploader("Upload an image or multiple images for vehicle registration plate detection", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("submit")

if uploaded_images:
    for uploaded_image in uploaded_images:
        # - Converting the uploaded image to a format compatible with YOLO
        img_bytes = uploaded_image.read()
        image = Image.open(io.BytesIO(img_bytes))

        results = model(source=image)

        for result in results:
            boxes = result.boxes
            
        number_of_plates = boxes.shape[0]

        # - Displaying the detection results for each image
        for result in results:
            im_array = result.plot()  
            im = Image.fromarray(im_array[..., ::-1])

            # - Display the image with bounding boxes
            st.image(im, caption=f"Image {uploaded_images.index(uploaded_image) + 1}. Object Detection Result - {number_of_plates} vehicle registration plates are detected.", use_column_width=True)
