import streamlit as st
from PIL import Image
from box import Box

config = Box.from_yaml(filename="../config/config.yaml")
amenities = config.amenities

st.write("# Amenity Detection")
st.write("**Amenities to detect:** ")

amenities = st.multiselect("Select amenities to detect", amenities)

uploaded_file = st.file_uploader(":red[Choose an image...]")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)