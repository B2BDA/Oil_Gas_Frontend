import streamlit as st
from PIL import Image
# credits and reference: https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0
st.title("Object Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # this image goes into the predict function of the model
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button("Predict"):
        st.write("Predicted Class is: ")
        # make a predicted_class variable and append in the above statement
    