
import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf

st.title("Trang web chuẩn đoán ung thư")
st.write("Vui lòng up load file ảnh .png, ...")

file_uploader = st.file_uploader("Chọn một ảnh để chuẩn đoán")

class_names = ['cancer', 'non-cancer']

if file_uploader:
    image = Image.open(file_uploader)
    st.image(image, caption="Ảnh được chọn")


if st.button('Predict khi chưa sử dụng phân cụm'):

    if file_uploader is None:
        st.write("Chưa tải ảnh lên")
    else:
        image_path = file_uploader
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        scale_img = np.expand_dims(image_array, axis=0)

        model2 = load_model("oral cancer detection.h5")
        pred = model2.predict(scale_img)
        
        import time
        my_bar = st.progress(0)
        with st.spinner("Predicting"):
            time.sleep(2)
        
        output = class_names[np.argmax(pred)]
        st.title(f"Tình trạng: {output} với tỷ lệ {100 * np.round(pred[0][np.argmax(pred)], 2)}")

        
if st.button('Predict khi sử dụng phân cụm'):

    if file_uploader is None:
        st.write("Chưa tải ảnh lên")
    else:
        image_path = file_uploader
        
        ## Xử ly phân cụm
        image = tf.keras.preprocessing.image.load_img(image_path, target_size = (50, 50))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        scale_img = np.expand_dims(image_array, axis=0)

        model2 = load_model("model.h5")
        pred = model2.predict(scale_img)
        output = class_names[np.argmax(pred)]
        ## Kết thúc xử lý

        import time
        my_bar = st.progress(0)
        with st.spinner("Predicting"):
            time.sleep(2)
        
        st.title(f"Tình trạng: {output} với tỷ lệ {100 * np.round(pred[0][np.argmax(pred)], 2)}")
