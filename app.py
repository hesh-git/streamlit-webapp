import streamlit as st
import requests
from glob import glob
import os
from utils import does_zip_have_nifti, store_data, get_random_string
from preprocess import preprocess

import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

global temp_data_directory
temp_data_directory = ''

lottie_file = load_lottieurl('https://assets10.lottiefiles.com/private_files/lf30_4FGi6N.json')
data_key = 'has_data'
data_has_changed = False
 
st.set_page_config(page_title='3D Visualization', page_icon=':pill:', layout='wide')
# local_css("style/style.css")
st.title("TumorScope3D")
st.subheader("Automatic Brain Tumor Segmentation Tool")

# st_lottie(lottie_file, height=1000, key='coding')

input_path = st.file_uploader('Upload files')
data_path = ""

# Upload section
with st.container():
    st.write('---')
    if input_path:
        if does_zip_have_nifti(input_path):
            data_path = get_random_string(15)
            temp_data_directory = f'./data/{data_path}/'
            os.makedirs(temp_data_directory, exist_ok=True)
            store_data(input_path, temp_data_directory)
            data_has_changed = True

            # Preprocess section
            scaler = MinMaxScaler()

            flair_path = f'data/{data_path}/BraTS19_2013_2_1_t2.nii'
            t1_path = f'data/{data_path}/BraTS19_2013_2_1_t1.nii'
            t2_path = f'data/{data_path}/BraTS19_2013_2_1_t2.nii'
            t1ce_path = f'data/{data_path}/BraTS19_2013_2_1_t1ce.nii'

            temp_image_t2=nib.load(t2_path).get_fdata()
            temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

            temp_image_t1ce=nib.load(t1ce_path).get_fdata()
            temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

            temp_image_flair=nib.load(flair_path).get_fdata()
            temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
            temp_image_t1 = nib.load(t1_path).get_fdata()
            temp_image_t1=scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)

            temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2,temp_image_t1], axis=3)
            #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
            #cropping x, y, and z
            test_img=temp_combined_images[56:184, 56:184, 13:141]

            # test_img_input = np.expand_dims(test_img, axis=0)
            print(test_img.shape)

            # Visualize a random slice of test_img in browser. Need to display 2 in 1 row and other two in next row
            st.write('---')
            st.write('### Random slice of the MRI scan')
            st.write('---')

            col1, col2 = st.columns(2)

            col1.image(test_img[:, :, 64, 0], width=500)
            col2.image(test_img[:, :, 64, 1], width=500)

            col3, col4 = st.columns(2)

            col3.image(test_img[:, :, 64, 2], width=500)
            col4.image(test_img[:, :, 64, 3], width=500)


