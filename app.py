import streamlit as st
import requests
from glob import glob
import os
from utils import does_zip_have_nifti, store_data, get_random_string
from preprocess import preprocess
from model import predict
import io
import vtk
from ipywidgets import embed
import streamlit.components.v1 as components
from itkwidgets import view

import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


prediction_seg, prediction_edge = None, None

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def download_results():
    with io.BytesIO() as buffer:
        np.save(buffer, prediction_seg)  # Save the array to the buffer
        st.download_button(
            label="Download Segmentation Mask (.npy)",
            data=buffer.getvalue(),
            file_name="neurowhiz_result_seg.npy",
            mime="application/octet-stream"  # Set appropriate MIME type
        )
    with io.BytesIO() as buffer:
        np.save(buffer, prediction_edge)
        st.download_button(
            label="Download Edge Mask (.npy)",
            data=buffer.getvalue(),
            file_name="neurowhiz_result_edge.npy",
            mime="application/octet-stream"  # Set appropriate MIME type
        )


global temp_data_directory
temp_data_directory = ''

lottie_file = load_lottieurl('https://assets10.lottiefiles.com/private_files/lf30_4FGi6N.json')
data_key = 'has_data'
data_has_changed = False
 
st.set_page_config(page_title='NeuroWhiz', page_icon=':pill:', layout='wide')
# local_css("style/style.css")

st.title("NeuroWhiz")
st.subheader("Automatic 3D Brain Tumor Segmentation and Edge Detection Tool")

# st_lottie(lottie_file, height=1000, key='coding')

input_path = st.file_uploader('Upload a single zip folder containing 3D MRI images of all 4 modalities in .nii format')
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

            # # Visualize input image
            # path_to_file = glob('data/*/*.nii')
            # if path_to_file:
            #     with st.container():
            #         reader = vtk.vtkNIFTIImageReader()
            #         reader.SetFileName(path_to_file[0])
            #         reader.Update()

            #         view_width = 1800
            #         view_height = 800

            #         snippet = embed.embed_snippet(views=view(reader.GetOutput()))
            #         html = embed.html_template.format(title="", snippet=snippet)
            #         components.html(html, width=view_width, height=view_height)

            with st.spinner('Please wait...'):
                # Preprocess section
                input = preprocess(temp_data_directory)

                # Send to model and get prediction
                prediction_seg, prediction_edge = predict(input)

            if prediction_seg is not None and prediction_edge is not None:
                st.success('Prediction done.')
                
                # if st.button("Download Results"):
                #     download_results()

                with io.BytesIO() as buffer:
                    np.save(buffer, prediction_seg)  # Save the array to the buffer
                    st.download_button(
                        label="Download Segmentation Mask (.npy)",
                        data=buffer.getvalue(),
                        file_name="neurowhiz_result_seg.npy",
                        mime="application/octet-stream"  # Set appropriate MIME type
                    )
                with io.BytesIO() as buffer:
                    np.save(buffer, prediction_edge)
                    st.download_button(
                        label="Download Edge Mask (.npy)",
                        data=buffer.getvalue(),
                        file_name="neurowhiz_result_edge.npy",
                        mime="application/octet-stream"  # Set appropriate MIME type
                    )


            else:
                st.error('Prediction failed.')
        else:
            st.warning('Zip folder does not have folders with NII files.')
