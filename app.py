import streamlit as st
import requests
from glob import glob
import os
from utils import does_zip_have_nifti, store_data, get_random_string, centered_rounded_image, deleteTempData
from preprocess import preprocess
from model import predict
import io
# import vtk
from ipywidgets import embed
import streamlit.components.v1 as components
from itkwidgets import view

import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from streamlit_option_menu import option_menu



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

def display_slice(data, index):
    """Displays a single 2D slice of the 3D MRI scan image.

    Args:
        data (np.ndarray): The 3D MRI scan image data.
        index (int): The index of the slice to display (0-based).
    """

    slice = data[:, :, index]  # Extract the desired slice
    st.image(slice, caption=f"Slice {index+1} of 128")  # Display with informative caption

global temp_data_directory
temp_data_directory = ''

lottie_file = load_lottieurl('https://assets10.lottiefiles.com/private_files/lf30_4FGi6N.json')
data_key = 'has_data'
data_has_changed = False
 
st.set_page_config(page_title='NeuroWhiz', page_icon='🧠', layout='wide')
# local_css("style/style.css")
# Custom CSS styles
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #1a535c;
        text-align: center;
        padding-top: 50px;
    }
    .description {
        font-size: 20px;
        color: #4ecdc4;
        text-align: center;
        padding-bottom: 30px;
    }
    .file-uploader {
        text-align: center;
        padding-top: 50px;
        padding-bottom: 50px;
    }
    .footer {
        font-size: 14px;
        color: #ffffff;
        text-align: center;
        padding-top: 30px;
        padding-bottom: 20px;
    }
    
        .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 50px;
        background-color: #1a535c;
        color: #ffffff;
        font-size: 20px;
        font-weight: bold;
    }
    .navbar-brand {
        text-decoration: none;
        color: #ffffff;
    }
    .navbar-links {
        list-style-type: none;
        display: flex;
        padding: 0;
    }
    .navbar-link {
        margin-left: 20px;
        text-decoration: none;
        color: #ffffff;
    }
    .navbar-link:hover {
        text-decoration: underline;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Navbar as a sidebar
with st.sidebar:
    selected = option_menu(
        menu_title= '🧠NeuroWhiz',
        menu_icon= None,
        options= ['🏠 Home', '📚 How To Use', '🛠️ Tool'],
        default_index= 0, 
    )
    
if selected == '🏠 Home':
    # Custom CSS styles for the homepage
    st.markdown(
        """
        <style>
        .header {
            padding: 20px;
            background-color: #1a535c;
            color: #ffffff;
            text-align: center;
        }
        .content {
            padding: 20px;
        }
        .info {
            font-size: 18px;
            margin-top: 20px;
        }
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4ecdc4;
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #2b7a78;
        }
        .image {
            width: 100%;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown("<h1>Brain Tumor Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    #center image
    
    #st.image("images/brain50.jpg", use_column_width=False)

    # Main content
    st.markdown("<div class='content'>", unsafe_allow_html=True)
    st.write("Welcome to our Brain Tumor Segmentation application!")
    st.write("Brain tumor segmentation is a crucial task in medical image analysis, as it allows clinicians to identify and localize tumors within brain MRI scans.")
    st.write("Our application utilizes state-of-the-art deep learning techniques to automatically segment brain tumors from MRI images.")
    st.write("Below are some example images of brain tumors and their segmentations:")
    # Placeholder images
    
    st.image("images/Home.jpeg", caption="MRI Image with Tumor", use_column_width=True)
    

    # Info
    st.markdown("<div class='info'>", unsafe_allow_html=True)
    st.write("For more information about brain tumor segmentation and our research project, click the button below.")
    if st.button("Learn More", key="learn_more_button"):
        # Add code to navigate to the about page
        pass
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    
elif selected == '📚 How To Use':
    st.write('Instructions on how to use the tool')
    
elif selected == '🛠️ Tool':
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
                flair = input[0][:,:,:,0]
                t1ce = input[0][:,:,:,1]
                t2 = input[0][:,:,:,2]
                t1 = input[0][:,:,:,3]

                # Send to model and get prediction
                prediction_seg, prediction_edge, original_prediction_seg = predict(input)
                print('Original Prediction Seg:', original_prediction_seg.shape)

            deleteTempData()

            if prediction_seg is not None and prediction_edge is not None:
                    
                st.success('Prediction done.')
                
                # Visualize output image

                st.write('---')
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h5 style='text-align: center;'>T1</h5>", unsafe_allow_html=True)
                    slice_index_1 = st.slider("Select Slice", min_value=0, max_value=127, value=0, key="t1")
                    display_slice(t1, slice_index_1)
                with col2:
                    st.markdown("<h5 style='text-align: center;'>T2</h5>", unsafe_allow_html=True)
                    slice_index_2 = st.slider("Select Slice", min_value=0, max_value=127, value=0, key="t2")
                    display_slice(t2, slice_index_2)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h5 style='text-align: center;'>T1CE</h5>", unsafe_allow_html=True)
                    slice_index_3 = st.slider("Select Slice", min_value=0, max_value=127, value=0, key="t1ce")
                    display_slice(t1ce, slice_index_3)
                with col2:
                    st.markdown("<h5 style='text-align: center;'>FLAIR</h5>", unsafe_allow_html=True)
                    slice_index_4 = st.slider("Select Slice", min_value=0, max_value=127, value=0, key="flair")
                    display_slice(flair, slice_index_4)
                col1, col2 = st.columns(2)
                with col1:
                    st.write('##### Segmentation Result')
                    slice_index_5 = st.slider("Select Slice", min_value=0, max_value=127, value=0, key="seg")
                    # display_slice(original_prediction_seg[0,:,:,:,1], slice_index_5)

                # Download results

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



            
            
st.markdown("<p class='footer'>Developed with ❤️ by Team NeuroWhiz </p>", unsafe_allow_html=True)
