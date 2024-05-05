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
    
    # Apply custom CSS styles for the dark theme and improved design
    st.markdown(
        """
        <style>
        /* Set the dark mode theme for the application */
        body {
            background-color: #121212;
            color: #ffffff;
            font-family: 'Segoe UI', 'Helvetica', 'Arial', sans-serif;
        }
        /* Header styles */
        .header {
            padding: 20px;
            background-color: #1a535c;
            color: #ffffff;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Logo styles */
        .logo {
            display: block;
            margin: 0 auto;
            width: 150px;
            border-radius: 50%;
        }
        /* Content styles */
        .content {

            background-color: #333;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        /* Styles for headings and text */
        h1, h2 {
            color: #4ecdc4;
        }
        /* Button styles */
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4ecdc4;
            color: #121212;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s, color 0.3s;
        }
        .button:hover {
            background-color: #2b7a78;
            color: #ffffff;
        }
        /* Image styles */
        .image {
            width: 100%;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header with logo and title

    # st.image("images/neurowhiz-high-resolution-logo-transparent.png", caption="", use_column_width=True)
    logo_path = 'images/neurowhiz-high-resolution-logo-transparent-resized.png'  # Update the path to your logo image
    st.image(logo_path, use_column_width="never", width=400, output_format="auto")
    # st.markdown("<img class='logo' src='images/neurowhiz_logo.png' alt='Neurowhiz Logo' />", unsafe_allow_html=True)
    st.markdown("<h1>NeuroWhiz: Accurate Brain Tumor Segmentation with Edge Detection</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Main content
    st.markdown("<div class='content'>", unsafe_allow_html=True)
    st.write("Welcome to NeuroWhiz, our Brain Tumor Segmentation and Edge Detection application!")
    st.write("""
            Are you tired of the manual diagnosis of brain tumor subregions and boundries through MRIs? Introducing NeuroWhiz - a state-of-the-art tool that automates the brain tumor segmentation and edge detection. Our model, not only detects tumor subregions but also the boudries of tumor subregions that help to radiologists and neurosurgions to make their diagnosis and decision making more confidenly and accurately.
With NeuroWhiz, you'll benefit from the latest explainability techniques that provide insight into the decision-making process of our AI model. Our NeuroWhiz application takes 3D multi modal MRI image as an input and segment the brain tumor with suregions while detecting the tumor boundries too. Our user-friendly interface ensures easy use and saving you time. Trust NeuroWhiz to provide you with faster, more accurate diagnoses of brain tumors a crucial role in clinical care.
            
            """)

    st.write("<br>", unsafe_allow_html=True)
    

    st.image("images/overviewFinal.png", caption="", width=600, use_column_width=True)


    st.markdown("<h2>The Power of NeuroWhiz</h2>", unsafe_allow_html=True)
    st.write("""
            Analyzing Brain MRI has never been easier. Our state-of-the-art framework takes in an multi modal MRI and segment the brain tumor and detect the edges.
But NeuroWhiz isn't just about convenience - it's also about accuracy. Our deep learning model leverages the power of self supervised learning and uses
a unique dual-decoder architecture, focusing on edge identification and segmentation accuracy
enhancement. Utilizing a dual-decoder 3D-Unet model, we prioritize accuracy and fine-grained
details in tumour segmentation and introduce an additional tumour edge detection task as well
to the model, aiming to move beyond traditional single-decoder approaches. Below are the some results that got from NeuroWhiz.

             """)
    
    st.write("<br>", unsafe_allow_html=True)
    

    st.image("images/Home2.jpeg", caption="", width=600, use_column_width=True)

    st.write("""
            In the above visualization column (a) to (d) are the 3D MRI image inputs of 4 modalities. (e) represents
original ground truth segmentation masks. (f) is the predictions of NeuroWhiz.

             """)

    st.markdown("<h2>NeuroWhiz Incorparate Explainability too..</h2>", unsafe_allow_html=True)

    st.write("""
           Incorporating explainability into AI systems for the medical domain enhances trust, promotes ethical practices, facilitates knowledge generation, improves error detection, empowers patients, and ensures regulatory compliance.
             """)
    
    st.write("<br>", unsafe_allow_html=True)
    
    st.image("images/XAIFinal.png", caption="", width=600, use_column_width=True)

    # Info section with button
    st.markdown("<div class='content'>", unsafe_allow_html=True)
    st.write("For more information about brain tumor segmentation and our research project, click the button below.")
    if st.button("Learn More", key="learn_more_button"):
        # Add code to navigate to the about page
        pass
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
