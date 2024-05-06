import streamlit as st
import requests
from glob import glob
import os
from utils import does_zip_have_nifti, store_data, get_random_string, centered_rounded_image, deleteTempData, create_seg_image, download_seg_results, display_images_from_folder
from preprocess import preprocess
from model import predict
import io
import time
# import vtk
from ipywidgets import embed
import streamlit.components.v1 as components
from itkwidgets import view

import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from streamlit_option_menu import option_menu
from np_to_jpg import convert_np_to_color_jpeg
from XAI.xai import generate_xai

prediction_seg, prediction_edge = None, None

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def display_slice(data, index):
    """Displays a single 2D slice of the 3D MRI scan image.

    Args:
        data (np.ndarray): The 3D MRI scan image data.
        index (int): The index of the slice to display (0-based).
    """
    
    slice = data[:, :, index]
    col1, col2, col3 = st.columns([3,2,3])  # Adjust width ratios as needed

    # Add content to center column
    with col2:
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
        options= ['🏠 Home', '📚 How To Use', '🛠️ Segmentation Tool'],
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
    st.title('Unleash the Power of Precision: A User Guide for 3D Brain Tumor Segmentation and Edge Detection')

    st.markdown("""
    Welcome! This comprehensive guide empowers you to harness the capabilities of our innovative web application for 3D brain tumor segmentation and edge detection. This tool plays a vital role in medical image analysis, aiding in the localization, characterization, and treatment planning of brain tumors.

    **What is NeuroWhiz tool?**

    """)    

    # Steps to Use the Tool
    st.header('Steps to Use the Tool:')

    st.markdown("""
    1. **Navigate to the 'Tool' Page:** Within the web app, locate the designated section for tumor segmentation and edge detection.
    """)

    st.image('images/step1.gif')
                
    st.markdown("""
    2. **Prepare Your Input Data:**
    * Ensure you have four separate modalities of 3D MRI images in NIfTI (.nii) format (T1, T2, T1ce, and FLAIR).
    * **Create a Single Zip Archive:** Combine all four NIfTI images into a single compressed zip folder for efficient upload.
    * Make sure each of 4 .nii files ends with following strings: '_t1.nii', '_t2.nii', '_t1ce.nii', '_flair.nii'.
    * Sample zip folder inputs can be found [here](https://drive.google.com/drive/folders/1scIsgHcb5ykqyp9uK3XOyGNN-3O6aPW-?usp=drive_link)
    """)

    st.image('images/step2.png', caption='Example of input zip file content')
                
    st.markdown("""
    3. **Upload Your Data:**
    * Locate the designated upload area on the tool page.
    * Click the "Browse Files" button or its equivalent.
    * Select your prepared zip folder containing the four MRI images.
    * Click "Upload" or a similar action button to initiate the process.
    """)

    st.image('images/step3.gif')
                
    st.markdown("""
    4. **Wait for the Results:** The tool will automatically analyze your uploaded data. Be patient, as processing complex medical images might take a few minutes.

    5. **Visualize and Download Results:**

    """)

    # # Visualization (Screenshots or Animated GIF)
    # st.image('output_visualization.png', caption='Example of Segmentation and Edge Detection Results')

    st.markdown("""
    * Once processing is complete, the tool will present you with the segmentation mask and edge mask visualizations.
    * In addition to that, as XAI is integrated, you can also visualize the saliency maps to understand the model's decision-making process. 
    * You'll be able to examine these visualizations directly within the web app to gain insights into the tumor's location and boundaries.
    * Download options might also be available to save the masks for further analysis on your local machine.

    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image('images/step5-1.png', caption='This is the visualization of input MRI modalities', use_column_width=True)
    with col2:
        st.image('images/step5-2.png', caption='This is the visualization of corresponding segmentation and edge mask outputs', use_column_width=True)
        

    # Additional Tips and Considerations
    st.header("""
    **Additional Tips and Considerations:**

    * **Data Quality:** For optimal results, ensure the uploaded MRI images are high quality and free from artifacts (distortions or errors).
    * **Interpretation:** While the tool provides segmentation and edge detection results, consulting with a qualified medical professional for diagnosis and treatment planning remains crucial.
    * **Ongoing Development:** Stay tuned! We're constantly striving to enhance our tool's capabilities and address user feedback.
                
    """)
    
elif selected == '🛠️ Segmentation Tool':
    st.title("Brain Tumor Segmentation and Edge Detection Tool")
    st.divider()
    input_path = st.file_uploader('Upload a single zip folder containing 3D MRI images of all 4 modalities in .nii format')
    data_path = ""

    # Upload section
    with st.container():
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

                

                with st.spinner('Please wait for Segmentation Results...'):
                    # Preprocess section
                    input = preprocess(temp_data_directory)
                    flair = input[0][:,:,:,0]
                    t1ce = input[0][:,:,:,1]
                    t2 = input[0][:,:,:,2]
                    t1 = input[0][:,:,:,3]

                    # Send to model and get prediction
                    prediction_seg, prediction_edge, original_prediction_seg, model = predict(input)
                    print('Original Prediction Seg:', original_prediction_seg.shape)

                deleteTempData()

                if prediction_seg is not None and prediction_edge is not None:
                        
                    # st.toast('Prediction done', icon='✅')
                    st.toast(":green[Prediction done]", icon='✅')
                    
                    # Visualize output image

                    st.subheader('Visualization of Segmentation and Edge Detection Results', divider='blue')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h5 style='text-align: center;'>T1</h5>", unsafe_allow_html=True)
                        slice_index_1 = st.slider("Select Slice", min_value=0, max_value=127, value=77, key="t1")
                        display_slice(t1, slice_index_1)
                    with col2:
                        st.markdown("<h5 style='text-align: center;'>T2</h5>", unsafe_allow_html=True)
                        slice_index_2 = st.slider("Select Slice", min_value=0, max_value=127, value=77, key="t2")
                        display_slice(t2, slice_index_2)
                        
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h5 style='text-align: center;'>T1CE</h5>", unsafe_allow_html=True)
                        slice_index_3 = st.slider("Select Slice", min_value=0, max_value=127, value=77, key="t1ce")
                        display_slice(t1ce, slice_index_3)
                    with col2:
                        st.markdown("<h5 style='text-align: center;'>FLAIR</h5>", unsafe_allow_html=True)
                        slice_index_4 = st.slider("Select Slice", min_value=0, max_value=127, value=77, key="flair")
                        display_slice(flair, slice_index_4)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write('##### Segmentation Result')
                        slice_index_5 = st.slider("Select Slice", min_value=0, max_value=127, value=77, key="seg")
                        seg_image = create_seg_image(original_prediction_seg)
                        print('Seg Image:', seg_image.shape)
                        # display_result(seg_image, slice_index_5)

                        # Save a slice as a jpg image
                        plt.imsave('seg_image.jpg', seg_image[:,:,slice_index_5])
                        
                        col7, col8, col9 = st.columns([2,3,2]) 
                        # Show the image in browser
                        with col8:
                            st.image('seg_image.jpg', width=130)
                            download_seg_results('seg_images', 'neurowhiz_segment_results.zip', 'seg_dwn_btn')
                        # Delete image from storage
                        os.remove('seg_image.jpg')

                    with col2:
                        st.write('##### Edge Detection Result')
                        slice_index_6 = st.slider("Select Slice", min_value=0, max_value=127, value=77, key="edge")
                        print("==================================")
                        print('Edge Image:', prediction_edge.shape)
                        # edge_image = create_seg_image(prediction_edge)
                        # display_result(edge_image, slice_index_6)

                        # Save a slice as a jpg image
                        plt.imsave('edge_image.jpg', prediction_edge[:,:,slice_index_6])
                        # Show the image in browser
                        col7, col8, col9 = st.columns([2,3,2]) 
                        with col8:
                            st.image('edge_image.jpg', width=130)
                            download_seg_results('edge_images', 'neurowhiz_edge_results.zip', 'edg_dwn_btn')
                        # Delete image from storage
                        os.remove('edge_image.jpg')
                        
                    
                    # Convert numpy predictions to jpeg images
                    convert_np_to_color_jpeg(seg_image, 'seg_images')
                    convert_np_to_color_jpeg(prediction_edge, 'edge_images')
                        
                    #XAI Visualization
                    time.sleep(2)
                    st.toast(':violet[Generating XAI Visualizations]', icon='⏳')
                    heatmaps = []
                    with st.spinner('Please wait for XAI Results...'):
                        heatmaps = generate_xai(input[0], model)
                        st.toast(':green[XAI Visualizations Generated]', icon='✅')
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.subheader('XAI Visualization')
                        display_images_from_folder(folder_path="XAI_Results/XAI") 
                else:
                    st.error('Prediction failed.')
            else:
                st.warning('Zip folder does not have folders with NII files.')    
            
st.markdown("<p class='footer'>Developed with ❤️ by Team NeuroWhiz </p>", unsafe_allow_html=True)
