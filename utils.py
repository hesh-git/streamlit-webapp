import zipfile
import os
import streamlit as st
import random
import string
import numpy as np
import shutil
import io
import time
MAX_SIZE = 300000000 # 300MB

temp_zip_folder = './temp/'
data_folder_path = './data/'
temp_zip_file = temp_zip_folder + 'data.zip'

if not os.path.isdir('./temp'):
    os.makedirs('./temp/')

def does_zip_have_nifti(file):

    with zipfile.ZipFile(file) as zip_ref:
        name_list = zip_ref.namelist()
        for item in name_list:
            if item[-4:] == '.nii':
                return True
    st.warning('Zip folder does not have folders with DICOM files.')
    return False

def is_zip_oversized(path, max_size=MAX_SIZE):
    if os.path.getsize(path) > max_size:
        return True
    return False

    with st.status("Downloading data...", expanded=True) as status:
                    st.write("Searching for data...")
                    time.sleep(2)
                    st.write("Found URL.")
                    time.sleep(1)
                    st.write("Downloading data...")
                    time.sleep(1)
                    status.update(label="Download complete!", state="complete", expanded=False)

def store_data(file, temp_data_directory, temporary_location=temp_zip_file):
    st.toast(':violet[Loading data from zip]', icon='⏳')
    time.sleep(4)
    with open(temporary_location, 'wb') as out:
        out.write(file.getbuffer())
    
    if is_zip_oversized(temporary_location):
        st.warning('Oversized zip file.')
        # clear_data_storage(temporary_location)
        return False

    with zipfile.ZipFile(temporary_location) as zip_ref:        
        zip_ref.extractall(temp_data_directory + '/')
        st.success('MRI Data loaded successfully ✅')

    return True

def get_random_string(length):
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    return result_str

def deleteTempData():
    if os.path.exists(data_folder_path):
        shutil.rmtree(data_folder_path)
    return True

def centered_rounded_image(image_path, width, round_radius):
    st.markdown(
        f"""
        <style>
        .centered-image {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .rounded-image {{
            border-radius: {round_radius}px;
            overflow: hidden;
        }}
        </style>
        <div class="centered-image rounded-image">
            <img src="{image_path}" width="{width}" style="display:block;">
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_seg_image(prediction_seg):
    test_prediction_argmax=np.argmax(prediction_seg, axis=4)[0,:,:,:]
    return test_prediction_argmax


def download_seg_results(original_folder_path, file_name, key):
  # Create a BytesIO object to store the zip file in memory
  with io.BytesIO() as zip_buffer:
    # Create a ZipFile object using the BytesIO object
    with zipfile.ZipFile(zip_buffer, mode="w") as zip_file:
      # Iterate through all files in the original folder
      for root, _, files in os.walk(original_folder_path):
        for filename in files:
          # Get the full path of the file
          file_path = os.path.join(root, filename)

          # Add the file to the zip archive with arcname preserving folder structure
          zip_file.write(file_path, arcname=os.path.relpath(file_path, original_folder_path))

    # Set the download button content as the zip file in memory
    st.download_button(
        label="Download Results",
        data=zip_buffer.getvalue(),
        file_name=file_name,
        mime="application/zip",
        key=key
    )


def display_images_from_folder(folder_path):
    # List all files in the folder
    image_names = os.listdir(folder_path)

    # Filter out only image files (if needed)
    image_names = [filename for filename in image_names if filename.endswith((".png", ".jpg", ".jpeg"))]

    # Display images with separators
    cols = st.columns(5)
    for i, image_name in enumerate(image_names):
        cols[i].image(os.path.join(folder_path, image_name), caption=f"Encoder Layer {i+1}", width=150)
    
    