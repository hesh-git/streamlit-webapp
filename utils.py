import zipfile
import os
import streamlit as st
import random
import string
import numpy as np
import shutil

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

def store_data(file, temp_data_directory, temporary_location=temp_zip_file):
    st.warning('Loading data from zip.')

    with open(temporary_location, 'wb') as out:
        out.write(file.getbuffer())
    
    if is_zip_oversized(temporary_location):
        st.warning('Oversized zip file.')
        # clear_data_storage(temporary_location)
        return False

    with zipfile.ZipFile(temporary_location) as zip_ref:        
        zip_ref.extractall(temp_data_directory + '/')
        st.success('The file is uploaded')

    return True

def get_random_string(length):
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    return result_str

def deleteTempData():
    if os.path.exists(data_folder_path):
        shutil.rmtree(data_folder_path)
    return True