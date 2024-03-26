import streamlit as st
import nibabel as nib

def main():
  """Function to upload and display a random slice of 3D MRI image"""
  uploaded_file = st.file_uploader("Choose a 3D MRI Image (.nii)", type=["nii", "nii.gz"])
  if uploaded_file is not None:
    try:
      # Read the uploaded file as bytes
      file_bytes = uploaded_file.read()

      # Load the MRI image from bytes using nibabel
      img = nib.load(file_bytes)  # Directly load from bytes
      img_data = img.get_fdata()

      # Get random slice index
      num_slices = img_data.shape[2]
      random_slice = random.randint(0, num_slices-1)

      # Select and display random slice
      slice_data = img_data[:, :, random_slice]
      st.image(slice_data, caption=f"Random slice {random_slice+1} of MRI image", use_column_width=True)
      st.success("Image uploaded and random slice displayed successfully!")
    except Exception as e:
      st.error(f"Error: {e}")
  else:
    st.info("Upload a 3D MRI image (.nii or .nii.gz) to display a random slice.")

if __name__ == "__main__":
  import random  # Import random for generating random slice
  main()
