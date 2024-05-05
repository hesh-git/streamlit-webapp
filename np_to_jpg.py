import os
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def convert_np_to_color_jpeg(nii_data, output_folder, colormap='viridis'):
    # Check if the file exists
    # if not os.path.exists(nii_path):
    #     print(f"Error: The file '{nii_path}' does not exist.")
    #     return

    # Load the NIfTI file
    # try:
    #     nii_img = nib.load(nii_path, mmap=False)
    #     nii_data = nii_img.get_fdata()
    # except nib.filebasedimages.ImageFileError as e:
    #     # Try loading gzip-compressed NIfTI file
    #     try:
    #         nii_img = nib.load(nii_path, mmap=False, file_class=nib.GzipFile)
    #         nii_data = nii_img.get_fdata()
    #     except Exception as ex:
    #         print(f"Error: Cannot load the NIfTI file '{nii_path}'.")
    #         print(ex)
    #         return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Normalize the data to the range [0, 1]
    data_min, data_max = nii_data.min(), nii_data.max()
    normalized_data = (nii_data - data_min) / (data_max - data_min)

    # Apply the colormap to each slice individually and save as color JPEG
    for z in range(normalized_data.shape[-1]):
        # Get the 2D grayscale slice for the current z-slice
        slice_data = normalized_data[:, :, z]

        # Apply the colormap to the grayscale slice
        colormap_func = cm.get_cmap(colormap)
        colored_data = colormap_func(slice_data)

        # Rescale the data to 0-255 and convert to uint8 (required for JPEG)
        colored_data_rescaled = (colored_data * 255).astype('uint8')

        # Save the color image as JPEG
        output_path = os.path.join(output_folder, f'slice_{z:03d}.jpg')
        plt.imsave(output_path, colored_data_rescaled)

    print(f'Conversion complete. {normalized_data.shape[-1]} slices saved as color JPEG images.')

