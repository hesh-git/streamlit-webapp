import numpy as np
import tensorflow as tf

# from process import get_last_layer, get_last_conv_layer

from XAI.segmentation_model import get_deepseg, load_images, get_xai_segmentation_model, two_decoder_unet_model
from XAI.explain import get_neuroxai, get_neuroxai_cnn


def generate_xai(mri_vol,s_model):

    # NeuroXAI parameters
    DIMENSION = "3d"
    MODALITY = "FLAIR"
    XAI_MODE = "segmentation"
    CLASS_IDs = [1,2,3]
    TUMOR_LABEL = "all" # for GCAM visualization
    #LAYER_NAME = '1'
    #LAYER_NAME = 'conv3d_50' #conv3d_10
    XAI="GCAM"

    layers = ['conv3d','conv3d_1', 'conv3d_2', 'conv3d_3', 'conv3d_4', 'conv3d_5','conv3d_6','conv3d_7','conv3d_8','conv3d_9']
    for i in layers:
        IMG_SHAPE = (128, 128, 128)

        # get the segmentation model
        # s_model = two_decoder_unet_model(128,128,128,4,4,WEIGHTS= "weights/dual_decoder_simsiam_3d_unet_weights.hdf5")
        LAYER_NAME = i
        model = get_xai_segmentation_model(s_model, LAYER_NAME)
        
        # Sample MRI case
        ID = "IMG_10"
        SLICE_ID = 77
        CLASS_ID = 2 #np.argmax(predictions[0])
        TUMOR_LABEL="all" # for grad-CAM
        io_imgs = mri_vol
        io_imgs = np.expand_dims(io_imgs, axis=0)

        im_orig = io_imgs[:,:,:,SLICE_ID,0] # 2D FLAIR


        get_neuroxai_cnn(ID, model, io_imgs, CLASS_ID, SLICE_ID,LAYER_NAME, MODALITY, 
                        XAI_MODE, XAI, DIMENSION, CLASS_IDs, TUMOR_LABEL, 
                        SAVE_RESULTS=True, SAVE_PATH="OurXAI_Seg_GCAM")
