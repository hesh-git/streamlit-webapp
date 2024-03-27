import numpy as np
from keras.models import load_model

model = load_model('dual-decoder/third/dual_decoder_simsiam_3d_unet.hdf5',
                   custom_objects={
                       'total_loss': total_loss, 
                       'weighted_categorical_crossentropy': weighted_categorical_crossentropy([wt00, wt11, wt22, wt33]), 
                       'loss': losses,
                       'iou_score':sm.metrics.IOUScore(threshold=0.5),
                       'dice_coef':dice_coef
                    }
                   )

def predict(test_img_input):
    # test_prediction = model.predict(test_img_input)
    test_prediction_seg, test_prediction_edg = model.predict(test_img_input)
    test_prediction_argmax=np.argmax(test_prediction_seg, axis=4)[0,:,:,:]
    test_prediction_edge_argmax=np.argmax(test_prediction_edg, axis=4)[0,:,:,:]