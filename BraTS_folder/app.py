from flask import Flask, render_template, request, redirect, url_for
import cv2
import nibabel as nib
import tensorflow as tf
import keras
from metrics import dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from detection import is_tumor_detected
#import wand.image

IMG_SIZE=128
VOLUME_SLICES = 100 
VOLUME_START_AT = 22
start_slice= 25

app = Flask(__name__)


model = keras.models.load_model('modelperclasseval\model_per_class.h5', 
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                   "sensitivity":sensitivity,
                                                   "specificity":specificity,
                                                   "dice_coef_necrotic": dice_coef_necrotic,
                                                   "dice_coef_edema": dice_coef_edema,
                                                   "dice_coef_enhancing": dice_coef_enhancing
                                                  }, compile=False)


@app.route('/')
def index():
    return render_template('input.html')

@app.route('/', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        
        # Preprocessing of the image
        img = nib.load(filename)
        img_data = img.get_fdata()
        X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
        for j in range(VOLUME_SLICES):
                X[j,:,:,0] = cv2.resize(img_data[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))

        p = model.predict(X/np.max(X), verbose=1)
        plt.figure(figsize=(18, 50))
        f, axarr = plt.subplots(1,1, figsize = (18, 50))
        axarr.imshow(cv2.resize(img_data[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
        axarr.imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)

        f.subplots_adjust(left=0, right=1, bottom=0, top=1)
        f.savefig('subplot.png',dpi = 300, bbox_inches='tight', pad_inches=0)
        segmented_image = cv2.imread('subplot.png')
        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        #sharpened = cv2.filter2D(segmented_image, -1, kernel)

# Apply thresholding
        tumor_detected = is_tumor_detected(segmented_image)

        if tumor_detected:
            result_desc = "Tumour detected" 
        else:
             result_desc = "Tumour not detected"
        with open('subplot.png', 'rb') as f:
            img = Image.open(f)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            img_data = base64.b64encode(buffer.getvalue()).decode()
        return render_template('output.html', segmented_img= img_data, text_message=result_desc)


    

if __name__ == '__main__':
    app.run(debug=True)




