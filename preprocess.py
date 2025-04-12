from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image_path): 
    img = Image.open(image_path).convert('L')

    #resise image to 8x8 to match training images

    img = img.resize((8,8), Image.Resampling.LANCZOS)

    img_array = np.array(img)
    # invert colors 
    #img_array = 255 - img_array

    img_scaled = (img_array/255.0) * 16
    plt.imshow(img_scaled, cmap='gray', interpolation='nearest')
    plt.show()

    flat_img = img_scaled.flatten()

    return flat_img