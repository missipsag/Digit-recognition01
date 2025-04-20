from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, invert=False, threshold=128):
   
    # 1. charger l'image et la convertir en grayscale
    img = Image.open(image_path).convert('L')
    
    #optionnel : inverser les couleurs des pixels
    if invert:
        img = ImageOps.invert(img)
    
    # 3. convertir l'image à un tableau numpy
    img_array = np.array(img)
    
    # 4. Appliquer un limite maximale.
    #    Les pixels plus sombre deviennent font partie du digit, les reste du background
    binary_array = np.where(img_array < threshold, 0, 255).astype(np.uint8)
    binary_img = Image.fromarray(binary_array)
    
    # 5. Croper l'image
    #    On trouve les pixel dont la valeur est 0 
    coords = np.column_stack(np.where(binary_array == 0))
    if coords.size != 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        cropped_img = binary_img.crop((x_min, y_min, x_max + 1, y_max + 1))
    else:
        # Si aucun digit est trouvé utiliser l'image complètement
        cropped_img = binary_img

    # 6. Centrer le digit dans une image 8x8
    #    En Utilisant ImageOps.pad on ajoute du padding pour center l'image.
    processed_img = ImageOps.pad(cropped_img, (8, 8), color=255, centering=(0.5, 0.5))
    
    # 7. Scaler les valeur des pixels pour matcher le dataset.
    
    processed_array = np.array(processed_img)
   
    img_scaled = (processed_array / 255.0) * 16
    
    # 8. transformer l'image en un vecteur 64
    flat_img = img_scaled.flatten()
    
    # This is for showing the preprocessed image of the digit
    """plt.imshow(img_scaled / 16, cmap='gray', interpolation='nearest')
    plt.title("Processed 8x8 Image")
    plt.show()"""
    
    return flat_img
