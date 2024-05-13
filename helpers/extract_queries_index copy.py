import os
import random
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter

import warnings
warnings.filterwarnings('error')
Image.MAX_IMAGE_PIXELS = 1000000000 


min_scale=0.25
max_scale=0.5

# def Customtransform(img, idx, p_path):

#     k = random.uniform(2.5,5) #[25%-50%]
#     crop_size = int(min(int(img.size[1]/k), int(img.size[0]/k)))
#     center_crop = T.RandomCrop(size=crop_size)
#     cc = center_crop(img)
#     augmenter = T.RandAugment()
#     patch = augmenter(cc)

#     path = p_path+'/query_{}.png'.format(idx)
#     patch.save(path) 
#     return path

def random_aspect_ratio():
    """Generate a random aspect ratio between 4:3 and 16:9, common camera aspect ratios."""
    common_ratios = [(4, 3), (16, 9), (3, 2)]  # Add more as needed
    return random.choice(common_ratios)
def apply_random_rotation(image, max_angle=5):
    """Apply a slight random rotation to the image."""
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, expand=True, fillcolor='white')  # Fill with white or black
def add_gaussian_noise(image, mean=0, std=0.1):
    """
    Add Gaussian noise to an image.
    """
    np_image = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 1) * 255
    return Image.fromarray(noisy_image.astype(np.uint8))
def randomly_alter_pixels(image, alteration_chance=0.01):
    """
    Randomly alter some pixels in the image.
    """
    np_image = np.array(image)
    mask = np.random.rand(*np_image.shape[:2]) < alteration_chance
    random_pixels = np.random.randint(0, 256, np_image.shape)
    np_image[mask] = random_pixels[mask]
    return Image.fromarray(np_image)
def add_colored_shadow_effect(image, opacity=128, blur_radius=10, offset=(0,0)):
    # Ensure image is in RGBA mode to handle transparency
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Generate a random color for the shadow to simulate different light sources
    shadow_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), opacity)

    # Create shadow layer
    shadow = Image.new('RGBA', image.size, (0,0,0,0))
    shadow_draw = ImageDraw.Draw(shadow)

    # Calculate shadow dimensions
    shadow_dims = [offset[0], offset[1], image.size[0] + offset[0], image.size[1] + offset[1]]
    
    # Draw an ellipse (or your desired shape) for the shadow on the shadow layer
    shadow_draw.ellipse(shadow_dims, fill=shadow_color)

    # Blur the shadow to create a soft effect
    shadow_blurred = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Composite the blurred shadow onto the original image
    result_image = Image.alpha_composite(shadow_blurred, image)

    # If the original image was not in RGBA, convert back to the original mode
    if image.mode != 'RGBA':
        result_image = result_image.convert(image.mode)
    
    return result_image
def Customtransform(image, idx, p_path):
    """
    Perform a random "camera-esque" crop on the given image.
    """

    randaugment = T.RandomApply([
        T.RandomRotation(5),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        T.RandomPerspective(distortion_scale=0.05, p=0.5),
        # T.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    ], p=0.5)

    image = randaugment(image)
    image = add_gaussian_noise(image)
    image = randomly_alter_pixels(image)
    image = add_colored_shadow_effect(image)
        
    aspect_ratio = random_aspect_ratio()
    width, height = image.size
    base_scale = random.uniform(min_scale, max_scale)

    # Calculate new dimensions based on the aspect ratio
    if random.choice([True, False]):
        new_width = int(width * base_scale)
        new_height = int(new_width / aspect_ratio[0] * aspect_ratio[1])
    else:
        new_height = int(height * base_scale)
        new_width = int(new_height / aspect_ratio[1] * aspect_ratio[0])
    if new_width > width or new_height > height:
        # If calculated crop size is bigger than the image, adjust it.
        new_width = min(new_width, width)
        new_height = min(new_height, height)

    x = random.randint(0, width - new_width)
    y = random.randint(0, height - new_height)
    crop = image.crop((x, y, x + new_width, y + new_height))
    query = apply_random_rotation(crop)
    
    path = p_path+'/query_{}.png'.format(idx)
    query.save(path) 
    return path


def generate_queries(doc_images_path, p_path):
    # all_items = os.listdir(doc_images_path)
    # doc_items = [item for item in all_items if os.path.isfile(os.path.join(doc_images_path, item)) and item.lower().endswith('.png')]
    # docs = [os.path.join(doc_images_path, file_name) for file_name in doc_items]

    doc_labels_df = pd.read_csv('./doc_labels.csv')

    start = datetime.datetime.now()
    df = pd.DataFrame(columns=['PatchPath', 'doc_name', 'doc_label'])
    print("images found:", len(doc_labels_df))
    for idx, row in doc_labels_df.iterrows():
        try:
            filepath = os.path.join(doc_images_path, row.values[0])
            img = Image.open(filepath)
            patch_path = Customtransform(img, idx, p_path)
            doc = str(row.values[0])
            label = int(row.values[1])
            # print(filepath, doc_items[idx], int(doc), int(page))
            df = df.append({'PatchPath':patch_path, 'doc_name':doc ,'doc_label':label}, ignore_index=True)
            if idx%100 == 0:
                print("sample-pairs extracted:",df.shape[0],":=:" ,np.round(idx/len(df)*100, 2),"%. done")
        except Exception as e:
            print(e)
    df.to_csv(p_path+'/query_labels.csv',index=False)
    end = datetime.datetime.now()
    print("#queries extracted:",len(df))
    print("time taken:", end-start)

docs_path = '/home/csgrad/indratej/my_projects/thesis/data/pdfs/imgs/'
queries_path = '/home/csgrad/indratej/my_projects/thesis/data/pdfs/queries'

if os.path.exists(queries_path) is False:
    os.mkdir(queries_path)
    print(("created directory at {}".format(queries_path)))

generate_queries(docs_path, queries_path)