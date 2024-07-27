import glob
import random
import os
from pandas.core.common import flatten
from torchvision import transforms
from PIL import Image, ImageFilter
from .symbol_dataset import SymbolDataset
import random
import numpy as np
import cv2
random.seed(1)
TRAIN_DATA_PATH = "data/train"
TEST_DATA_PATH = "data/test"


class RandomPaddingTransform:
    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, img):
        top = random.randint(0, 100)
        bottom = random.randint(0, 100)
        left = random.randint(0, 100)
        right = random.randint(0, 100)
        return transforms.Pad((left, top, right, bottom), fill=self.fill)(img)
    
def sharpen_image(image):
    return image.filter(ImageFilter.SHARPEN)

# def sharpen_image(image):
#     image_np = np.array(image)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     image_sharp = cv2.filter2D(image_np, -1, kernel)
#     return Image.fromarray(image_sharp)

# def adaptive_threshold(image):
#     image_np = np.array(image)
#     # No need to convert to grayscale, the image is already in grayscale
#     image_thresh = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     return Image.fromarray(image_thresh)

# def crop_center(image, crop_size):
#     width, height = image.size
#     left = (width - crop_size) / 2
#     top = (height - crop_size) / 2
#     right = (width + crop_size) / 2
#     bottom = (height + crop_size) / 2
#     return image.crop((left, top, right, bottom))

# def resize_with_padding(image, target_size):
#     width, height = image.size
#     ratio = min(target_size / width, target_size / height)
#     new_size = (int(width * ratio), int(height * ratio))
#     image = image.resize(new_size, Image.LANCZOS)
#     new_image = Image.new("L", (target_size, target_size), 255)
#     new_image.paste(image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))
#     return new_image

# class AddWhiteBackgroundAndResize:
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, image):
#         # Ensure image is in grayscale mode
#         if image.mode != 'L':
#             image = image.convert('L')

#         # Create a new white background image of the target size
#         # 'L' mode means grayscale, where 255 is white
#         background = Image.new('L', self.size, 255)

#         # Calculate the position to center the original image on the new background
#         orig_width, orig_height = image.size
#         target_width, target_height = self.size
#         start_x = (target_width - orig_width) // 2
#         start_y = (target_height - orig_height) // 2

#         # Paste the original image onto the white background
#         background.paste(image, (start_x, start_y))

#         return background



TRAIN_TRANSFORMS = transforms.Compose([
    #transforms.Resize((128, 128), interpolation=Image.LANCZOS),
    #AddWhiteBackgroundAndResize((128, 128)),
    RandomPaddingTransform(fill=255),
    #transforms.Lambda(lambda img: resize_with_padding(img, 45)),
    #transforms.RandomResizedCrop(45, scale=(0.8, 1.0)),  # Random resized crop
    #transforms.Lambda(lambda img: adaptive_threshold(img)),
    transforms.Resize((128, 128), interpolation=Image.LANCZOS),
    #transforms.Lambda(lambda img: resize_with_padding(img, 45)),
    
    transforms.Lambda(lambda x: sharpen_image(x)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]
    )

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.LANCZOS),
    #AddWhiteBackgroundAndResize((128, 128)),
    #RandomPaddingTransform(fill=255),
    #transforms.Lambda(lambda img: crop_center(img, min(img.size))),
    #transforms.Lambda(lambda img: adaptive_threshold(img)),
    #transforms.Resize((45, 45), interpolation=Image.LANCZOS),
    #transforms.Lambda(lambda img: resize_with_padding(img, 45)),
    transforms.Lambda(lambda x: sharpen_image(x)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]
    )

def get_data_paths(data_path):
    image_paths = []
    classes = []
    for path in glob.glob(data_path + '/*'):
        class_name = os.path.basename(path)
        if class_name not in classes:
            classes.append(class_name)
        image_paths.append(glob.glob(path + '/*'))
    image_paths = list(flatten(image_paths))
    random.shuffle(image_paths)
    return image_paths, classes

train_image_paths, classes = get_data_paths(TRAIN_DATA_PATH)
test_image_paths, _ = get_data_paths(TEST_DATA_PATH)

train_dataset = SymbolDataset(image_paths=train_image_paths, classes=classes, transform=TRAIN_TRANSFORMS)
test_dataset = SymbolDataset(image_paths=test_image_paths, classes=classes, transform=EVAL_TRANSFORMS)
