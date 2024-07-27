import glob
import random
import os
from pandas.core.common import flatten
from torchvision import transforms
from .symbol_dataset import SymbolDataset

TRAIN_DATA_PATH = "data/train"
TEST_DATA_PATH = "data/test"
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

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

train_dataset = SymbolDataset(image_paths=train_image_paths, classes=classes, transform=IMAGE_TRANSFORMS)
test_dataset = SymbolDataset(image_paths=test_image_paths, classes=classes, transform=IMAGE_TRANSFORMS)
