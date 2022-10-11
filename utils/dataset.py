import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform
from typing import Dict, Union
import torch
from utils.paths import DATA
from PIL import Image

class AIROGSLiteDataset(Dataset):

    def __init__(self, args, transform: transform = None) -> None:
        self.data_dir = args.data_dir
        self.labels = np.asarray(pd.read_csv(self.data_dir + "/dev_labels2.csv"))
        self.transform = transform
        self.data_name = args.data_name

    def __len__(self) -> int:
        return len(self.labels)
    
    def shuffle(self):
        np.random.shuffle(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        img_name, label = self.labels[idx]
        img_path = self.data_dir + "/" + self.data_name + "/" + img_name + ".jpg"
        #img_path = self.data_dir + "/cropped/" + img_name + ".jpg"
        image = io.imread(img_path)
        image = np.array(image) / 255
        assert (label=="NRG") or (label=="RG")

        if label == 'NRG':  
            label = [1, 0]
        else: 
            label = [0, 1]
        label = np.array(label)
        img_path = np.array(img_path)
        sample = {
            "image": image, 
            "label": label,
            "path": img_path
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample: Dict) -> Dict:
        image = sample['image']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        sample['image'] = img

        return sample

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors.
        Drops the image path because it's a string
        and doesn't really matter. 
    """

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
