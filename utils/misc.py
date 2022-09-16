from typing import Sized
import torch
import matplotlib.pyplot as plt

from utils.dataset import AIROGSLiteDataset, Rescale

def show_head(dataset: AIROGSLiteDataset) -> None:
    ''' Show first 4 images of the dataset, for debug purposes. '''
    fig = plt.figure()
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample['image'].shape, sample['label'])
        image = sample['image']
        image = image.numpy.transpose((1, 2, 0)) # back to w x h x c
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('{}'.format(sample['label']))
        ax.axis('off')
        plt.imshow(image)
        if i == 3:
            plt.show()
            break
