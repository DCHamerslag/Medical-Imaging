
from utils.dataset import AIROGSLiteDataset, Rescale, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms, utils

def main():
    trsfm = transforms.Compose([
        Rescale((300, 300)),
        ToTensor()
    ])
    dataset = AIROGSLiteDataset(transform=trsfm)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        print("loaded batch")
        print(batch['image'].shape)
        print(batch['label'].shape)
        break

if __name__ == "__main__":  
    main()