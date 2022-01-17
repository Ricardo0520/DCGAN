import setting
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import torch.utils

trans = transforms.Compose([
    transforms.Resize(setting.image_size),
    transforms.CenterCrop(setting.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

dataset = datasets.ImageFolder(root=setting.dataroot,transform=trans)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=setting.batch_size,shuffle=True)