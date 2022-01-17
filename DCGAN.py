import random
import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.random import seed
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import normalize


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataroot = '/home/edward_krucal/Downloads/Datasets/Celeba'

workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100

ngf = 64
ndf = 64
num_epochs = 1
lr = 0.0002
betal = 0.5
ngpu = 1

trans = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

dataset = datasets.ImageFolder(root=dataroot,transform=trans)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision.utils as vutils


def weights_init(m):
    classname = m.__class__.__name__
    if classname .find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1 :
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0) 


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
# print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64,nz,1,1,device=device)

real_label =  1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(),lr = lr,betas = (betal, 0.999))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(betal,0.999))

img_list = []
G_losses = []
D_losses=[]
iters = 0
print("Starting Tranining...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader,1):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,),real_label,device = device,dtype=torch.float32)
        output = netD(real_cpu)
        output = output.view(-1)
        errD_real =criterion(output,label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size,nz,1,1,device=device)
        fake = netG(noise)
        label = torch.full((b_size,),fake_label,device = device,dtype=torch.float32)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output,label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_fake+errD_real
        optimizerD.step()

        netG.zero_grad()
        label = torch.full((b_size,),real_label,device = device,dtype=torch.float32)
        output = netD(fake).view(-1)
        errG = criterion(output,label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 ==0:
            print(f"epoch:{num_epochs} iters:{i}/{len(dataloader)} loss_D:{errD.item()} loss_G:{errG.item()}")

        G_losses.append(errG.item()) 
        D_losses.append(errD.item())

        if iters%500 ==0 or i == (len(dataloader)-1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            
            img_list.append(vutils.make_grid(fake,padding=2,normalize=True))
        
        iters += 1

plt.figure(figsize=(10,5))
plt.title('Generator and Discriminator Loss')
plt.plot(G_losses,label='G')
plt.plot(D_losses,label='D')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()
plt.show()


fig = plt.figure(figsize=(8,8))
plt.axis('off')
ims=[ [plt.imshow(np.transpose(i,(1,2,0)))] for i  in img_list]
ani = animation.ArtistAnimation(fig,ims,interval =1000,repeat_delay = 1000,blit = True)
HTML(ani.to_jshtml())


real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis('off')
plt.title('Real Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],padding=5,normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis('off')
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()



