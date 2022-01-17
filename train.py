import Data
import setting
import Model
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML


def weights_init(m):
    classname = m.__class__.__name__
    if classname .find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1 :
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0) 


netG = Model.Generator(setting.ngpu).to(setting.device)
netG.apply(weights_init)
netD = Model.Discriminator(setting.ngpu).to(setting.device)
netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64,setting.nz,1,1,device=setting.device)

real_label =  1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(),lr = setting.lr,betas = (setting.betal, 0.999))
optimizerG = optim.Adam(netG.parameters(),lr=setting.lr,betas=(setting.betal,0.999))

img_list = []
G_losses = []
D_losses=[]
iters = 0
print("Starting Tranining...")

for epoch in range(setting.num_epochs):
    for i, data in enumerate(Data.dataloader,1):
        netD.zero_grad()
        real_cpu = data[0].to(setting.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,),real_label,device = setting.device,dtype=torch.float32)
        output = netD(real_cpu)
        output = output.view(-1)
        errD_real =criterion(output,label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size,setting.nz,1,1,device=setting.device)
        fake = netG(noise)
        label = torch.full((b_size,),fake_label,device = setting.device,dtype=torch.float32)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output,label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_fake+errD_real
        optimizerD.step()

        netG.zero_grad()
        label = torch.full((b_size,),real_label,device = setting.device,dtype=torch.float32)
        output = netD(fake).view(-1)
        errG = criterion(output,label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 ==0:
            print(f"epoch:{epoch+1} iters:{i}/{len(Data.dataloader)} loss_D:{errD.item()} loss_G:{errG.item()}")

        G_losses.append(errG.item()) 
        D_losses.append(errD.item())

        if iters % 500 ==0 or i == (len(Data.dataloader)-1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            
            img_list.append(vutils.make_grid(fake,padding=2,normalize=True))
        
        iters += 1

    torch.save(netD, './model/netD_%d.pkl'%epoch)  # 保存所有的网络参数
    torch.save(netG, './model/netG_%d.pkl'%epoch)
    torch.save(netD.state_dict(), './model/netD_parameter_%d.pkl'%epoch)  # 保存优化选项默认字典，不保存网络结构
    torch.save(netG.state_dict(), './model/netG_parameter_%d.pkl'%epoch)




# fig = plt.figure(figsize=(8,8))
# plt.axis('off')
# ims=[ plt.imshow(np.transpose(i,(1,2,0))) for i  in img_list]
# ani = animation.ArtistAnimation(fig,ims,interval =1000,repeat_delay = 1000,blit = True)
# HTML(ani.to_jshtml())

plt.figure(figsize=(10,5))
plt.title('Generator and Discriminator Loss')
plt.plot(G_losses,label='G')
plt.plot(D_losses,label='D')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()
plt.show()

print('fixed_noise')
fixed_noise = torch.randn(64,setting.nz,1,1,device=setting.device)
fake = netG(fixed_noise).detach().cpu()
img=vutils.make_grid(fake, padding=2, normalize=True)

real_batch = next(iter(Data.dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis('off')
plt.title('Real Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(setting.device)[:64],padding=5,normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis('off')
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()



