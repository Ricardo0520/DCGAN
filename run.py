import setting
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.utils.data

netD=torch.load('./model/netD_7.pkl')
netG=torch.load('./model/netG_7.pkl')

fixed_noise = torch.randn(64,setting.nz,1,1,device=setting.device)
fake = netG(fixed_noise).detach().cpu()
img=vutils.make_grid(fake, padding=2, normalize=True)

plt.axis('off')
plt.title("Fake Images")
plt.imshow(np.transpose(img,(1,2,0)))
plt.show()

