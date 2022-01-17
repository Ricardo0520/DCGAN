import random
import torch

manualSeed = 999
# manualSeed = random.random(1,1000) #每次生成不同种子
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataroot = './Celeba'


batch_size = 128
image_size = 64
nc = 3
nz = 100

ngf = 64
ndf = 64
num_epochs = 8
lr = 0.0002
betal = 0.5
ngpu = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
