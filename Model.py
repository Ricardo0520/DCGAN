import setting
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(setting.nz, setting.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(setting.ngf * 8),
            nn.ReLU(True),
            # state size. (setting.ngf*8) x 4 x 4
            nn.ConvTranspose2d(setting.ngf * 8, setting.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(setting.ngf * 4),
            nn.ReLU(True),
            # state size. (setting.ngf*4) x 8 x 8
            nn.ConvTranspose2d( setting.ngf * 4, setting.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(setting.ngf * 2),
            nn.ReLU(True),
            # state size. (setting.ngf*2) x 16 x 16
            nn.ConvTranspose2d( setting.ngf * 2, setting.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(setting.ngf),
            nn.ReLU(True),
            # state size. (setting.ngf) x 32 x 32
            nn.ConvTranspose2d( setting.ngf, setting.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (setting.nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (setting.nc) x 64 x 64
            nn.Conv2d(setting.nc, setting.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (setting.ndf) x 32 x 32
            nn.Conv2d(setting.ndf, setting.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(setting.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (setting.ndf*2) x 16 x 16
            nn.Conv2d(setting.ndf * 2, setting.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(setting.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (setting.ndf*4) x 8 x 8
            nn.Conv2d(setting.ndf * 4, setting.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(setting.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (setting.ndf*8) x 4 x 4
            nn.Conv2d(setting.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)