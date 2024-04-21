import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, block=ResidualBlock, blocks = [2, 2, 2, 2], in_channels = 3, classes = 19, power = 5):
        super(Model, self).__init__()
        factor = 2**power
        self.inplanes = factor
        # First layer for edges
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, self.inplanes, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(self.inplanes),
                        nn.ReLU())
        # Maxpooling 
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.max = nn.MaxPool2d(2, stride=2)

        # Encoder
        self.layer0 = self._make_layer(block, self.inplanes*2, blocks[0], stride = 1)
        self.x_att0 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.inplanes*2, blocks[1], stride = 1)
        self.x_att1 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.inplanes*2, blocks[2], stride = 1)
        self.x_att2 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.layer3 = self._make_layer(block, self.inplanes*2, blocks[3], stride = 1)
        self.x_att3 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1, stride=2, padding=0)

        # Bottleneck
        self.bottleneck = self.bottleneck_block(external_channels=self.inplanes, internal_channels=self.inplanes*2, kernel_size=3,stride=1,padding=1)
        self.g_att3 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.upB = nn.ConvTranspose2d(self.inplanes, self.inplanes//2, kernel_size=2, stride=2)
        # Decoder (expanding path)
        self.upconv3 = self.expand_block(self.inplanes, self.inplanes//2, kernel_size=3, stride=1, padding=1) 
        self.g_att2 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.up3 = nn.ConvTranspose2d(self.inplanes, self.inplanes//2, kernel_size=2, stride=2)

        self.upconv2 = self.expand_block(self.inplanes, self.inplanes//2, kernel_size=3, stride=1, padding=1)  # Adjusting input channels
        self.g_att1 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.up2 = nn.ConvTranspose2d(self.inplanes, self.inplanes//2, kernel_size=2, stride=2)

        self.upconv1 = self.expand_block(self.inplanes, self.inplanes//2, kernel_size=3, stride=1, padding=1)  # Adjusting input channels
        self.g_att0 = nn.Conv2d(in_channels=self.inplanes, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.up1 = nn.ConvTranspose2d(self.inplanes, self.inplanes//2, kernel_size=2, stride=2)

         # Output layer
        self.conv_out = self.convolve_block(self.inplanes, self.inplanes//2, kernel_size=3, stride=1, padding=1)
        self.output = nn.Conv2d(self.inplanes, classes, kernel_size=1)

        # Attention functions
        self.activateRelu = nn.ReLU(inplace=True)
        self.activateSig = nn.Sigmoid()
        self.conv_psi = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.upconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        #self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def convolve_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.inplanes = out_channels
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.5),
        )

    def bottleneck_block(self, external_channels, internal_channels, kernel_size=3, stride=1, padding=1):
        self.inplanes = internal_channels
        return nn.Sequential(
            nn.Conv2d(external_channels, internal_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.5),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.5),
        )
    
    def expand_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.inplanes = out_channels
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.5),
        )
    
    def forward(self, x):
        # Encoder
        conv0 = self.conv1(x)       # (2**1)*power, size/(2**0)
        x = self.maxpool(conv0)     # size/(2**0)
        conv1 = self.layer0(x)      # (2**2)*power, size/(2**0)
        x_att0 = self.x_att0(conv1)
        x = self.max(conv1)         # size/(2**1)
        conv2 = self.layer1(x)      # (2**3)*power, size/(2**1)
        x_att1 = self.x_att1(conv2)
        x = self.max(conv2)         # size/(2**2)
        conv3 = self.layer2(x)      # (2**4)*power, size/(2**2)
        x_att2 = self.x_att2(conv3)
        x = self.max(conv3)         # size/(2**3)
        conv4 = self.layer3(x)      # (2**5)*power, size/(2**3)
        x_att3 = self.x_att3(conv4)
        x = self.max(conv4)         # size/(2**4)

        # Bottleneck
        bottleneck = self.bottleneck(x) # (2**5)*power, size/(2**3)
        g_att3 = self.g_att3(bottleneck)
        bottleneck = self.upB(bottleneck)
        y = conv4 * self.upconv(self.activateSig(self.conv_psi(self.activateRelu(x_att3 + g_att3))))

        upconv3 = self.upconv3(torch.cat([y, bottleneck], dim=1)) # (2**4)*power, size/(2**2)
        g_att2 = self.g_att2(upconv3)
        upconv3 = self.up3(upconv3)
        y = conv3 * self.upconv(self.activateSig(self.conv_psi(self.activateRelu(x_att2 + g_att2))))

        upconv2 = self.upconv2(torch.cat([y, upconv3], dim=1)) # (2**3)*power, size/(2**1)
        g_att1 = self.g_att1(upconv2)
        upconv2 = self.up2(upconv2)
        y = conv2 * self.upconv(self.activateSig(self.conv_psi(self.activateRelu(x_att1 + g_att1))))

        upconv1 = self.upconv1(torch.cat([y, upconv2], dim=1)) # (2**2)*power, size/(2**0)
        g_att0 = self.g_att0(upconv1)
        upconv1 = self.up1(upconv1)
        y = conv1 * self.upconv(self.activateSig(self.conv_psi(self.activateRelu(x_att0 + g_att0))))

        # Output
        output = self.conv_out(torch.cat([y, upconv1], dim=1)) # (2**1)*power, size/(2**0)
        output = self.output(output)                               # (2**0)*power, size/(2**0)

        return output