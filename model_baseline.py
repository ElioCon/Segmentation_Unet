import torch
import torch.nn as nn

# create Segmentation model
class Model(nn.Module):
    def __init__(self, in_channels=3, classes=19): 
        super(Model, self).__init__()
        # TODO: Implement a convolutional neural network for image segmentation.
        # (1) Design the layers to extract features from input images.
        # (2) Ensure the final layer outputs a tensor of shape [B, 1, X, Y],
        #     where B is the batch size, and X and Y are the dimensions of the output segmentation map.
        #     This indicates the segmentation prediction for each pixel in the input image.
        factor = 2**3
        #factor2 = factor*(2**1)
        # Encoder (contracting path)
        self.conv1 = self.convolve_block(in_channels, (2**1)*factor, kernel_size=3, stride=1, padding=1)
        self.conv2 = self.convolve_block((2**1)*factor, (2**2)*factor, kernel_size=3, stride=1, padding=1)
        self.conv3 = self.convolve_block((2**2)*factor, (2**3)*factor, kernel_size=3, stride=1, padding=1)
        self.conv4 = self.convolve_block((2**3)*factor, (2**4)*factor, kernel_size=3, stride=1, padding=1)
        self.conv5 = self.convolve_block((2**4)*factor, (2**5)*factor, kernel_size=3, stride=1, padding=1)
        
        #Max pool
        self.max = nn.MaxPool2d(2, stride=2)
        
        # Bottleneck
        self.bottleneck = self.bottleneck_block(external_channels=(2**5)*factor,internal_channels=(2**6)*factor,kernel_size=3,stride=1,padding=1)
        
        # Transpose convolution (perform unpooling operation, image size * 2)
        #self.unpool = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    
        # Decoder (expanding path)
        self.upconv4 = self.expand_block((2**6)*factor, (2**5)*factor, kernel_size=3, stride=1, padding=1)
        self.upconv3 = self.expand_block((2**5)*factor, (2**4)*factor, kernel_size=3, stride=1, padding=1) 
        self.upconv2 = self.expand_block((2**4)*factor, (2**3)*factor, kernel_size=3, stride=1, padding=1)  # Adjusting input channels
        self.upconv1 = self.expand_block((2**3)*factor, (2**2)*factor, kernel_size=3, stride=1, padding=1)  # Adjusting input channels
        #self.upconv1 = self.expand_block(192//factor, 64//factor, kernel_size=3, stride=1)   # Adjusting input channels
        
        # Output layer
        self.conv_out = self.convolve_block((2**2)*factor, (2**1)*factor, kernel_size=3, stride=1, padding=1)
        self.output = nn.Conv2d((2**1)*factor, classes, kernel_size=1)
        
    def convolve_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def bottleneck_block(self, external_channels, internal_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(external_channels, internal_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(internal_channels, external_channels, kernel_size=2, stride=2)
        )
    
    def expand_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels=out_channels//2, kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Encoder (contracting path)
        #if len(x.size()) == 3:
        #    x.unsqueeze(0)
        conv1 = self.conv1(x)
        x = self.max(conv1)
        conv2 = self.conv2(x)
        x = self.max(conv2)
        conv3 = self.conv3(x)
        x = self.max(conv3)
        conv4 = self.conv4(x)
        x = self.max(conv4)
        conv5 = self.conv5(x)
        x = self.max(conv5)
        
        # Bottleneck
        bottleneck = self.bottleneck(x)
        
        # Decoder (expanding path)
        #print(conv4.shape)
        #print(bottleneck.shape)
        upconv4 = self.upconv4(torch.cat([conv5, bottleneck], dim=1))
        #print(conv3.shape)
        #print(upconv3.shape)
        upconv3 = self.upconv3(torch.cat([conv4, upconv4], dim=1))
        #print(conv2.shape)
        #print(upconv2.shape)
        upconv2 = self.upconv2(torch.cat([conv3, upconv3], dim=1))
        #print(conv1.shape)
        #print(upconv1.shape)
        upconv1 = self.upconv1(torch.cat([conv2, upconv2], dim=1))
        # Output layer
        output = self.conv_out(torch.cat([conv1, upconv1], dim=1))
        output = self.output(output)
        return output
