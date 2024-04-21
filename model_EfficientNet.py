import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, reduced_dim, 1),
            nn.SiLU(),  # SiLU activation
            nn.Conv2d(reduced_dim, input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super().__init__()
        mid_channels = in_channels * expansion_factor
        
        self.use_residual = in_channels == out_channels and stride == 1
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, 1, bias=False) if expansion_factor != 1 else nn.Identity()
        self.bn0 = nn.BatchNorm2d(mid_channels)
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.se_layer = SqueezeExcitation(mid_channels, reduced_dim=int(mid_channels / expansion_factor))
        self.project_conv = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.se_layer(x)
        x = self.project_conv(x)
        x = self.bn2(x)
        if self.use_residual:
            x += identity
        return x

class Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=19, initial_power=5):
        super().__init__()
        self.factor = 2**initial_power
        # Initial block (customized for the appropriate number of input channels, e.g., 3 for RGB images)
        self.encoder0 = MBConv(in_channels=in_channels, out_channels=(2**0)*self.factor, expansion_factor=1, stride=1)

        # Encoder: Increasing channels and reducing dimensions
        self.encoder1 = MBConv(in_channels=(2**0)*self.factor, out_channels=(2**1)*self.factor, expansion_factor=6, stride=1)
        self.encoder2 = MBConv(in_channels=(2**1)*self.factor, out_channels=(2**2)*self.factor, expansion_factor=6, stride=1)
        self.encoder3 = MBConv(in_channels=(2**2)*self.factor, out_channels=(2**3)*self.factor, expansion_factor=6, stride=1)

        # Bottleneck
        self.bottleneck = self.bottleneck_block(external_channels=(2**3)*self.factor,internal_channels=(2**4)*self.factor,kernel_size=3,stride=1,padding=1)

        # Maxpolling
        self.max = nn.MaxPool2d(2, stride=2)

        # Decoder and upsample
        self.decoder3 = self.double_conv((2**4)*self.factor, (2**3)*self.factor)
        self.upconv2 = nn.ConvTranspose2d((2**3)*self.factor, (2**2)*self.factor, 2, stride=2)
        self.decoder2 = self.double_conv((2**3)*self.factor, (2**2)*self.factor)
        self.upconv1 = nn.ConvTranspose2d((2**2)*self.factor, (2**1)*self.factor, 2, stride=2)
        self.decoder1 = self.double_conv((2**2)*self.factor, (2**1)*self.factor)
        self.upconv0 = nn.ConvTranspose2d((2**1)*self.factor, (2**0)*self.factor, 2, stride=2)
        self.decoder0 = self.double_conv((2**1)*self.factor, (2**0)*self.factor)

        # Final classifier
        self.final_conv = nn.Conv2d((2**0)*self.factor, num_classes, 1)

    # Helper function for double convolution
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
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

    def forward(self, x):
        # Encoder path
        #print(x.shape)
        enc0 = self.encoder0(x)     # size//(2**0)
        x = self.max(enc0)          # size//(2**1)
        #print(enc0.shape)
        #print(x.shape)
        enc1 = self.encoder1(x)     # size//(2**1)
        x = self.max(enc1)          # size//(2**2)
        #print(enc1.shape)
        #print(x.shape)           
        enc2 = self.encoder2(x)     # size//(2**2)
        x = self.max(enc2)          # size//(2**3)
        #print("enc2")
        #print(enc2.shape)
        #print(x.shape)  
        enc3 = self.encoder3(x)     # size//(2**3)
        x = self.max(enc3)          # size//(2**4)
        #print(enc3.shape)
        #print(x.shape)

        # Bottelneck
        bottelneck = self.bottleneck(x) # size//(2**3)

        # Decoder path
        #print("Decoder")
        #print(bottelneck.shape)
        #print(enc3.shape)
        dec3 = self.decoder3(torch.cat([bottelneck, enc3], dim=1)) # size//(2**3)
        #print("\n")
        #print(dec3.shape)
        dec2 = self.upconv2(dec3)           # size//(2**2) 
        #print(dec2.shape)
        #print(enc2.shape)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))  # size//(2**2)
        #print("\n")
        #rint(dec3.shape)
        dec1 = self.upconv1(dec2)           # size//(2**1)
        #print(dec1.shape)
        #print(enc1.shape)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))  # size//(2**1)
        #print("\n")
        #print(dec2.shape)
        dec0 = self.upconv0(dec1)           # size//(2**0)
        #print(dec1.shape)
        dec0 = self.decoder0(torch.cat([dec0, enc0], dim=1))  # size//(2**0)
        #print(dec1.shape)

        # Output layer
        out = self.final_conv(dec0)
        return out