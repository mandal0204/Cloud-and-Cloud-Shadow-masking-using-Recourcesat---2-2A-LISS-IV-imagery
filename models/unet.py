import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(prev_channels, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU()))
            prev_channels = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_channels, prev_channels*2, 3, padding=1), 
            nn.InstanceNorm2d(prev_channels*2, affine=True),
            nn.ReLU(),
            nn.Conv2d(prev_channels*2, prev_channels*2, 3, padding=1), 
            nn.InstanceNorm2d(prev_channels*2, affine=True),
            nn.ReLU())
        bottleneck_channels = prev_channels*2
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        decoder_in_channels = bottleneck_channels
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(decoder_in_channels, feature, kernel_size=2, stride=2))
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(feature*2, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU()))
            decoder_in_channels = feature
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.up_convs[i](x)
        return self.final_conv(x)

model = UNet(in_channels=3, out_channels=3).to(device)