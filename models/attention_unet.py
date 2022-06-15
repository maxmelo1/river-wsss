import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size = 1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
    
    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.psi(nn.ReLU()(x1 + g1))
        out = nn.Sigmoid()(out)
        return out*x

class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        self.attention = AttentionBlock(out_channels, in_channels, int(out_channels / 2))
        self.conv_bn1 = DoubleConv(in_channels+out_channels, out_channels)
        self.conv_bn2 = DoubleConv(out_channels, out_channels)
    
    def forward(self, x, x_skip):
        # note : x_skip is the skip connection and x is the input from the previous block
        # apply the attention block to the skip connection, using x as context
        x_attention = self.attention(x_skip, x)
        # upsample x to have th same size as the attention map
        x = nn.functional.interpolate(x, x_skip.shape[2:], mode = 'bilinear', align_corners = False)
        # stack their channels to feed to both convolution blocks
        x = torch.cat((x_attention, x), dim = 1)
        x = self.conv_bn1(x)
        return self.conv_bn2(x)


class AttUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(AttUNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                #AttentionBlock(feature, feature*2, feature)
                AttentionUpBlock(feature*2, feature)
                # nn.ConvTranspose2d(
                #     feature*2, feature, kernel_size=2, stride=2,
                # )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            skip_connection = skip_connections[idx//2]
            x = self.ups[idx](x, skip_connection)
            

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    import torchsummary
    x = torch.randn((3, 3, 161, 161))
    model = AttUNET(in_channels=3, out_channels=1)
    preds = model(x)
    #assert preds.shape == x.shape

    #torchsummary.summary(model, x)
    print(preds.shape)

if __name__ == "__main__":
    test()