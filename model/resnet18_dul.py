from module import *


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = torch.relu(out)

        return out


class ResNet_18_dul(nn.Module):
    def __init__(self, ResBlock, input_cl, latent_dim, nb_classes):
        """
        :param input_cl: 输入图像的通道数
        :param latent_dim: embedding向量的维度
        :param nb_classes: 分类类别数
        """
        super(ResNet_18_dul, self).__init__()

        self.in_channel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_cl, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)

        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)

        self.fc = nn.Linear(latent_dim, nb_classes)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    # 重采样
    def _reparamterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        self.mu = self.mu_head(x)
        self.logvar = self.logvar_head(x)
        embedding = self._reparamterize(self.mu, self.logvar)
        output = self.fc(embedding)
        return output


if __name__ == '__main__':
    x = torch.rand((8, 1, 28, 28))
    model = ResNet_18_dul(ResBlock, input_cl=1, latent_dim=256, nb_classes=10)
    mu, logvar, output = model(x)
    kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
    kl_loss = kl_loss.sum(dim=1)
    print(kl_loss.shape)

