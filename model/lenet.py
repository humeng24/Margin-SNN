from module import *


# conv2d(in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeors')
# input: (bs, 1, 28, 28)
class Lenet(nn.Module):
    def __init__(self, input_cl):
        super(Lenet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_cl, 6, 5, 1, 2),           # output: (bs, 6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)       # output: (bs, 6, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),          # output: (bs, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)       # output: (bs, 16, 5, 5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # 将多维向量展平为一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        self.feature = x
        return x


if __name__ == '__main__':
    lenet = Lenet(input_cl=3)
    print(lenet)
