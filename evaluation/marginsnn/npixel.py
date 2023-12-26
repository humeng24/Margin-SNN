from module import *
from model import *
from evaluation.marginsnn.complexity import *
import utils.utils as utils
from tqdm import tqdm
from datasets.load_dataset import data_loader
from torchattacks import OnePixel


dataset = 'cifar10'       # mnist | fashion_mnist | emnist | cifar10 | cifar100
model_s = 'resnet_dul'              # lenet | resnet | vgg11 | resnet_dul
img_complexity = 'g0'           # g0 | g1 | g2
use_normalization = False
seed = 6620
cuda = 1
save_path = f"/data/khp/save_model/{dataset}/label_embedding/"                 # baseline | label_embedding
log_path = f"/data/khp/log/{dataset}/label_embedding/"
writer_path = f'/data/khp/drawing/{model_s}_{dataset}_label_embedding'

# 网络设置
if 'mnist' in dataset:
    input_cl = 1
elif 'cifar' in dataset:
    input_cl = 3

node_num = 256
nb_classes = 10


# 自定义网络
class base_net(nn.Module):
    def __init__(self, backbone, classes):
        super(base_net, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(in_features=node_num, out_features=classes)

    def forward(self, x):
        x = self.backbone(x)
        self.feature = x
        x = self.fc(x)
        return x


# 自定义网络
class dul_net(nn.Module):
    def __init__(self, backbone, classes):
        super(dul_net, self).__init__()
        self.backbone = backbone
        self.classes = classes
        self.label_embedding = nn.Embedding(classes, node_num)
        # self.fc = nn.Linear(in_features=node_num, out_features=10)

    def forward(self, x):
        # get embedding
        x_embedding = self.backbone(x)

        self.mu = self.backbone.mu.clone()
        self.logvar = self.backbone.logvar.clone()

        y = torch.tensor(range(self.classes)).to(device)
        y_embedding = self.label_embedding(y)
        self.y_embedding = y_embedding

        output = torch.mm(x_embedding, y_embedding.T)
        return output


class margin_loss(nn.Module):
    def __init__(self, beta=1, alpha=0.01):
        super(margin_loss, self).__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, mu, logvar, label_embedding, output, target):
        distance_matrix = torch.mm(label_embedding, label_embedding.T)
        logits_mask = torch.scatter(
            torch.ones_like(distance_matrix),
            1,
            torch.arange(distance_matrix.shape[0]).view(-1, 1).to(device),
            0
        )

        base_loss = F.cross_entropy(output, target)

        kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
        kl_loss = kl_loss.sum(dim=1).mean()

        distance_matrix = distance_matrix * logits_mask
        return base_loss + self.beta * torch.mean(distance_matrix) + self.alpha * kl_loss


def test():
    if dataset == 'mnist':
        eps = 0.3
    elif dataset == 'fashion_mnist':
        eps = 0.2
    elif dataset == 'cifar10':
        eps = 8/255
        class_dict = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
                      5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        label = [label for _, label in class_dict.items()]

    confusion = utils.ConfusionMatrix(num_classes=nb_classes, labels=label)

    model.eval()
    adversary = OnePixel(model)

    test_clnloss = 0
    clncorrect = 0

    test_advloss = 0
    advcorrect = 0
    robustness = 0

    output_ls = []          # list for storing softmax output

    for clndata, target in tqdm(test_loader):
        clndata = clndata.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(clndata)
            # mu, logvar = model.mu.clone(), model.logvar.clone()
            # label_embedding = model.y_embedding

        output_ls.append(output)          # softmax output

        test_clnloss += F.cross_entropy(output, target, reduction='sum').item()

        clnpred = output.max(1, keepdim=True)[1]
        clncorrect += clnpred.eq(target.view_as(clnpred)).sum().item()

        advdata = adversary(clndata, target)

        with torch.no_grad():
            output = model(advdata)

        test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
        advpred = output.max(1, keepdim=True)[1]

        advcorrect += advpred.eq(target.view_as(advpred)).sum().item()
        robustness += advpred.eq(clnpred).sum().item()

        confusion.update(advpred.cpu().numpy(), target.cpu().numpy())

    confusion.plot()                # plot confusion matrix
    confusion.summary()             # print confusion matrix

    test_clnloss /= len(test_loader.dataset)
    # test_advloss /= len(test_loader.dataset)

    # print(F.softmax(torch.cat(output_ls)).cpu().detach().numpy()[:100, :])

    # 干净样本准确率
    print('Test set: avg cln loss: {:.4f},'
          ' cln acc: {}/{} ({:.0f}%)'.format(
          test_clnloss, clncorrect, len(test_loader.dataset),
          100. * clncorrect / len(test_loader.dataset)))
    test_advloss /= len(test_loader.dataset)

    # 对抗样本准确率
    print('Test set: avg adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.0f}%)'.format(
        test_advloss, advcorrect, len(test_loader.dataset),
        100. * advcorrect / len(test_loader.dataset)))

    # 计算鲁棒性
    print('Test set: avg adv loss: {:.4f},'
              ' adv robustness: {}/{} ({:.0f}%)\n'.format(
        test_advloss, robustness, len(test_loader.dataset),
        100. * robustness / len(test_loader.dataset)))
    return test_clnloss, clncorrect, advcorrect


def compute_fuzz(model, f=None, if_to_f=False):
    """
    :param model: 模型
    :param writer: tensorboard日志文件
    :param f: 本地日志文件
    :param if_to_f: 是否写进本地日志文件（默认为否）
    :return:
    """
    fuzz_dict = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            param = param.detach().cpu().numpy()
            param = param.reshape(1, -1)
            fuzz = utils.fuzziness(param)

            fuzz_dict[name] = fuzz

            if if_to_f:
                print_log("\nThe fuzziness of {} = {:.4f}".format(name, fuzz), f)
    return fuzz_dict


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # write logs into files
    with open(log_path, 'a+') as f:
        f.write(log_info + '\n')


if __name__ == '__main__':

    g = image_complexity(img_complexity)

    # 数据集选择
    data_selection = data_loader()
    train_loader, test_loader = data_selection(dataset, normalize=use_normalization, test_batch=256)

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')       # 启动GPU
    utils.setup_seed(seed)                                                      # 设置随机数种子

    # 选择模型
    if model_s == 'lenet':
        backbone_model = Lenet(input_cl=input_cl)
    elif model_s == 'resnet':
        backbone_model = ResNet_18(ResBlock, input_cl=input_cl)
    elif model_s == 'vgg11':
        backbone_model = VGG11(input_cl=input_cl)
    elif model_s == 'resnet_dul':
        backbone_model = ResNet_18_dul(ResBlock, input_cl=input_cl, latent_dim=node_num)
    else:
        raise ValueError("没有此模型")

    model = dul_net(backbone_model, nb_classes)
    model = model.to(device)

    loss_function = margin_loss()
    # 记录精度最高的模型的各层模糊度
    # model_name = 'resnet_cifar10_512nodenum_0.001lr_6666seed_without_normalization.pth'
    model_name = 'resnet_dul_cifar10_256nodenum_0.0003lr_551seed_without_normalization.pth'

    model.load_state_dict(torch.load(save_path + model_name, map_location='cpu'))
    print("Load done")

    # 记录对抗样本准确率
    if 'mnist' in dataset:
        eps = 0.3
    elif 'cifar' in dataset:
        eps = 8/255

    test()