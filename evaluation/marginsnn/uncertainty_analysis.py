from module import *
from model import *
from evaluation.marginsnn.complexity import *
import utils.utils as utils
from tqdm import tqdm
from model.resnet import *
from datasets.load_dataset import data_loader
# from advertorch.attacks import SinglePixelAttack
from advertorch.attacks import LinfPGDAttack, FGSM


dataset = 'cifar10'              # mnist | fashion_mnist | emnist | cifar10 | cifar100 | svhn
model_s = 'resnet18'              # lenet | resnet | vgg11 | resnet_dul | resnet34
img_complexity = 'g0'           # g0 | g1 | g2
use_normalization = False
cuda = 1
save_path = f"/data/khp/save_model/{dataset}/label_embedding/"
log_path = f"/data/khp/log/{dataset}/label_embedding/"
writer_path = f'/data/khp/drawing/{model_s}_{dataset}_label_embedding'

# 网络设置
if 'mnist' in dataset:
    input_cl = 1
elif 'cifar' in dataset:
    input_cl = 3
elif dataset == 'svhn':
    input_cl = 3

seed = 6600
node_num = 256

if dataset == 'cifar100':
    nb_classes = 100
else:
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
    def __init__(self, backbone, classes, latent_dim):
        super(dul_net, self).__init__()
        self.backbone = backbone
        self.classes = classes
        self.label_embedding = nn.Embedding(classes, latent_dim)
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)

    # 重采样
    def _reparamterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # get embedding
        x_embedding = self.backbone(x)

        self.mu = self.mu_head(x_embedding)
        self.logvar = self.logvar_head(x_embedding)

        x_embedding = self._reparamterize(self.mu, self.logvar)

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


def test(eps, nb_iter, eps_iter):

    class_dict = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
                  5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    label = [label for _, label in class_dict.items()]

    model.eval()
    pgd = LinfPGDAttack(model, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter)
    fgsm = FGSM(model, eps=eps)

    test_clnloss = 0
    clncorrect = 0

    pgd_advloss = 0
    pgd_advcorrect = 0

    fgsm_advloss = 0
    fgsm_advcorrect = 0

    logvar_ls = []          # list for storing softmax output

    for clndata, target in tqdm(test_loader):
        clndata = clndata.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(clndata)
            mu, logvar = model.mu.clone(), model.logvar.clone()
            # label_embedding = model.y_embedding

        logvar_ls.append(torch.exp(logvar).sqrt())          # softmax output

        test_clnloss += F.cross_entropy(output, target, reduction='sum').item()

        clnpred = output.max(1, keepdim=True)[1]
        clncorrect += clnpred.eq(target.view_as(clnpred)).sum().item()

        pgd_advdata = pgd.perturb(clndata, target)
        fgsm_advdata = fgsm.perturb(clndata, target)

        with torch.no_grad():
            pgd_output = model(pgd_advdata)
            fgsm_output = model(fgsm_advdata)

        pgd_advloss += F.cross_entropy(
            pgd_output, target, reduction='sum').item()
        pgd_advpred = pgd_output.max(1, keepdim=True)[1]

        pgd_advcorrect += pgd_advpred.eq(target.view_as(pgd_advpred)).sum().item()

        fgsm_advloss += F.cross_entropy(
            fgsm_output, target, reduction='sum').item()
        fgsm_advpred = fgsm_output.max(1, keepdim=True)[1]

        fgsm_advcorrect += fgsm_advpred.eq(target.view_as(fgsm_advpred)).sum().item()

    print(f'Var of model: {torch.mean(torch.cat(logvar_ls))}')

    test_clnloss /= len(test_loader.dataset)
    # test_advloss /= len(test_loader.dataset)

    # 干净样本准确率
    print('Test set: avg cln loss: {:.4f},'
          ' cln acc: {}/{} ({:.0f}%)'.format(
          test_clnloss, clncorrect, len(test_loader.dataset),
          100. * clncorrect / len(test_loader.dataset)))

    pgd_advloss /= len(test_loader.dataset)

    # PGD攻击
    print('Test set: avg pgd adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.0f}%)'.format(
        pgd_advloss, pgd_advcorrect, len(test_loader.dataset),
        100. * pgd_advcorrect / len(test_loader.dataset)))

    fgsm_advloss /= len(test_loader.dataset)

    # FGSM攻击
    print('Test set: avg fgsm adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.0f}%)'.format(
        fgsm_advloss, fgsm_advcorrect, len(test_loader.dataset),
        100. * fgsm_advcorrect / len(test_loader.dataset)))


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
    elif model_s == 'resnet34':
        backbone_model = ResNet34(input_channels=input_cl)
    elif model_s == 'resnet18':
        backbone_model = ResNet18(input_channels=input_cl)
    else:
        raise ValueError("没有此模型")

    model = dul_net(backbone_model, nb_classes, latent_dim=node_num)
    model = model.to(device)

    loss_function = margin_loss()

    # 记录精度最高的模型的各层模糊度
    # model_name = 'resnet_cifar10_512nodenum_0.001lr_6666seed_without_normalization.pth'
    model_name = 'resnet18_cifar10_256nodenum_0.0003lr_1003seed_without_normalization_le_margin_kl.pth'

    model.load_state_dict(torch.load(save_path + model_name, map_location='cpu'))
    print("Load done")

    # 记录对抗样本准确率
    if input_cl == 1:
        eps = 0.3
        nb_iter = 40
        eps_iter = 0.01
        cw_lr = 0.01
    else:
        eps = 8 / 255
        nb_iter = 10
        eps_iter = eps / 10
        cw_lr = 5e-4

    test(eps, nb_iter=nb_iter, eps_iter=eps_iter)