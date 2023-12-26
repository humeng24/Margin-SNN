from module import *
from model import *
import utils.utils as utils
from tqdm import tqdm
from model.resnet import *
from evaluation.marginsnn.adaptive_attack import *
from datasets.load_dataset import data_loader


dataset = 'mnist'       # mnist | fashion_mnist | emnist | cifar10 | cifar100 | svhn
model_s = 'resnet18'              # lenet | resnet | vgg11 | resnet_dul | resnet18
img_complexity = 'g0'           # g0 | g1 | g2
distance_metric = 'euclidean'   # euclidean | cosine
use_normalization = False
cuda = 1
save_path = f"/data/khp/save_model/{dataset}/label_embedding/"
log_path = f"/data/khp/log/{dataset}/label_embedding/"
writer_path = f'/data/khp/drawing/{model_s}_{dataset}_label_embedding'

# 网络设置
if 'mnist' in dataset:
    input_cl = 1
else:
    input_cl = 3

node_num = 256
nb_classes = 10


# 自定义网络
class net(nn.Module):
    def __init__(self, backbone, classes, latent_dim):
        super(net, self).__init__()
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
    def __init__(self, beta=0.35, alpha=0.01):
        super(margin_loss, self).__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, mu, logvar, label_embedding, output, target):
        if distance_metric == 'cosine':
            label_distance = torch.mm(label_embedding, label_embedding.T)
            logits_mask = torch.scatter(
                torch.ones_like(label_distance),
                1,
                torch.arange(label_distance.shape[0]).view(-1, 1).to(device),
                0
            )

            base_loss = F.cross_entropy(output, target)

            kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
            kl_loss = kl_loss.sum(dim=1).mean()

            label_distance = label_distance * logits_mask
            self.beta = - self.beta
        elif distance_metric == 'euclidean':
            classes = label_embedding.shape[0]
            label_distance = (label_embedding.unsqueeze(1).repeat((1, classes, 1)) - label_embedding.repeat(classes, 1, 1)).pow(2).sum(dim=2)

            base_loss = F.cross_entropy(output, target)

            kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
            kl_loss = kl_loss.sum(dim=1).mean()

            # distance_matrix = distance_matrix * logits_mask
        return base_loss - self.beta * torch.mean(label_distance) + self.alpha * kl_loss


def test(eps, nb_iter, eps_iter):

    class_dict = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
                  5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    label = [label for _, label in class_dict.items()]

    confusion = utils.ConfusionMatrix(num_classes=nb_classes, labels=label)

    model.eval()
    adversary = PGD(model, eps=eps, steps=nb_iter, alpha=eps_iter)
    # adversary = FGSM(model, eps=eps)

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
            mu, logvar = model.mu.clone(), model.logvar.clone()
            label_embedding = model.y_embedding

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

        confusion.update(advpred.cpu().numpy(), clnpred.cpu().numpy())

    test_clnloss /= len(test_loader.dataset)

    # 对抗样本准确率
    print('Test set: avg adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.0f}%)'.format(
        test_advloss, advcorrect, len(test_loader.dataset),
        100. * advcorrect / len(test_loader.dataset)))

    return test_clnloss, clncorrect, advcorrect


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
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

    # 数据集选择
    data_selection = data_loader()
    train_loader, test_loader = data_selection(dataset, normalize=use_normalization)

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')  # 启动GPU

    # 选择模型
    if model_s == 'lenet':
        backbone_model = Lenet(input_cl=input_cl)
    elif model_s == 'vgg11':
        backbone_model = VGG11(input_cl=input_cl)
    elif model_s == 'resnet34':
        backbone_model = ResNet34(input_channels=input_cl)
    elif model_s == 'resnet18':
        backbone_model = ResNet18(input_channels=input_cl)
    else:
        raise ValueError("没有此模型")

    model = net(backbone_model, nb_classes, node_num)

    model_name = "/data/khp/save_model/mnist/label_embedding/resnet18_mnist_256nodenum_0.0003lr_1003seed_without_normalization_le_margin_kl_advtrain_mix.pth"
    # model_name = "/data/khp/save_model/mnist/resnet18_mnist_256nodenum_0.0003lr_1003seed_without_normalization_le_margin_kl_advtrain.pth"
    model.load_state_dict(torch.load(os.path.join(save_path, model_name), map_location='cpu'))
    # model.load_state_dict(torch.load(model_name, map_location='cpu'))
    model = model.to(device)

    loss_function = margin_loss()
    eps_ls = np.arange(0, 1.1, 0.05)
    adv_ls = []
    print(eps_ls)
    for eps in eps_ls:
        _, adv_acc, _ = test(eps, nb_iter, eps_iter)
        adv_ls.append(adv_acc)
    plt.plot(eps_ls, adv_ls)
    plt.show()