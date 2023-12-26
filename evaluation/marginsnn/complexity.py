import utils.utils as utils
import pandas as pd
import numpy as np
from model import *
from module import *
from datasets.load_dataset import data_loader
from model import *
from model.resnet import *
import advertorch.attacks as attacks


dataset = 'mnist'              # mnist | fashion_mnist | emnist | cifar10 | cifar100 | svhn
model_s = 'resnet18'             # lenet | resnet | vgg11 | resnet_dul | resnet34 | resnet18

# 网络设置
if 'mnist' in dataset:
    input_cl = 1
elif 'cifar' in dataset:
    input_cl = 3
elif dataset == 'svhn':
    input_cl = 3

node_num = 256

if dataset == 'cifar100':
    nb_classes = 100
else:
    nb_classes = 10

__all__ = ['complexity', 'fish_ratio', 'image_complexity']
cuda = 4
device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

# 连接mongodb并且获取数据
# my_mongo = utils.Mongo()
# data_frame = my_mongo.get_collection("resnet_fashion_mnist_clean", 'epoch_2')
# data_frame = my_mongo.get_df(data_frame)

# pandas删除某一列，1.del df["columns"]，改变原始数据，2.df.drop('columns', axis=1)，不改变原始数据
# data_frame = data_frame.drop(["_id", "cross_entropy", "predict"], axis=1)

# feature_ls = data_frame.columns.values
# feature_ls = feature_ls[:-1]            # 存放特征名的list


class net(nn.Module):
    def __init__(self, backbone, classes, latent_dim):
        super(net, self).__init__()
        self.backbone = backbone
        self.classes = classes
        self.label_embedding = nn.Embedding(classes, node_num)
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

        self.x_embedding = x_embedding
        self.y_embedding = y_embedding

        output = torch.mm(x_embedding, y_embedding.T)
        return output

# class_n = 10               # 分类问题的类别数


# 使用ovo策略计算，若有n个类别，则有n*(n-1)/2个子问题，其中参数data表示数据框，class_num表示类别数
def complexity(data_frame, class_num):
    class_dict = dict()  # 创建空字典分别存放各个类
    # data_frame = data_frame.drop(["_id", "cross_entropy", "predict"], axis=1)  # pandas删除某一列，1.del df["columns"]，改变原始数据，2.df.drop('columns', axis=1)，不改变原始数据
    feature_ls = data_frame.columns.values
    feature_ls = feature_ls[:-1]  # 存放特征名的list

    for i in range(class_num):
        cs = f"class_{i}"
        # print(data_frame.loc[data_frame.ground_truth == i].shape)
        class_dict[cs] = data_frame.loc[data_frame["ground_truth"].astype(int) == i]

    class_ls = list(class_dict.keys())
    complexity_num = class_num * (class_num - 1) / 2
    fisher_complexity_ls = np.zeros(int(complexity_num))
    intra_inter_complexity_ls = np.zeros(int(complexity_num))
    index = 0

    # 遍历类
    for ith in range(class_num):
        compare_1 = class_ls[ith]
        csf_0 = class_dict[compare_1]
        label_embedding_0 = csf_0[feature_ls].values
        label_embedding_0 = torch.from_numpy(label_embedding_0)
        distance_intra = (label_embedding_0.unsqueeze(1).repeat((1, label_embedding_0.shape[0], 1)) - label_embedding_0.repeat(label_embedding_0.shape[0], 1, 1)).pow(2).sum(dim=2)
        eye_matrix = np.eye(label_embedding_0.shape[0]) * 500000
        distance_intra = np.min((distance_intra.detach().cpu().numpy()+eye_matrix), axis=0)
        # print(np.min(distance_intra, axis=0))
        mean_0 = csf_0.mean(axis=0)
        std_0 = csf_0.std(axis=0)
        # 遍历除了该类的其他类
        for jth in range(ith + 1, class_num):
            compare_2 = class_ls[jth]
            csf_1 = class_dict[compare_2]
            label_embedding_1 = csf_1[feature_ls].values
            label_embedding_1 = torch.from_numpy(label_embedding_1)
            distance_inter = (label_embedding_1.unsqueeze(1).repeat((1, label_embedding_0.shape[0], 1)) - label_embedding_0.repeat(label_embedding_1.shape[0], 1, 1)).pow(2).sum(dim=2)
            # print((distance_inter.detach().cpu().numpy()))
            distance_inter = np.min((distance_inter.detach().cpu().numpy()), axis=0)

            mean_1 = csf_1.mean(axis=0)
            std_1 = csf_1.std(axis=0)
            # 计算fisher判别率
            fish_ls = []
            # print("Intra:", np.sum(distance_intra))
            # print("Inter:", np.sum(distance_inter))
            intra_inter_complexity_ls[index] = np.sum(distance_intra) / np.sum(distance_inter)
            for kth in feature_ls:
                # fish_dict[kth] = fish_ratio(mean_0[kth], mean_1[kth], std_0[kth], std_1[kth])
                fish_ls.append(fish_ratio(mean_0[kth], mean_1[kth], std_0[kth], std_1[kth]))
            # fish_arr = np.array(list(fish_dict.values()))
            fisher_complexity_ls[index] = 1 / np.mean(fish_ls)
            # complexity_ls[index] = np.max(fish_arr[:-1])
            index += 1
    # print(complexity_ls)
    return np.mean(intra_inter_complexity_ls)


# fisher判别率计算公式
def fish_ratio(mean_1, mean_2, std_1, std_2):
    up = np.square(mean_1 - mean_2)
    down = std_1 + std_2
    return up / (down + 1e-14)


# Computing image complexity
class image_complexity:

    def __init__(self, ft):
        self.fun = ft

    def G0(self, x):
        fuzzy_pixel = x * np.log(x + 1e-12) + (1 - x) * np.log(1 - x + 1e-12)
        return -np.sum(fuzzy_pixel, axis=1) / (x.shape[1] * np.log(2))

    def G1(self, x):
        def low(xi):
            return xi * (np.exp(1 - xi) - np.exp(xi - 1))

        def high(xi):
            return (1 - xi) * (np.exp(xi) - np.exp(-xi))

        yita = np.where(x < 0.5, low(x), high(x))
        return np.sum(yita, axis=1) * 2 * np.sqrt(np.e) / (x.shape[1] * (np.e - 1))

    def G2(self, x):
        return np.sum(x * (1 - x), axis=1) * 4 / x.shape[1]

    def __call__(self, x):
        x = x.cpu().detach().numpy()
        x = x.reshape(x.shape[0], -1)
        if self.fun.lower() == 'g0':
            return self.G0(x)
        elif self.fun.lower() == 'g1':
            return self.G1(x)
        elif self.fun.lower() == 'g2':
            return self.G2(x)


def test():

    if dataset == 'mnist':
        eps = 0.3
    elif dataset == 'fashion_mnist':
        eps = 0.2
    elif dataset == 'cifar10':
        eps = 8/255
    elif dataset == 'cifar100':
        eps = 0.003
    elif dataset == 'svhn':
        eps = 0.003

    adversary = attacks.LinfPGDAttack(model, eps=eps, nb_iter=10)
    model.eval()

    test_clnloss = 0
    clncorrect = 0

    test_advloss = 0
    advcorrect = 0
    robustness = 0

    # matrix for calculating embedding
    embedding_matrix = np.zeros((len(test_dataloader.dataset), node_num+1))
    correct_matrix = np.zeros(1)

    for i, (clndata, target) in enumerate(test_dataloader):
        clndata = clndata.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(clndata)

        x_embedding = model.x_embedding.clone()
        embedding_matrix[i * data_selection.test_batch:(i + 1) * data_selection.test_batch, :] = np.concatenate(
            (x_embedding.detach().cpu().numpy(), target.unsqueeze(1).detach().cpu().numpy()), axis=1)

        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()

        clnpred = output.max(1, keepdim=True)[1]
        clncorrect += clnpred.eq(target.view_as(clnpred)).sum().item()

        advdata = adversary.perturb(clndata, target)

        with torch.no_grad():
            output = model(advdata)

        test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
        advpred = output.max(1, keepdim=True)[1]

        advcorrect += advpred.eq(target.view_as(advpred)).sum().item()
        robustness += advpred.eq(clnpred).sum().item()
        correct_matrix = np.concatenate((correct_matrix, advpred.eq(target.view_as(advpred)).reshape((1, -1))[0].detach().cpu().numpy()), axis=0)

    # embedding_df = pd.DataFrame(embedding_matrix, columns=[f'feature_{i + 1}' for i in range(node_num)] + ['label'])

    test_clnloss /= len(test_dataloader.dataset)
    # test_advloss /= len(test_loader.dataset)

    # 干净样本准确率
    print('--------------- calculate correct rate ----------------')
    print('Test set: avg cln loss: {:.4f},'
          ' cln acc: {}/{} ({:.0f}%)'.format(
          test_clnloss, clncorrect, len(test_dataloader.dataset),
          100. * clncorrect / len(test_dataloader.dataset)))
    test_advloss /= len(test_dataloader.dataset)

    # 对抗样本准确率
    print('Test set: avg adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.0f}%)'.format(
        test_advloss, advcorrect, len(test_dataloader.dataset),
        100. * advcorrect / len(test_dataloader.dataset)))

    # 计算鲁棒性
    print('Test set: avg adv loss: {:.4f},'
              ' adv robustness: {}/{} ({:.0f}%)\n'.format(
        test_advloss, robustness, len(test_dataloader.dataset),
        100. * robustness / len(test_dataloader.dataset)))
    return test_clnloss, clncorrect, advcorrect, correct_matrix


if __name__ == '__main__':
    # model_path = "/data/khp/save_model/cifar10/label_embedding/resnet18_cifar10_256nodenum_0.0003lr_1003seed_without_normalization_le_margin_kl.pth"
    # model_path = "/data/khp/save_model/cifar10/label_embedding/resnet18_cifar10_256nodenum_0.0003lr_1003seed_without_normalization_le_kl.pth"
    # model_path = "/data/khp/save_model/cifar10/label_embedding/beta/resnet18_cifar10_256nodenum_0.0003lr_1003seed_0.001beta_without_normalization_le_margin_kl.pth"
    # model_path = "/data/khp/save_model/cifar10/label_embedding/beta/resnet18_cifar10_256nodenum_0.0003lr_1003seed_0.01beta_without_normalization_le_margin_kl.pth"
    # model_path = "/data/khp/save_model/cifar10/label_embedding/beta/resnet18_cifar10_256nodenum_0.0003lr_1003seed_1beta_without_normalization_le_margin_kl.pth"
    # model_path = "/data/khp/save_model/cifar10/label_embedding/beta/resnet18_cifar10_256nodenum_0.0003lr_1003seed_0.0001beta_without_normalization_le_margin_kl.pth"
    # model_path = "/data/khp/save_model/cifar10/label_embedding/resnet18_cifar10_256nodenum_0.0003lr_783seed_without_normalization_le_margin_kl.pth"
    # model_path = "/data/khp/save_model/fashion_mnist/label_embedding/resnet18_fashion_mnist_256nodenum_0.0003lr_783seed_without_normalization.pth"
    # model_path = "/data/khp/save_model/mnist/label_embedding/resnet18_mnist_256nodenum_0.0003lr_1003seed_without_normalization_le_margin_kl.pth"
    # model_path = "/data/khp/save_model/mnist/label_embedding/resnet18_mnist_256nodenum_0.0003lr_1003seed_without_normalization_le_margin_kl_advtrain_mix.pth"
    model_path = "/data/khp/save_model/mnist/resnet18_mnist_256nodenum_0.0003lr_1003seed_without_normalization_le_margin_kl_advtrain.pth"
    if model_s == 'lenet':
        backbone_model = Lenet(input_cl=input_cl)
    elif model_s == 'resnet':
        backbone_model = ResNet_18(ResBlock, input_cl=input_cl)
    elif model_s == 'vgg11':
        backbone_model = VGG11(input_cl=input_cl)
    elif model_s == 'resnet34':
        backbone_model = ResNet34(input_channels=input_cl)
    elif model_s == 'resnet18':
        backbone_model = ResNet18(input_channels=input_cl)
    else:
        raise ValueError("没有此模型")

    model = net(backbone_model, nb_classes, latent_dim=node_num)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)

    # g0 = image_complexity('g0')

    data_selection = data_loader()

    # Print the information of dataset
    train_dataloader, test_dataloader = data_selection(dataset, test_batch=256)

    embedding_matrix = np.zeros((1, 256))
    # var_matrix = np.zeros((1, 256))
    var_matrix = np.zeros(1)
    label_matrix = np.zeros(1, dtype=np.int32)
    correct_matrix = np.zeros(1)
    for data, label in test_dataloader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        logvar = torch.exp(model.logvar).sqrt()
        var = logvar.detach().cpu().numpy()
        var = 1 / np.mean(1 / var, axis=1)

        # print(data[index, :, :, :].shape)
        # print(data.shape)
        # print(label[index].shape)
        # print(label.shape)

        pred = output.max(1, keepdim=True)[1]
        # print(pred.eq(label.view_as(pred)).reshape((1,-1))[0])

        embedding = model.mu.detach().cpu().numpy()
        # logvar = torch.exp(model.logvar).sqrt()
        # var = logvar.detach().cpu().numpy()
        embedding_matrix = np.concatenate((embedding_matrix, embedding), axis=0)
        var_matrix = np.concatenate((var_matrix, var), axis=0)
        label_matrix = np.concatenate((label_matrix, np.round(label.detach().cpu().numpy()).astype(int)), axis=0)
        # correct_matrix = np.concatenate((correct_matrix, pred.eq(label.view_as(pred)).reshape((1,-1))[0].detach().cpu().numpy()), axis=0)
        # img_fuzz = g0(data)
        # print(max(img_fuzz))

    # 可视化
    # var_matrix = var_matrix[1:]
    # print(min(var_matrix))
    # print(max(var_matrix))
    # print(np.median(var_matrix))
    #
    # min_index = np.where(var_matrix < 0.8)[0]
    # median_index = np.where(var_matrix > 0.8)[0]
    # max_index = np.where(var_matrix > 0.9)[0]
    #
    # median_index = np.setdiff1d(median_index, max_index, True)
    # print(len(min_index))
    # print(len(max_index))
    # print(len(median_index))
    #
    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # fig.suptitle("Median Variance", fontsize=15)
    #
    # for i in range(16):
    #     plt.subplot(4, 4, i + 1)
    #     # plt.imshow(test_dataloader.dataset[median_index[i]][0].permute(1, 2, 0), cmap="gray")
    #     plt.imshow(test_dataloader.dataset[median_index[i]][0].reshape((28, 28)), cmap="gray")
    #     plt.axis('off')  # 关闭坐标轴
    # plt.savefig(f"/data/khp/drawing/{dataset}_median_variance.png")
    # plt.show()
    #
    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # fig.suptitle("High Variance", fontsize=15)
    # for i in range(16):
    #     plt.subplot(4, 4, i + 1)
    #     # plt.imshow(test_dataloader.dataset[max_index[i]][0].permute(1, 2, 0), cmap="gray")
    #     plt.imshow(test_dataloader.dataset[max_index[i]][0].reshape((28, 28)), cmap="gray")
    #     plt.axis('off')  # 关闭坐标轴
    # plt.savefig(f"/data/khp/drawing/{dataset}_high_variance.png")
    # plt.show()
    #
    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # fig.suptitle("Low Variance", fontsize=15)
    # for i in range(16):
    #     plt.subplot(4, 4, i + 1)
    #     # plt.imshow(test_dataloader.dataset[min_index[i]][0].permute(1, 2, 0), cmap="gray")
    #     plt.imshow(test_dataloader.dataset[min_index[i]][0].reshape((28, 28)), cmap="gray")
    #     plt.axis('off')  # 关闭坐标轴
    # plt.savefig(f"/data/khp/drawing/{dataset}_low_variance.png")
    # plt.show()

    # print(1 / np.mean(1 / var_matrix[1:, :], axis=1))
    # print(correct_matrix[1:].shape)
    # # print(embedding_matrix[1:, :])
    label_matrix = label_matrix.reshape((10001, 1))
    embedding_label = np.concatenate((embedding_matrix[1:, :], label_matrix[1:, :]), axis=1)
    # print(embedding_label.shape)
    colum_name = [f"feature_{i}" for i in range(256)] + ["ground_truth"]
    df = pd.DataFrame(embedding_label, columns=colum_name)
    # # print(df)
    #
    # _, _, _, correct_matrix = test()
    # print(correct_matrix)
    # print(np.mean(var_matrix[1:, :], axis=1))
    # save_dict = {"var": 1 / np.mean(1 / var_matrix[1:, :], axis=1), "correct": correct_matrix[1:]}
    # save_df = pd.DataFrame(save_dict)
    #
    # save_df.to_csv(f"/home/khp/{dataset}_advtrain.csv")

    print(complexity(df, 10))
