import argparse
import os.path
import os

import advertorch.attacks as attacks

from module import *
from model import *
from evaluation.marginsnn.complexity import *
import utils.utils as utils
from model.resnet import *
from adver.adver_attack.attack import attack
from datasets.load_dataset import data_loader

from torch.utils.tensorboard import SummaryWriter



# model
class net(nn.Module):
    def __init__(self, backbone, classes, latent_dim):
        super(net, self).__init__()
        self.backbone = backbone
        self.classes = classes
        self.label_embedding = nn.Embedding(classes, node_num)
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)

    # re-sampling
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


class margin_loss(nn.Module):
    def __init__(self, beta, alpha):
        super(margin_loss, self).__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, mu, logvar, label_embedding, output, target):
        if distance_metric == 'cosine':
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

            label_distance = distance_matrix * logits_mask
            return base_loss + self.beta * torch.mean(label_distance) + self.alpha * kl_loss
        elif distance_metric == 'euclidean':
            classes = label_embedding.shape[0]
            label_distance = (label_embedding.unsqueeze(1).repeat((1, classes, 1)) - label_embedding.repeat(classes, 1, 1)).pow(2).sum(dim=2)
            base_loss = F.cross_entropy(output, target)
            kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
            kl_loss = kl_loss.sum(dim=1).mean()
            return base_loss - self.beta * torch.mean(label_distance) + self.alpha * kl_loss


def train(train_loader, model, optimizer, scheduler, loss_function=None):
    model.train()
    train_loss = 0
    correct = 0

    for i, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        mu, logvar = model.mu.clone(), model.logvar.clone()

        label_embedding = model.y_embedding
        loss = loss_function(mu, logvar, label_embedding, output, target)

        train_loss += loss.item() * data.shape[0]
        loss.backward(retain_graph=True)
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    scheduler.step()

    print("Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            train_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)
    ))
    return train_loss / len(train_loader.dataset), correct/ len(train_loader.dataset)*100


def test(eps, nb_iter, eps_iter, model, test_loader):

    adversary = attacks.LinfPGDAttack(model, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter)
    model.eval()

    test_clnloss = 0
    clncorrect = 0

    test_advloss = 0
    advcorrect = 0
    robustness = 0

    for clndata, target in test_loader:
        clndata = clndata.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(clndata)

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

    test_clnloss /= len(test_loader.dataset)

    # 干净样本准确率
    print('Test set: avg cln loss: {:.4f},'
          ' cln acc: {}/{} ({:.2f}%)'.format(
          test_clnloss, clncorrect, len(test_loader.dataset),
          100. * clncorrect / len(test_loader.dataset)))
    test_advloss /= len(test_loader.dataset)

    # 对抗样本准确率
    print('Test set: avg adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.2f}%)'.format(
        test_advloss, advcorrect, len(test_loader.dataset),
        100. * advcorrect / len(test_loader.dataset)))

    return test_clnloss, clncorrect / len(test_loader.dataset)*100, advcorrect / len(test_loader.dataset)*100


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


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    create_path(save_path)
    create_path(writer_path)
    create_path(log_path)

    writer = SummaryWriter(writer_path)       # 记录损失精度信息
    g = image_complexity(img_complexity)

    # 日志设置
    file_ls = os.walk(log_path).__next__()[2]
    if not file_ls:
        log_name = f'{model_s}_{dataset}_label_embedding_1.txt'
    else:
        pattern = re.compile("[_.]")
        file_ls = [int(pattern.split(file)[-2]) for file in file_ls]
        idx = max(file_ls) + 1
        log_name = f"{model_s}_{dataset}_label_embedding_{idx}.txt"

    log = os.path.join(log_path, log_name)
    os.mknod(log)

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
    train_loader, test_loader = data_selection(dataset, normalize=use_normalization, test_batch=500)

    print_log("-"*20 + " hyperparameters" + '-'*20, log)
    print_log(f"\nArgs: {args}", log)
    print_log(f"\nSeed: {seed}", log)
    print_log(f"\nEpochs: {epochs}", log)
    print_log("\n" + data_selection.type, log)
    print_log(f"\nLearning rate: {lr}", log)
    print_log(f"\nTrain batch size: {data_selection.train_batch}", log)
    print_log(f"\nTest batch size: {data_selection.test_batch}", log)
    print_log(f"\nSelection {img_complexity} for image complexity calculation", log)

    train_batch_size = data_selection.train_batch

    print_log(f"\nUsing {str(device)} for training\n", log)

    utils.setup_seed(seed)                                                      # 设置随机数种子

    print_log("-"*20 + " model params " + '-'*20, log)

    # 选择模型
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
        raise ValueError("Have no model {}".format(model_s))

    model = net(backbone_model, nb_classes, latent_dim=node_num)

    print_log("\n" + str(model) + "\n", log)

    model = model.to(device)

    loss_function = margin_loss(beta=args.beta, alpha= args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=lr)     # Adam
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)
    print_log("-" * 20 + " Training process" + '-' * 20, log)

    print_log('\nTraining start!\n', log)

    start_time = time.time()

    accuracy = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        my_col = f"epoch_{epoch}"
        # feature_collection = my_mongo.get_collection(my_db, my_col)
        train_loss, train_acc = train(train_loader, model, optimizer, scheduler, loss_function)
        test_loss, test_acc, adv_acc = test(eps, nb_iter, eps_iter, model, test_loader)

        # saving parameters
        for name, param in model.named_parameters():
            if 'bn' not in name:
                writer.add_histogram(name, param, epoch)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('adv_acc', adv_acc, epoch)

        print_log("-" * 5 + f" epoch_{epoch} " + "-" * 5, log)
        print_log(f"\nTrain loss = {train_loss}, train accuracy = {np.round(train_acc,2)}%", log)
        print_log(f"\nTest loss = {test_loss}, test accuracy = {np.round(test_acc,2)}%, adver accuracy = {np.round(adv_acc,2)}%", log)

        if adv_acc > accuracy:
            model_name = f"{model_s}_{dataset}_{node_num}nodenum_{lr}lr_{seed}seed_{args.beta}beta_{args.alpha}alpha_{data_selection.type.split(' ')[1]}_normalization_le_margin_kl.pth"
            accuracy = adv_acc
            torch.save(model.state_dict(), os.path.join(save_path, model_name))
            print_log(f"\nSaving model state dict in {os.path.abspath(os.path.join(save_path, model_name))}\n", log)

    print_log(f"\nTraining done!\n", log)

    writer.close()
    end_time = time.time()

    print_log(f"\nTotal training time: {end_time - start_time}", log)

    print_log("-" * 20 + " Result " + '-' * 20, log)
    # 记录精度最高的模型的各层模糊度
    print_log('\n'+ '-'*5 + f"Compute fuzziness of each layer" + '-'*5 + '\n', log)
    model.load_state_dict(torch.load(save_path + model_name))
    compute_fuzz(model, f=log, if_to_f=True)

    # 记录对抗样本准确率
    print_log('\n' + '-'*5 + f"Compute adversarial accuracy of trained model" + '-'*5 + '\n', log)
    print_log(f'\nEpsilon of attack: {eps}', log)

    # FGSM攻击
    print_log("-"*5 + f" Using FGSM attack " + "-"*5, log)
    fgsm = attacks.FGSM(model, eps=eps)
    attack(adver_method=fgsm, model=model, data_loader=test_loader, device=device, fp=log)

    print_log("-" * 5 + f" Using PGD attack " + "-" * 5, log)
    pgd = attacks.LinfPGDAttack(model, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter)
    attack(adver_method=pgd, model=model, data_loader=test_loader, device=device, fp=log)

    print_log("-" * 5 + f" Using BIM attack " + "-" * 5, log)
    bim = attacks.LinfBasicIterativeAttack(model, eps=eps, eps_iter=eps_iter)
    attack(adver_method=bim, model=model, data_loader=test_loader, device=device, fp=log)

    print_log("-" * 5 + f" Using MIM attack " + "-" * 5, log)
    mim = attacks.MomentumIterativeAttack(model, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter)
    attack(adver_method=mim, model=model, data_loader=test_loader, device=device, fp=log)

    print_log("-" * 5 + f" Using CW attack " + "-" * 5, log)
    cw = attacks.CarliniWagnerL2Attack(model, num_classes=nb_classes, confidence=0,
                learning_rate=cw_lr, max_iterations=1000, initial_const=0.001)
    attack(adver_method=cw, model=model, data_loader=test_loader, device=device, fp=log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Margin-SNN')
    parser.add_argument('--dataset', default='cifar10', help='dataset: mnist | fashion_mnist | cifar10 | cifar100 | svhn')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=400, help='training epochs')
    parser.add_argument('--model-s', type=str, default="resnet18", help='Backbone: lenet | resnet | vgg11 | resnet_dul | resnet34 | resnet18')
    parser.add_argument('--distance-metric', type=str, default="euclidean", help='euclidean | cosine')
    parser.add_argument('--img-complexity', type=str, default="g0", help='g0 | g1 | g2')
    parser.add_argument('--use-normalization', action="store_true")
    parser.add_argument('--cuda', type=int, default=0, help='CUDA')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--latent-dim', type=int, default=256, help='dim. of latent space of resampling')
    parser.add_argument('--alpha', type=float, default=0.01, help='KL item')
    parser.add_argument('--beta', type=float, default=0.5, help='Margin item')
    parser.add_argument('--num-cluster', type=int, default=3, help='number of clusters')
    parser.add_argument('--base-dir', type=str, default="../results-cifar10/", help='all results are put in the subfolder')
    args = parser.parse_args()
    # parameter
    lr = args.lr
    epochs = args.epochs
    seed = args.seed
    start_epoch = 1

    node_num = args.latent_dim
    dataset = args.dataset
    model_s = args.model_s
    distance_metric = args.distance_metric
    img_complexity = args.img_complexity
    use_normalization = args.use_normalization
    cuda = args.cuda
    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')       # gpu

    save_path = os.path.join(args.base_dir, "save_model", dataset, "label_embedding_margin_kl")
    log_path = os.path.join(args.base_dir, "log", dataset, "label_embedding_margin_kl")
    writer_path = os.path.join(args.base_dir, "drawing", model_s+"_"+dataset + "_label_embedding_margin_kl")

    # setting
    if 'mnist' in dataset:
        input_cl = 1
    elif 'cifar' in dataset:
        input_cl = 3
    elif dataset == 'svhn':
        input_cl = 3

    if dataset == 'cifar100':
        nb_classes = 100
    else:
        nb_classes = 10
    main()





