import pymongo
from pymongo import MongoClient
from module import *
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# client = MongoClient('172.29.5.158', 27017)
# mongo_auth = client.admin
# mongo_auth.authenticate('admin', 'passwd')
# mydb = client["clean_feature"]
# mycol = mydb['resnet_mnist_1']
# data = pd.DataFrame(list(mycol.find()))

# python远程连接mongodb
class Mongo(object):
    def __init__(self):
        self.client = MongoClient('172.29.5.27', 27017)
        self.mongo_auth = self.client.admin
        self.mongo_auth.authenticate('admin', 'passwd')

    def get_database(self, db):
        return self.client[db]

    def get_collection(self, db, col):
        db = self.get_database(db)
        return db[col]

    def get_client(self):
        return self.client

    @staticmethod
    def get_df(data_obj):
        return pd.DataFrame(list(data_obj.find()))


# 使用hook捕获某一层的输出，一般用在读取保存好的模型以及预训练模型上，自定义模型可直接在forward处用self.xx保存该处输出
class layerActivation(object):
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hookfn)

    # 获取features某一层的输出output
    def hookfn(self, model, input, output):
        self.features = output

    # 删除hook
    def remove(self):
        self.hook.remove()


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# torch读取预训练模型参数
def update_model(model_1, model_2):
    model_dict = model_1.state_dict()
    state_dict = model_2.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if
                  (k in model_dict.keys()) and (np.shape(model_dict[k]) == np.shape(v))}
    model_dict.update(state_dict)
    model_1.load_state_dict(model_dict)
    return model_1


# torch读取预训练模型参数
def load_model(model_s, state):
    model_dict = model_s.state_dict()
    state_dict = torch.load(state)
    state_dict = {k: v for k, v in state_dict.items() if
                  (k in model_dict.keys()) and (np.shape(model_dict[k]) == np.shape(v))}
    model_dict.update(state_dict)
    model_s.load_state_dict(model_dict)
    return model_s


def fuzziness(param_ls):
    rho = 1 / (1 + np.abs(param_ls))
    fzi = rho * np.log(rho + 1e-10) + (1 - rho) * np.log((1 - rho) + 1e-10)
    return - np.mean(fzi)


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # write logs into files
    with open(log_path, 'a+') as f:
        f.write(log_info + '\n')


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        # 计算测试样本的总数
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
