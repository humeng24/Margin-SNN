import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/home/khp/fashion_mnist.csv")
true_ls = df.loc[df.correct == 1, "var"]
false_ls = df.loc[df.correct == 0, "var"]
print(len(true_ls))
print(len(false_ls))
plt.rcParams.update({"font.size":16})
sns.distplot(true_ls, bins=10, hist=True, kde=False)
# 添加x轴和y轴标签
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Var", fontsize=15)
plt.ylabel("Number", fontsize=15)
plt.title("Examples of unsuccessful attacks(FMnist)")
plt.tight_layout()  # 处理显示不完整的问题
plt.savefig("/home/khp/unsuccessful_fmnist.png")
plt.show()

plt.rcParams.update({"font.size":16})
sns.distplot(false_ls, bins=10, hist=True, kde=False)
# 添加x轴和y轴标签
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Var", fontsize=15)
plt.ylabel("Number", fontsize=15)
plt.title("Examples of successful attacks(FMnist)")
plt.tight_layout()  # 处理显示不完整的问题
plt.savefig("/home/khp/successful_fmnist.png")
plt.show()
