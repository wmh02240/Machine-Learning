from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1.获取数据集
# 获取鸢尾花数据集
iris = load_iris()
# print("鸢尾花数据集的返回值：\n", iris)    # 返回值是一个继承自字典的Bench
# print("鸢尾花的特征值:\n", iris["data"])
# print("鸢尾花的目标值：\n", iris.target)
# print("鸢尾花特征的名字：\n", iris.feature_names)
# print("鸢尾花目标值的名字：\n", iris.target_names)
# print("鸢尾花的描述：\n", iris.DESCR)

# 把数据转换成dataframe的格式
iris_d = pd.DataFrame(iris['data'], columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_d['Species'] = iris.target
print(iris_d)

def plot_iris(iris, col1, col2):
    sns.lmplot(data=iris, x=col1, y=col2, hue="Species", fit_reg=True) 
    #data:数据值，x,y:具体x轴，y轴的数据索引值，hue：目标值，fit_reg:是否进行线性拟合
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('iris data distribute')
    plt.show()


# plot_iris(iris_d, 'Sepal_Length', 'Sepal_Width')
# plot_iris(iris_d, 'Sepal_Width', 'Petal_Length')
# plot_iris(iris_d, 'Sepal_Length', 'Petal_Length')

# plot_iris(iris_d, 'Sepal_Width', 'Sepal_Length')
# plot_iris(iris_d, 'Petal_Length', 'Sepal_Width')
# plot_iris(iris_d, 'Petal_Length', 'Sepal_Length')

# 2.数据基本处理
# 对鸢尾花数据集进行分割
# 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
# print(iris.data, iris.target)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state=22)
print("x_train:\n", x_train.shape)
# 随机数种子
# x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
# x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)
# print("如果随机数种子不一致：\n", x_train == x_train1)
# print("如果随机数种子一致：\n", x_train1 == x_train2)

# 3、特征工程：标准化
# 特征预处理定义:通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程.
# 为什么我们要进行归一化/标准化？
# 特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些算法无法学习到其它的特征. 
# 归一化后所有数据属于同一量纲级别。
# 1.归一化
#   1.1实例化一个转换器
# trnsfer = MinMaxScaler(feature_range=(0,1))
# #   1.2调用fit_transform方法
# minmax_data = trnsfer.fit_transform(x_train)
# print("经过归一化处理之后的数据：", minmax_data)

# 归一化的缺点：注意最大值最小值是变化的，另外，最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。怎么办？
# 2.标准化:通过对原始数据进行变换把数据变换到均值为0,标准差为1范围内
#   2.1实例化一个转换器
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
# print("标准化的结果：", x_train)
# print("每一列数据的均值：", x_train.mean())
# print("每一列数据的方差：", x_train.var())
x_test = transfer.transform(x_test)


# 4、机器学习(模型训练)
estimator = KNeighborsClassifier(n_neighbors=5)

# 模型选择与调优——网格搜索和交叉验证
# 准备要调的超参数
param_dict = {"n_neighbors": [1, 3, 5, 7, 9]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3, n_jobs=2)
# 对估计器的指定参数值进行详尽搜索
# estimator：估计器对象
# param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
# cv：指定几折交叉验证
# fit：输入训练数据
# score：准确率
# 结果分析：
# bestscore_:在交叉验证中验证的最好结果
# bestestimator：最好的参数模型
# cvresults:每次交叉验证后的验证集准确率结果和训练集准确率结果

estimator.fit(x_train, y_train)


# 5、模型评估
# 方法1：比对真实值和预测值
y_predict = estimator.predict(x_test)

print("预测结果为:\n", y_predict)
print("比对真实值和预测值：\n", y_predict == y_test)

# 方法2：直接计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)


print("最好的参数模型：\n", estimator.best_estimator_, '\n')
print("在交叉验证中验证的最好结果：\n", estimator.best_score_, '\n')
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)