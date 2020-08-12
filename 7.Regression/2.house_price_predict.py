from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import joblib


def linear_model1():
    """
    线性回归:正规方程
    :return:None
    """
    # 1.获取数据
    data = load_boston()

    # 2.数据集划分
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=22)

    # 3.特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.机器学习-线性回归(正规方程)
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 5.1 获取系数等值
    y_predict = estimator.predict(x_test)
    print("模型的准确率:", estimator.score(x_test, y_test))
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)

    # 5.2 评价
    # 均方误差
    error = mean_squared_error(y_test, y_predict)
    print("误差为:\n", error)


def linear_model2():
    """
    线性回归:梯度下降法
    :return:None
    """
    # 1.获取数据
    data = load_boston()

    # 2.数据集划分
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=22)

    # 3.特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.机器学习-随机梯度下降法
    estimator = SGDRegressor(max_iter=1000, learning_rate="constant", eta0=0.001)
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 5.1 获取系数等值
    y_predict = estimator.predict(x_test)
    print("模型的准确率:", estimator.score(x_test, y_test))  #回归问题评价指标没有准确率
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)

    # 5.2 评价
    # 均方误差
    error = mean_squared_error(y_test, y_predict)
    print("误差为:\n", error)


def linear_model3():
    """
    线性回归:岭回归
    :return:
    """
    # 1.获取数据
    data = load_boston()

    # 2.数据集划分 
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=22)

    # 3.特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.机器学习-线性回归(岭回归)
    # estimator = Ridge(alpha=1, fit_intercept=True, )
    # estimator = RidgeCV(alphas=(0.1, 1, 10))
    # estimator.fit(x_train, y_train)


    # # 4.2 模型保存
    # joblib.dump(estimator, "./7.Regression/house_price_test.pkl")

    # 4.3 模型加载
    estimator = joblib.load("./7.Regression/house_price_test.pkl")


    # 5.模型评估
    # 5.1 获取系数等值
    y_predict = estimator.predict(x_test)
    score = estimator.score(x_test, y_test)
    print("模型的准确率：", score)
    print("预测值为:\n", y_predict)
    print("模型中的系数为:\n", estimator.coef_)
    print("模型中的偏置为:\n", estimator.intercept_)

    # 5.2 评价
    # 均方误差
    error = mean_squared_error(y_test, y_predict)
    print("误差为:\n", error)


if __name__ == "__main__":
    # linear_model1()
    # linear_model2()
    linear_model3()

"""
线性回归的损失和优化：
    1.损失：最小二乘法
    2.优化：
        2.1正规方程
        2.2梯度下降法
    3.正规方程：一蹴而就
        利用矩阵的逆和转置进行进一步求解，只适合样本和特征比较少的情况
    4.梯度下降法
        举例：
            山：可微分的函数
            山底：函数的最小值
        梯度的概念：
            单变量：切线
            多变量：向量
        梯度下降法中关注的两个参数：
            1.学习率
            2.负号：梯度的方向是函数在给定点上升最快的方向，那么梯度的反方向就是函数在给定点下降最快的方向
"""