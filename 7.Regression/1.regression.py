from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

x = [[80, 86],
[82, 80],
[85, 78],
[90, 90],
[86, 82],
[82, 90],
[78, 80],
[92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

score_data = pd.DataFrame(x, columns=['english', 'math'])
print(score_data)
score_data['mean_score'] = y
print(score_data)


def plot_iris(data, col1, col2):
    sns.lmplot(data=data, x=col1, y=col2, hue="mean_score", fit_reg=True) 
    #data:数据值，x,y:具体x轴，y轴的数据索引值，hue：目标值，fit_reg:是否进行线性拟合
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('mean_score data distribute')
    plt.show()

plot_iris(score_data, 'english', 'math')


# 实例化API
estimator = LinearRegression()
# 使用fit方法进行训练
estimator.fit(x,y)

coef = estimator.coef_
intercept = estimator.intercept_
print(coef, intercept)

print(estimator.predict([[100, 80]]))