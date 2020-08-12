from sklearn.neighbors import KNeighborsClassifier

x = [[1, 1], [2, 3], [0, 9], [0, 5]]
y = [1, 1, 0, 0]

estimator = KNeighborsClassifier(n_neighbors=2)
estimator.fit(x, y)
print(estimator.classes_)
predict = estimator.predict([[0, 5]])
print(predict)


"""
欧氏距离：通过距离平方值来进行计算
曼哈顿距离：通过距离的绝对值进行计算
切比雪夫距离：维度的最大值进行计算
闵可夫斯基距离：
    p=1时，就是曼哈顿距离
    p=2时，就是欧氏距离
    p=无穷大时，就是切比雪夫距离
小结：上述四种计算距离方式都是将单位相同看待了，所以计算过程不是很科学。

标准化欧氏距离：在计算过程中添加标准差，对量纲数据进行处理
余弦距离：cos思想计算
汉明距离：一个字符串到另一个字串需要变化都少个字母
杰卡德距离：通过交并集计算
马氏距离：通过样本分布计算
"""