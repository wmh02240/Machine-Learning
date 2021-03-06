'''
一、机器学习工作流程：
    1.获取数据
    2.数据基本处理---》》》对数据的异常值缺失值等进行处理
    3.特征工程---》》》定义：把数据转换成机器更容易识别的数据  目的：数据和特征决定了机器学习的上线，而模型和算法只是逼近这个上限而已。  包含内容：特征提取、特征预处理、特征降维  
    4.模型训练（机器学习）---》》》选择合适的算法对模型进行训练
    5.模型评估---》》》队训练好的模型进行评估

二、机器学算法分类介绍
    1.监督学习
        定义：输入数据是由特征值和目标值组成
            函数输出是连续值则是回归问题--->>>房价预测问题
            函数输出值是离散的则是分类问题--->>>根据肿瘤形态判断是否是恶性或者良性
        案例：猫狗分类、房屋价格预测等
        代表算法：
            分类：K-近邻算法（KNN）、贝叶斯算法、决策树与随机森林、逻辑回归、神经网络
            回归：线性回归、岭回归

    2.无监督学习
        定义：输入数据是由特征值组成。输入数据没有被标记，也没有确定的结果。样本数据类别未知，需要对样本之间的相似度对样本进行分类（聚类），试图使类内差异最小化，类间差异最大化。
        目的：发现潜在结构
        案例：物以类聚、人以群分
        代表算法：K-Means、降维

    3.半监督学习
        定义：输入数据有特征值，但是一部分有目标值，一部分没有。
        已知：训练样本数据和待分类类别
        未知：训练样本有无标签均可
        应用：训练数据过多时监督学习效果不能满足需求，因此用来增强效果

    4.强化学习
        定义：实质上是make decisions问题，及自动进行决策，并且可以做连续决策。
        强化学习是一个动态的过程，上一步的数据输出是下一步的数据输入。
        四要素：agent、action、environment、reward
        案例：学下棋
        代表算法：马尔科夫决策、动态规划

三、模型评估
    1.分类问题模型评估
        1.1准确率：预测正确样本数与总样本数的比值
        1.2精确率：正确预测为正占全部预测为正的比值
        1.3召回率：正确预测为正占全部正样本的比值
        1.4F1-Score:评估模型的稳健性
        1.5AUC指标：用于评估样本不均衡的情况

    2.回归问题模型评估
        2.1均方根误差RMSE
        2.2相对平方误差RSE
        2.3平均绝对误差MAE
        2.4相对绝对误差RAE
        2.5决定系数R^2
'''
