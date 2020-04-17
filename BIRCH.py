import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch
from sklearn import metrics

##############################################
#                     生成数据
##############################################
# X为样本特征，Y为样本簇类别
# 共1000个样本，每个样本2个特征
# 共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
# 每个类别的方差分别为0.4, 0.3, 0.4, 0.3
X, y = make_blobs(n_samples=1000, n_features=2, 
	centers = [[-1,-1], [0,0], [1,1], [2,2]], 
	cluster_std = [0.4, 0.3, 0.4, 0.3], random_state=9)

##############################################
#                     开始聚类
##############################################
# 尝试多个threshold取值，和多个branching_factor取值
param_grid = {'threshold': [0.1, 0.3, 0.5],
'branching_factor': [10, 20, 50]}  
CH_all = []
X_all = []
y_all = []
paras_all = []
# 网格搜索方法
print("Threshold\tBranching factor\tMetrics")
for threshold in param_grid['threshold']:
    for branching_factor in param_grid['branching_factor']:
        clf = Birch(n_clusters = 4, 
        	threshold = threshold, branching_factor = branching_factor)
        clf.fit(X)
        X_all.append(X)
        y_pred = clf.predict(X)
        y_all.append(y_pred)
        CH = metrics.calinski_harabaz_score(X, y_pred)
        CH_all.append(CH)
        paras_all.append([threshold, branching_factor])
        print(threshold, "\t", branching_factor, "\t", '%.4f'%CH)

##############################################
#                     择优
##############################################
CH_max = max(CH_all)
idx = CH_all.index(CH_max)
X_max = X_all[idx]
y_max = y_all[idx]

##############################################
#                     绘图
##############################################
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y)
plt.title('Original Cluster Plot')
plt.xlabel('(1)')

plt.subplot(1, 2, 2)
plt.scatter(X_max[:, 0], X_max[:, 1], marker = 'o', c = y_max)
plt.title('Result (Threshold = ' + str(paras_all[idx][0])
 + ', BF = ' + str(paras_all[idx][1]) + ')')
plt.xlabel('(2)')


plt.show()