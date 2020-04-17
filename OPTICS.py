from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

########################################
#             产生样本点
########################################
np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [0, 5] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

########################################
#         OPTICS与DBSCAN对比
########################################
clust = OPTICS(min_samples = 110, xi = 0.05, min_cluster_size = 0.05)

# Run the fit
clust.fit(X)

labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

########################################
#             绘图
########################################
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# Reachability plot
plt.figure(1, figsize = (8, 4))
plt.plot(space, reachability, 
  'k-', linewidth = 2.5, alpha = 1, label = 'Noisy Points')
low = np.zeros(len(space))
plt.fill_between(space, reachability, low, facecolor = 'k')

colors = ['m', 'b', 'r', 'g', 'c']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    if len(Xk) != 0:
      print('Cluster ', klass, ': ', len(Xk))
      Rk = reachability[labels == klass]
      plt.plot(Xk, Rk, color + '-', 
        linewidth = 2.5, alpha = 1, label = 'Cluster ' + str(klass))
      low = np.zeros(len(Xk))
      plt.fill_between(Xk, Rk, low, facecolor = color)

plt.plot(space, np.full_like(space, 2., dtype=float), 'k--', alpha = 0.5)
plt.plot(space, np.full_like(space, 0.5, dtype=float), 'k--', alpha = 0.5)
plt.ylabel('Reachability ($\epsilon$ distance)')
plt.title('Reachability Plot')
plt.legend()
plt.show()

# OPTICS
plt.figure(2, figsize = (6, 4))
colors = ['m.', 'b.', 'r.', 'g.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    print('Cluster ', klass, ': ', len(Xk))
    plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
plt.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
plt.title('Automatic Clustering\nOPTICS')

plt.show()
########################################
#             对比绘图
########################################
plt.figure(2, figsize = (12, 4))
plt.subplot(1, 2, 1)
# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = X[labels_050 == klass]
    print('Cluster ', klass, ': ', len(Xk))
    plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
plt.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
plt.title('Clustering at $\epsilon$ = 0.5\nDBSCAN')
plt.xlabel('(a)')

# DBSCAN at 2.
plt.subplot(1, 2, 2)
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    print('Cluster ', klass, ': ', len(Xk))
    plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
plt.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
plt.title('Clustering at $\epsilon$ = 2\nDBSCAN')
plt.xlabel('(b)')
plt.show()
