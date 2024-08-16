import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# def visualize_model_output(data, data_labels):
#     print('draw visualize_model_output')
#     # 标准化数据
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
    
#     # 使用t-SNE进行降维
#     tsne = TSNE(n_components=2, random_state=42)
#     data_tsne = tsne.fit_transform(data_scaled)
    
#     # 绘图
#     plt.figure(figsize=(12, 6))
    
#     # 图 (a)
#     plt.subplot(1, 2, 1)
#     scatter_a = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=data_labels, cmap='coolwarm', alpha=0.7)
#     plt.title('(a)')
#     plt.axis('off')
    
#     # 图 (b) - 这里假设可能需要不同的处理，可以根据需要修改
#     plt.subplot(1, 2, 2)
#     scatter_b = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=data_labels, cmap='coolwarm', alpha=0.7)
#     plt.title('(b)')
#     plt.axis('off')
    
#     plt.colorbar(scatter_a, ax=plt.gca(), orientation='vertical', fraction=0.02)
    
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('theia_data.png')


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

# Random state.
RS = 42


# We import seaborn to make nice plots.

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def scatter(x, colors):
    x = TSNE(random_state=RS).fit_transform(x)
    # We choose a color palette with seaborn.
    palette = np.array([(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725)])

    # We create a scatter plot.
    f = plt.figure(figsize=(4, 4))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20,
                    c=palette[colors.astype(np.int64)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('vae_hidost_mal_tsne.png')
    plt.show()

    return f, ax, sc

"""
X: raw data, 
y: labels
"""
# X = np.concatenate((x_train_mal, mal_pred2), axis=0)
# X = X.reshape(len(X), 3514)
# y = np.concatenate((np.array([1] * len(x_train_mal)), np.array([0] * len(mal_pred2))), axis=0)

# digits_proj = TSNE(random_state=RS).fit_transform(X)

# scatter(digits_proj, y)
# foo_fig = plt.gcf() # 'get current figure'
# foo_fig.savefig('vae_hidost_mal_tsne.pdf')
# plt.show()


# # 示例数据
data = np.random.rand(100, 50)  # 假设数据有100个样本，每个样本有50维特征
data_labels = np.random.randint(0, 2, 100)  # 假设二分类标签
scatter(data, data_labels)
# visualize_model_output(data, data_labels)
