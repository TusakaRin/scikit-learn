import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


data = pd.read_csv('ex3.txt', sep='\t', header=None)
# tree = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=1e-6, min_samples_leaf=4)
tree = DecisionTreeClassifier(criterion='cmse', min_impurity_decrease=1e-6, min_samples_leaf=4)
y = (data.values[:, -1] != 0).astype('int64')
tree.fit(data.values[:, :-1], y)
plot_tree(tree)
plt.show()
