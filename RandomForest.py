import pandas as pd 
import numpy as np


class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples=3, max_features=5, min_impurity_decrease=0.0, max_leaf_nodes=float('inf')):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
    def _gini(self, y): #gini for multiclass
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    def random_tree(self, X, y, depth, max_depth, min_samples, max_features, leaf_count):
        if depth >= max_depth or len(y) <= min_samples or len(np.unique(y)) == 1:
            #either too deep, too little examples, too pure
            return {'class': np.bincount(y).argmax()}
        if self.max_leaf_nodes and leaf_count[0] >= self.max_leaf_nodes:
            return {'class': np.bincount(y).argmax()}
        
        if max_features is None:
            max_features = int(np.sqrt(X.shape[1]))  # Default: sqrt(total features)
    
        feature_indices = np.random.choice(X.shape[1], max_features, replace=False)
        
        best_gini = float('inf')
        best_split = None
        parent_gini = self._gini(y)

        for feature in feature_indices:
            for threshold in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                weighted_gini = (left_mask.sum() * self._gini(y[left_mask]) + 
                                right_mask.sum() * self._gini(y[right_mask])) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = (feature, threshold, left_mask, right_mask)

        if best_split is None:
            return {'class': np.bincount(y).argmax()}

        impurity_decrease = parent_gini - best_gini
        if impurity_decrease < self.min_impurity_decrease:
            return {'class': np.bincount(y).argmax()}
        feature, threshold, left_mask, right_mask = best_split        
        leaf_count[0] += 2

        left_tree = self.random_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples, max_features, leaf_count)
        right_tree = self.random_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples, max_features, leaf_count) 

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), len(X), replace=True)
            tree = self.random_tree(X[idx], y[idx], 0, self.max_depth, self.min_samples, self.max_features, [0])
            self.trees.append(tree)
    


    def predict_one(self, tree, X):
        if 'class' in tree:
            return tree['class']
        
        if X[tree['feature']] <= tree['threshold']:
            return self.predict_one(tree['left'], X)
        else:
            return self.predict_one(tree['right'], X)
        
    def predict_forest(self, trees, X):
        predict = []
        for tree in trees:
            preds = [self.predict_one(tree, x) for x in X]
            predict.append(preds)
        
        predict = np.array(predict)
        return [np.bincount(predict[:, i]).argmax() for i in range(len(X))]
    
    def score(self, X, y):
        y_pred = self.predict_forest(self.trees, X)
        accuracy = np.mean(y_pred == y)
        return accuracy


df = pd.read_csv('rf_dataEF&RF.csv')
X = df[[col for col in df.columns if col != 'target']].values
y = df['target'].values

# Manual split
np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))

X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

guy = RandomForest()
guy.fit(X_train, y_train)
print(guy.score(X_test, y_test))



import matplotlib.pyplot as plt


plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0d0d1a')

# ============================================================
# Panel 1: Class Distribution
# ============================================================
ax = axes[0]
ax.set_facecolor('#0d0d1a')
classes, counts = np.unique(y_train, return_counts=True)
colors_bar = ['#00d4ff', '#ff6b6b']
ax.bar(classes, counts, color=colors_bar, edgecolor='white', linewidth=1.5, width=0.4)
ax.set_xlabel('Class', color='white', fontsize=12)
ax.set_ylabel('Count', color='white', fontsize=12)
ax.set_title('Class Distribution', fontsize=14, color='white', fontweight='bold')
ax.tick_params(colors='white')
ax.grid(True, alpha=0.1, color='white')
for i, count in enumerate(counts):
    ax.text(classes[i], count + max(counts)*0.02, str(count), ha='center', color='white', fontweight='bold')

# ============================================================
# Panel 2: Accuracy vs Trees
# ============================================================
ax = axes[1]
ax.set_facecolor('#0d0d1a')
n_list = [1, 5, 10, 20, 50, 100, 200]
train_scores, test_scores = [], []
for n in n_list:
    m = RandomForest(n_trees=n, max_depth=10, max_features=5)
    m.fit(X_train, y_train)
    train_scores.append(m.score(X_train, y_train))
    test_scores.append(m.score(X_test, y_test))

ax.plot(n_list, train_scores, 'o-', color='#00d4ff', linewidth=2, markersize=8, label='Train', markeredgecolor='white', markeredgewidth=0.5)
ax.plot(n_list, test_scores, 'o-', color='#ff6b6b', linewidth=2, markersize=8, label='Test', markeredgecolor='white', markeredgewidth=0.5)
ax.fill_between(n_list, train_scores, test_scores, alpha=0.08, color='#00d4ff')
ax.set_xlabel('Number of Trees', color='white', fontsize=12)
ax.set_ylabel('Accuracy', color='white', fontsize=12)
ax.set_title('Accuracy vs Ensemble Size', fontsize=14, color='white', fontweight='bold')
ax.legend(facecolor='#1a1a3e', edgecolor='white', fontsize=10)
ax.grid(True, alpha=0.1, color='white')
ax.tick_params(colors='white')

# ============================================================
# Panel 3: Model Summary
# ============================================================
ax = axes[2]
ax.set_facecolor('#0d0d1a')
ax.axis('off')

summary = f"""RANDOM FOREST — FROM SCRATCH
—————————————————————————————————
Trees:                {guy.n_trees}
Max Depth:            {guy.max_depth}
Min Samples:          {guy.min_samples}
Max Features:         {guy.max_features}
Min Impurity Dec.:    {guy.min_impurity_decrease}
Max Leaf Nodes:       {guy.max_leaf_nodes}

Features:             {X_train.shape[1]}
Training Samples:     {len(X_train)}
Test Samples:         {len(X_test)}

Train Accuracy:       {guy.score(X_train, y_train):.4f}
Test Accuracy:        {guy.score(X_test, y_test):.4f}
Train-Test Gap:       {guy.score(X_train, y_train) - guy.score(X_test, y_test):.4f}

BUILT ENTIRELY WITH NUMPY
NO SCIKIT-LEARN"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=12, 
        color='white', fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8, edgecolor='#00d4ff'))

plt.suptitle('RANDOM FOREST — COMPLETE DASHBOARD', fontsize=20, color='white', 
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('random_forest_dashboard.png', dpi=300, bbox_inches='tight', facecolor='#0d0d1a')
plt.show()