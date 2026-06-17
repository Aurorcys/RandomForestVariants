import pandas as pd 
import numpy as np


class TotallyRandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
    def random_tree(self, X, y, depth, max_depth, min_samples):
        if depth >= max_depth or len(y) <= min_samples or len(np.unique(y)) == 1:
            #either too deep, too little examples, too pure
            return {'class': np.bincount(y).argmax()}
        
        feature = np.random.randint(0, X.shape[1])

        min_val, max_val = X[:, feature].min(), X[:, feature].max()
        threshold = np.random.uniform(min_val, max_val)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {'class': np.bincount(y).argmax()}
        
        left_tree = self.random_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples)
        right_tree = self.random_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples)

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
            tree = self.random_tree(X[idx], y[idx], 0, self.max_depth, self.min_samples)
            self.trees.append(tree)

    """
    structure:
    tree = {
        'feature': 2,
        'threshold': 4.0,
        'left': {
            'feature': 0,
            'threshold': 1.5,
            'left': {'class': 0},
            'right': {'class': 1}
        },
        'right': {'class': 1}
    }
    """
    


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


df = pd.read_csv('rf_data.csv')
X = df[[col for col in df.columns if col != 'target']].values
y = df['target'].values

# Manual split
np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))

X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

guy = TotallyRandomForest()
guy.fit(X_train, y_train)
print(guy.score(X_test, y_test))

        
        
        



        

