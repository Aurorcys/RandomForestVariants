import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


class ExtremelyRandomForestFuture:
    def __init__(self, n_trees=100, max_depth=10, min_samples=3, max_features=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
    def gini(self, y): #gini for multiclass
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    def random_split_gini(self, X, y):
        feature = np.random.randint(0, X.shape[1])
        min_val, max_val = X[:, feature].min(), X[:, feature].max()
        threshold = np.random.uniform(min_val, max_val)

        left_mask = X[:, feature] <= threshold #True and false matrix
        right_mask = ~left_mask

        if left_mask.sum() < 2 or right_mask.sum() < 2:
            return float('inf')
        
        left_gini = self.gini(y[left_mask])
        right_gini = self.gini(y[right_mask])

        n_left, n_right = left_mask.sum(), right_mask.sum()
        return (n_left * left_gini + n_right * right_gini) / len(y)
    
    def splitting_lookahead(self, X, y, max_features=None, n_random_tries=5):

        if max_features is None:
            max_features = int(np.sqrt(X.shape[1])) #root num of features

        feature_indices = np.random.choice(X.shape[1], max_features, replace=False)

        best_scores = float('inf')
        best_split = None

        for feature in feature_indices:
            for _ in range(n_random_tries):
                min_val, max_val = X[:, feature].min(), X[:, feature].max()
                threshold = np.random.uniform(min_val, max_val)

                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue #bad split

                best_left = float('inf')
                for _ in range(n_random_tries):
                    score = self.random_split_gini(X[left_mask], y[left_mask])
                    best_left = min(best_left, score)

                best_right = float('inf')
                for _ in range(n_random_tries):
                    score = self.random_split_gini(X[right_mask], y[right_mask])
                    best_right = min(best_right, score)

                n_left, n_right = left_mask.sum(), right_mask.sum()
                weighted_score = (n_left * best_left + n_right * best_right) / len(y)

                if weighted_score < best_scores:
                    best_scores = weighted_score
                    best_split = (feature, threshold, left_mask, right_mask)
                    
        return best_split        

    def random_tree(self, X, y, depth, max_depth, min_samples, max_features):
        if depth >= max_depth or len(y) <= min_samples or len(np.unique(y)) == 1:
            #either too deep, too little examples, too pure
            return {'class': np.bincount(y).argmax()}
        
        if max_features is None:
            max_features = int(np.sqrt(X.shape[1]))  # Default: sqrt(total features)
    
        best_split = self.splitting_lookahead(X, y, max_features)

        if best_split is None:
            return {'class': np.bincount(y).argmax()}
    
        feature, threshold, left_mask, right_mask = best_split        

        left_tree = self.random_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples, max_features)
        right_tree = self.random_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples, max_features) 

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
            tree = self.random_tree(X[idx], y[idx], 0, self.max_depth, self.min_samples, self.max_features)
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

guy = ExtremelyRandomForestFuture()
guy.fit(X_train, y_train)
print(guy.score(X_test, y_test))





n_trees_list = [1, 5, 10, 20, 50, 100, 200]
train_scores = []
test_scores = []

for n in n_trees_list:
    model = ExtremelyRandomForestFuture(n_trees=n, max_depth=10, min_samples=3, max_features=5)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(n_trees_list, train_scores, 'o-', label='Train Accuracy', color='blue')
plt.plot(n_trees_list, test_scores, 'o-', label='Test Accuracy', color='red')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Custom Model: Accuracy vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()