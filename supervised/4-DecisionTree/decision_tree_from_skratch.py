import numpy as np

class DecisionTreeClassifier:
    def _init_(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # If all samples belong to the same class, return that class
        if len(unique_classes) == 1:
            return unique_classes[0]

        # If max depth is reached, return the most common class
        if self.max_depth is not None and depth >= self.max_depth:
            return self._most_common_class(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return self._most_common_class(y)

        left_indices = X[:, best_split['feature']] < best_split['value']
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': best_split['feature'],
            'value': best_split['value'],
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_split = None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for value in thresholds:
                left_indices = X[:, feature] < value
                right_indices = ~left_indices

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gini = self._calculate_gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = {'feature': feature, 'value': value}

        return best_split

    def _calculate_gini(self, left_y, right_y):
        left_gini = 1 - sum((np.bincount(left_y) / len(left_y)) ** 2) if len(left_y) > 0 else 0
        right_gini = 1 - sum((np.bincount(right_y) / len(right_y)) ** 2) if len(right_y) > 0 else 0
        total_gini = (len(left_y) * left_gini + len(right_y) * right_gini) / (len(left_y) + len(right_y))
        return total_gini

    def _most_common_class(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if not isinstance(tree, dict):
            return tree

        feature = tree['feature']
        value = tree['value']

        if sample[feature] < value:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])