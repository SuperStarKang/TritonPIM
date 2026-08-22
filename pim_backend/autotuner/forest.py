# Stage 9 — Random Forest Surrogate (design/AutoTuner_Design.md §36)
#
# 수집한 (Config + Plan Features, GOOD/BAD) 데이터로 P(GOOD)를 예측한다.
# latency regression이 아니라 classification이다 (§34).
#
# pim_triton 환경에는 scikit-learn이 없으므로 numpy만으로 구현한다.
# tree-guided neighbor 생성(§38)이 필요로 하는 decision path 추적을
# 직접 노출한다.

import math
import random

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "prob")

    def __init__(self, prob):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.prob = prob

    @property
    def is_leaf(self):
        return self.left is None


def _best_split(X, y, feature_indices, min_samples_leaf):
    """gini impurity를 최소화하는 (feature, threshold)를 찾는다."""
    n = len(y)
    best = None
    for f in feature_indices:
        order = np.argsort(X[:, f], kind="stable")
        xs = X[order, f]
        ys = y[order]
        prefix_pos = np.cumsum(ys)
        total_pos = prefix_pos[-1]
        for i in range(min_samples_leaf, n - min_samples_leaf + 1):
            if xs[i - 1] == xs[i]:
                continue
            left_n = i
            right_n = n - i
            left_pos = prefix_pos[i - 1]
            right_pos = total_pos - left_pos
            p_l = left_pos / left_n
            p_r = right_pos / right_n
            gini_l = 1.0 - p_l * p_l - (1.0 - p_l) * (1.0 - p_l)
            gini_r = 1.0 - p_r * p_r - (1.0 - p_r) * (1.0 - p_r)
            gini = (left_n * gini_l + right_n * gini_r) / n
            if best is None or gini < best[0]:
                best = (gini, f, (xs[i - 1] + xs[i]) / 2.0)
    return best


class DecisionTreeClassifier:
    def __init__(self, max_depth, min_samples_leaf, n_sub_features, rng):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_sub_features = n_sub_features
        self.rng = rng
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        node = _Node(prob=float(y.mean()))
        if (depth >= self.max_depth
                or len(y) < 2 * self.min_samples_leaf
                or y.min() == y.max()):
            return node
        n_features = X.shape[1]
        features = self.rng.sample(
            range(n_features), min(self.n_sub_features, n_features))
        split = _best_split(X, y, features, self.min_samples_leaf)
        if split is None:
            return node
        gini_split, feature, threshold = split
        p = node.prob
        gini_node = 1.0 - p * p - (1.0 - p) * (1.0 - p)
        if gini_split >= gini_node:
            return node
        mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def predict_prob(self, x):
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prob

    def decision_path(self, x):
        """x를 분류할 때 지나간 분기 feature index (등장 순서, 중복 제거).

        tree가 이 configuration을 판단하며 중요하게 사용한 feature다 (§38).
        """
        node = self.root
        path = []
        while not node.is_leaf:
            if node.feature not in path:
                path.append(node.feature)
            node = node.left if x[node.feature] <= node.threshold else node.right
        return path


class RandomForestClassifier:
    def __init__(self, n_trees=25, max_depth=8, min_samples_leaf=2, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, n_features = X.shape
        n_sub = max(1, int(math.sqrt(n_features)))
        rng = random.Random(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            indices = [rng.randrange(n) for _ in range(n)]  # bootstrap
            tree = DecisionTreeClassifier(
                self.max_depth, self.min_samples_leaf, n_sub, rng)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict_prob(self, x):
        """P(GOOD) (§36)."""
        x = np.asarray(x, dtype=float)
        return sum(tree.predict_prob(x) for tree in self.trees) / len(self.trees)

    def sample_tree(self, rng):
        """§52 forest.sample_tree() — neighbor 생성을 guide할 tree 하나."""
        return rng.choice(self.trees)
