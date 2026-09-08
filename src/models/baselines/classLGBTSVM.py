import numpy as np
import time
from scipy.linalg import solve
import pandas as pd
from numpy import linalg
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator
from itertools import combinations
from collections import defaultdict

class LGBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1,d3=0.1, d4=0.1):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.models = {}
    def fit(self, X_train):
        C1 = X_train[X_train[:, -1] == 1, :-1]
        C2 = X_train[X_train[:, -1] != 1, :-1]
        A= C1[:,:-1]
        B=C2[:,:-1]
        R1=C1[:,-1]
        R2=C2[:,-1]
        p = A.shape[0]
        q = B.shape[0]
        lb1 = np.concatenate((np.zeros(p), np.zeros(q)))
        ub1 = np.concatenate((np.zeros(p), self.d1 * np.ones(q)))
        f1 = -(self.d3) * np.concatenate((np.zeros(p), np.ones(q)+R2))

        Q1=np.vstack((np.hstack((np.dot(A, A.T) + self.d3 * np.eye(p), np.dot(A, B.T))), 
                      np.hstack((np.dot(B, A.T), np.dot(B, B.T))))) + np.ones((p + q, p + q))
        Q1 = (Q1 + Q1.T) / 2

        if np.linalg.matrix_rank(Q1) < Q1.shape[1]:
            Q1 = Q1 + 1e-4 * np.eye(Q1.shape[0])

        G=np.concatenate((np.eye(q+p), -np.eye(p+q)))
        h=np.concatenate((ub1,-lb1))
        solvers.options['show_progress'] = False
        alpha2= solvers.qp(matrix(Q1,tc='d'),matrix(f1,tc='d'),matrix(G,tc='d'),matrix(h,tc='d'))
        x1 = np.array(alpha2['x'])

        lb2 = np.concatenate((np.zeros(q), np.zeros(p)))
        ub2 = np.concatenate((np.zeros(q), self.d2 * np.ones(p)))
        f2 = -(self.d4) * np.concatenate((np.zeros(q), np.ones(p)+R1))

        Q2 = np.vstack((np.hstack((np.dot(B, B.T) + self.d4 * np.eye(q), np.dot(B, A.T))), np.hstack((np.dot(A, B.T), np.dot(A, A.T))))) + np.ones((p + q, p + q))
        Q2 = (Q2 + Q2.T) / 2

        if np.linalg.matrix_rank(Q2) < Q2.shape[1]:
            Q2 = Q2 + 1e-4 * np.eye(Q2.shape[0])

        cd=np.concatenate((np.eye(q+p), -np.eye(p+q)))
        vcd=np.concatenate((ub2,-lb2))
        alpha2= solvers.qp(matrix(Q2,tc='d'),matrix(f2,tc='d'),matrix(cd,tc='d'),matrix(vcd,tc='d'))
        x2 = np.array(alpha2['x'])

        self.b1=-(1 / self.d3) *np.dot(np.hstack((np.ones(p).T,np.ones(q).T)),x1)
        self.b2 = (1 / self.d4) * np.dot(np.hstack((np.ones(p).T,np.ones(q).T)),x2)

        self.w1 = -(1 / self.d3) * np.dot(np.hstack((A.T,B.T)),x1)
        self.w2 = (1 / self.d4) * np.dot(np.hstack((B.T,A.T)),x2)
    def predict(self, X_test):
        m = X_test.shape[0]
        test_data = X_test[:, :-1]
        if test_data.shape[1] > self.w1.shape[0]:
            test_data = test_data[:, :self.w1.shape[0]]
        
        y1 = np.dot(test_data, self.w1) + self.b1 * np.ones(m)
        y2 = np.dot(test_data, self.w2) + self.b2 * np.ones(m)
        y_pred = np.sign(np.abs(y2) - np.abs(y1))
        y_pred = y_pred.T
        return y_pred
    def score(self, X_test):
        y_pred = self.predict(X_test)
        no_test, no_col = X_test.shape
        err = 0.0
        obs1 = X_test[:, no_col-1]
        for i in range(no_test):
            if np.sign(y_pred[0, i]) != np.sign(obs1[i]):
                err += 1
        acc = ((X_test.shape[0] - err) / X_test.shape[0]) * 100
        return acc

class OvO_LGBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1, d3=0.1, d4=0.1):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.models = {}
        self.labels = None
    
    def fit(self, A_train):
        X = A_train[:, :-1]  # Đặc trưng - 5 cột đầu
        y = A_train[:, -1]   # Nhãn - cột cuối
        self.labels = np.unique(y)
        for i in range(len(self.labels)):
            for j in range(i + 1, len(self.labels)):
                label_i = self.labels[i]
                label_j = self.labels[j]
                mask = (y == label_i) | (y == label_j)
                X_pair = X[mask]
                y_pair = y[mask]
                y_binary = np.ones_like(y_pair)
                y_binary[y_pair == label_j] = -1
                X_train_pair = np.column_stack((X_pair, y_binary))
                model = LGBTSVM(self.d1, self.d2, self.d3, self.d4)
                model.fit(X_train_pair)
                self.models[(label_i, label_j)] = model
    
    def predict(self, A_test):
        if self.labels is None:
            raise ValueError("Model chưa được huấn luyện")
        n_samples = A_test.shape[0]
        X_test_with_dummy_label = np.column_stack((A_test, np.zeros(n_samples)))
        votes = np.zeros((n_samples, len(self.labels)))
        label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        for (label_i, label_j), model in self.models.items():
            predictions = model.predict(X_test_with_dummy_label)
            for idx in range(n_samples):
                if predictions[0, idx] == 1: 
                    votes[idx, label_to_index[label_i]] += 1
                else:  
                    votes[idx, label_to_index[label_j]] += 1
        predicted_indices = np.argmax(votes, axis=1)
        predictions = self.labels[predicted_indices]
        return predictions
    def score(self, A_test):
        if self.labels is None:
            raise ValueError("Model chưa được huấn luyện")
        y_pred = self.predict(A_test)
        y_true = A_test[:, -1]
        accuracy = np.mean(y_pred == y_true) * 100
        return accuracy

class OvR_LGBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1, d3=0.1, d4=0.1):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.classifiers = {}
        self.classes = None
    
    def fit(self, X_train):
        X_features = X_train[:, :-1]
        y = X_train[:, -1]
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.ones(len(y))
            y_binary[y != cls] = -1
            X_binary = np.column_stack((X_features, y_binary))
            clf = LGBTSVM(d1=self.d1, d2=self.d2, d3=self.d3, d4=self.d4)
            clf.fit(X_binary)
            self.classifiers[cls] = clf
        return self

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        X_features = X_test[:, :-1]
        scores = []
        dummy_labels = np.zeros(n_samples)
        X_temp = np.column_stack((X_features, dummy_labels))
        for cls in self.classes:
            clf = self.classifiers[cls]
            pred = clf.predict(X_temp)
            scores.append(pred)
        scores = np.array(scores)
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample_scores = [scores[j][0][i] for j in range(len(self.classes))]
            best_class_idx = np.argmax(sample_scores)
            predictions[i] = self.classes[best_class_idx]
        
        return predictions
    def score(self, X_test):
        y_pred = self.predict(X_test)
        y_true = X_test[:, -1]
        correct = np.sum(y_pred == y_true)
        total = len(y_true)
        return (correct / total) * 100

class MultilabelLGBTSVM(BaseEstimator):

    def __init__(self, d1=0.1, d2=0.1, d3=0.1, d4=0.1):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.models = defaultdict(dict)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for pos in self.classes_:
            for neg in self.classes_:
                if pos == neg:
                    continue
                idx_pos = (y == pos)
                idx_neg = (y == neg)
                idx_rest = ~(idx_pos | idx_neg)
                X_pos = X[idx_pos]
                X_neg = X[idx_neg]
                X_rest = X[idx_rest]
                y_pos = np.ones(X_pos.shape[0])
                y_neg = -np.ones(X_neg.shape[0])
                y_rest = np.zeros(X_rest.shape[0])
                X_temp = np.vstack((X_pos, X_neg, X_rest))
                y_temp = np.hstack((y_pos, y_neg, y_rest)).reshape(-1,1)
                train_mat = np.hstack((X_temp, y_temp))
                model = LGBTSVM(d1=self.d1, d2=self.d2, d3=self.d3, d4=self.d4)
                model.fit(train_mat)
                self.models[pos][neg] = model
        return self

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes_)))
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        for pos in self.classes_:
            for neg in self.classes_:
                if pos == neg:
                    continue
                model = self.models[pos][neg]
                X_ext = np.hstack((X, np.zeros((X.shape[0],1))))
                preds = model.predict(X_ext).flatten()
                for i, p in enumerate(preds):
                    if p == 1:
                        votes[i, class_to_index[pos]] += 1
                    elif p == -1:
                        votes[i, class_to_index[neg]] += 1
        win_idx = np.argmax(votes, axis=1)
        return self.classes_[win_idx]

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return np.mean(preds == y_test)

