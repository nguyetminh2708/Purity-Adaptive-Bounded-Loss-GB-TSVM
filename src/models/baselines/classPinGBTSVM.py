import numpy as np
import time
from scipy.linalg import solve
import pandas as pd
from numpy import linalg
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import combinations
from collections import defaultdict

class PinGBTSVM(BaseEstimator):
    def __init__(self, d1, d2, eps1, eps2, tau):
        self.d1 = d1
        self.d2 = d2
        self.eps1 = eps1
        self.eps2 = eps2
        self.tau = tau
        self.models = {}
    def fit(self, X_train):
        A = X_train[X_train[:, -1] == 1, :-1]
        B = X_train[X_train[:, -1] != 1, :-1]
        C1 = A[:,:-1]
        C2=B[:,:-1]
        R1=A[:,-1]
        R2=B[:,-1]

        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))

        
        H1 = np.hstack((C1, e1))
        G1 = np.hstack((C2, e2))
        
        HH1 = np.dot(H1.T, H1) + self.eps1 * np.eye(H1.shape[1])

        HHG = linalg.solve(HH1, G1.T)
        kerH1 = np.dot(G1, HHG)
        kerH1 = (kerH1 + kerH1.T) / 2
        m1=kerH1.shape[0]
        e3= np.ones(m1)
        R11=-(e3+R2)
        solvers.options['show_progress'] = False
        vlb = self.tau*self.d1*(np.ones((m1,1)))
        
        G=-np.eye(m1)
        
        alpha1 = solvers.qp(matrix(kerH1,tc='d'),matrix(R11,tc='d'),matrix(G,tc='d'),matrix(vlb,tc='d'))
        alphasol1 = np.array(alpha1['x'])
        z = -np.dot(HHG,alphasol1)
        self.w1 = z[:len(z)-1]
        self.b1 = z[len(z)-1]

        # 2nd QPP

        QQ = np.dot(G1.T, G1)
        QQ = QQ + self.eps2 * np.eye(QQ.shape[1])
        QQP = linalg.solve(QQ, H1.T)
        kerH2 = np.dot(H1, QQP)
        kerH2 = (kerH2 + kerH2.T) / 2
        m2=kerH2.shape[0]
        e4 = np.ones(m2)
        R22=-(e4+R1)
        solvers.options['show_progress'] = False
        vlb = self.tau*self.d2*(np.ones((m2,1)))
    
        G=-np.eye(m2)
        alpha2= solvers.qp(matrix(kerH2,tc='d'),matrix(R22,tc='d'),matrix(G,tc='d'),matrix(vlb,tc='d'))
        alphasol2 = np.array(alpha2['x'])
        z = np.dot(QQP,alphasol2)
        self.w2 = z[:len(z)-1]
        self.b2 = z[len(z)-1]
    def predict(self, X_test):
        P_1 = X_test[:, :-1]
        y1 = np.dot(P_1, self.w1) + self.b1
        y2 = np.dot(P_1, self.w2) + self.b2
        y_pred = np.zeros((y1.shape[0], 1))
        for i in range(y1.shape[0]):
            if np.min([np.abs(y1[i]), np.abs(y2[i])]) == np.abs(y1[i]):
                y_pred[i] = 1
            else:
                y_pred[i] = -1
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

class OvO_PinGBTSVM(BaseEstimator):
    def __init__(self, d1=1.0, d2=1.0, eps1=1e-6, eps2=1e-6, tau=0.1):
        self.d1 = d1
        self.d2 = d2
        self.eps1 = eps1
        self.eps2 = eps2
        self.tau = tau
        self.binary_classifiers = {}
        self.classes_ = None
        self.class_pairs = None
    
    def fit(self, X_train, y_train=None):
        if y_train is None:
            X = X_train[:, :-1]
            y = X_train[:, -1]
        else:
            X = X_train
            y = y_train
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes < 2:
            raise ValueError("Cần ít nhất 2 lớp để phân loại")
        self.class_pairs = list(combinations(self.classes_, 2))
        for class_i, class_j in self.class_pairs:
            mask = (y == class_i) | (y == class_j)
            X_pair = X[mask]
            y_pair = y[mask]
            y_binary = np.where(y_pair == class_i, 1, -1)
            binary_data = np.column_stack([X_pair, y_binary])
            binary_classifier = PinGBTSVM(self.d1, self.d2, self.eps1, self.eps2, self.tau)
            binary_classifier.fit(binary_data)
            self.binary_classifiers[(class_i, class_j)] = binary_classifier
        return self
    def predict(self, X_test):
        n_samples = X_test.shape[0]
        votes = np.zeros((n_samples, len(self.classes_)))
        for class_i, class_j in self.class_pairs:
            classifier = self.binary_classifiers[(class_i, class_j)]
            test_data = np.column_stack([X_test, np.zeros(X_test.shape[0])])
            predictions = classifier.predict(test_data)
            for i in range(n_samples):
                if predictions[0, i] == 1:
                    class_idx = np.where(self.classes_ == class_i)[0][0]
                    votes[i, class_idx] += 1
                else:
                    class_idx = np.where(self.classes_ == class_j)[0][0]
                    votes[i, class_idx] += 1
        predictions = []
        for i in range(n_samples):
            max_votes = np.max(votes[i])
            tied_classes = np.where(votes[i] == max_votes)[0]
            predicted_class_idx = np.min(tied_classes)
            predictions.append(self.classes_[predicted_class_idx])
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test) * 100
        return accuracy


class OVR_PinGBTSVM(BaseEstimator):
    def __init__(self, d1=0.1, d2=0.1, eps1=0.05, eps2=0.05, tau = 1.0):
        self.d1 = d1
        self.d2 = d2
        self.eps1 = eps1
        self.eps2 = eps2
        self.tau = tau
        self.models = {}
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        for i, cls in enumerate(self.classes_):
            binary_y = np.ones(len(y))
            binary_y[y != cls] = -1
            binary_data = np.column_stack((X, binary_y))
            model = PinGBTSVM(d1=self.d1, d2=self.d2, eps1=self.eps1, eps2=self.eps2 ,tau = self.tau)
            model.fit(binary_data)
            self.models[cls] = model
        return self

    def predict(self, X, threshold=0):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        confidence_scores = np.zeros((n_samples, n_classes))
        for i, cls in enumerate(self.classes_):
            model = self.models[cls]
            dummy_y = np.ones((n_samples, 1))
            test_data = np.column_stack((X, dummy_y))
            P_1 = test_data[:, :-1]
            y1 = np.dot(P_1, model.w1) + model.b1
            y2 = np.dot(P_1, model.w2) + model.b2
            confidence = np.abs(y2) - np.abs(y1)
            confidence_scores[:, i] = confidence.flatten()  # hoặc np.squeeze(confidence)
        if np.isscalar(threshold):
            threshold = np.ones(n_classes) * threshold
        predictions = (confidence_scores > threshold).astype(int)
        return predictions

        
    def score(self, X, y_true):
        y_pred = self.predict(X)
        n_classes = len(self.classes_)
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true_bin = np.zeros((len(y_true), n_classes))
            y_true_flat = y_true.flatten()
            for i, label in enumerate(y_true_flat):
                if label in label_to_idx:
                    y_true_bin[i, label_to_idx[label]] = 1
                else:
                    raise ValueError(f"Nhãn {label} không có trong danh sách classes: {self.classes_}")
            y_true = y_true_bin
        elif len(y_true.shape) == 2 and y_true.shape[1] == n_classes:
            pass
        else:
            raise ValueError(f"Định dạng của y_true không được hỗ trợ. Kích thước hiện tại: {y_true.shape}")
        exact_matches = np.all(y_true == y_pred, axis=1)
        accuracy = np.mean(exact_matches) * 100
        
        return accuracy
