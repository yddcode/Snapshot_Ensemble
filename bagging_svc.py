from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse, random
from sklearn.svm import SVC
from models.DGP3 import DeepGP
import pandas as pd
from sklearn.metrics import f1_score, classification_report, recall_score, precision_score, accuracy_score, confusion_matrix
import sys, time, torch
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=100, n_features=4,
#                             n_informative=2, n_redundant=0,
#                             random_state=0, shuffle=False)
data_train = np.loadtxt(open('D:/vscode/vscodework/zangwen/doc7feature_train.csv', encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
data_test = np.loadtxt(open('D:/vscode/vscodework/zangwen/doc7feature_test.csv', encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
X_train, y_train = data_train[:, :-1], data_train[:, -1]     
X_test, y_test = data_test[:, :-1], data_test[:, -1]     

# clf = BaggingClassifier(base_estimator=SVC(), n_estimators=200, random_state=0)
clf = RandomForestClassifier(n_estimators=50, n_jobs=2)
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
# n_estimators=500:生成500个决策树
# clf = BaggingClassifier(base_estimator=tree, n_estimators=250, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
# clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

# clf = AdaBoostClassifier(n_estimators=100)
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# best_ntree = []
# for i in range(1, 200):
#     rf = RandomForestClassifier(n_estimators = i+1, n_jobs = -1)
#     rf_cv = cross_val_score(rf, X_train, y_train, cv=10).mean()
#     best_ntree.append(rf_cv)

# print(max(best_ntree), np.argmax(best_ntree)+1)
# 0.9782692307692308 106

# plt.figure(figsize=[20, 5])
# plt.plot(range(1,200), best_ntree)
# plt.show()



print("acc:", accuracy_score(y_test, y_pred))
print("precision", precision_score(y_test, y_pred, average='macro'))
print("recall", recall_score(y_test, y_pred, average='micro'))
print("F1", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# 10 SVC
# acc: 0.9617307692307693
# precision 0.9738677386193259
# recall 0.9617307692307693
# F1 0.9609762432463728
# 20
# acc: 0.9615384615384616
# precision 0.9740678861368516
# recall 0.9615384615384616
# F1 0.9607100066731376
# 200
# acc: 0.9778846153846154
# precision 0.9792673751230231
# recall 0.9778846153846154
# F1 0.9785979628385564

# tree
# acc: 0.9778846153846154
# precision 0.9792673751230231
# recall 0.9778846153846154
# F1 0.9785979628385564

# forest
# acc: 0.9790384615384615
# precision 0.9804529098802693
# recall 0.9790384615384615
# F1 0.979709261364433
# 20
# acc: 0.9625
# precision 0.9736901894092529
# recall 0.9625
# F1 0.9619366835785944

# [[406   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0 364   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0 405   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0 393   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0 402   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0 410   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0 418   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0 379   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0 335   0   0  79   0]
#  [  0   0   0   0   0   0   0   0   0 392   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0 418   0   0]
#  [  0   0   0   0   0   0   0   0  36   0   0 379   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0 384]]
#               precision    recall  f1-score   support

#          0.0       1.00      1.00      1.00       406
#          1.0       1.00      1.00      1.00       364
#          2.0       1.00      1.00      1.00       405
#          3.0       1.00      1.00      1.00       393
#          4.0       1.00      1.00      1.00       402
#          5.0       1.00      1.00      1.00       410
#          6.0       1.00      1.00      1.00       418
#          7.0       1.00      1.00      1.00       379
#          8.0       0.98      0.53      0.69       414
#          9.0       1.00      1.00      1.00       392
#         10.0       1.00      1.00      1.00       418
#         11.0       0.68      0.99      0.81       415
#         12.0       1.00      1.00      1.00       384

#     accuracy                           0.96      5200
#    macro avg       0.97      0.96      0.96      5200
# weighted avg       0.97      0.96      0.96      5200