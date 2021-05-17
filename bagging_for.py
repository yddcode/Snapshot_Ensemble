from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse, random
from sklearn.svm import SVC
from models.DGP3 import DeepGP
import pandas as pd
from sklearn.metrics import f1_score, classification_report, recall_score, precision_score, accuracy_score, r2_score
import sys, time, torch
from sklearn.model_selection import train_test_split, KFold
import csv
import pandas as pd

def loadDataSet(fileName):#读取数据
    data_train = np.loadtxt(open(fileName, encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    return X_train, y_train

def rand_train(dataMat,labelMat):#自助法采样
    len_train = len(labelMat)#获取样本1数
    index = []
    for i in range(len_train):#抽取样本数次样本
        index.append(random.randint(0,len_train-1)) #随机生成样本索引
    return dataMat[index],labelMat[index]#返回训练集与训练集标签

def test(X_test, y_test, gpmodule, t):
    X_test = torch.from_numpy(X_test)
    pred = gpmodule(X_test)
    pred = pred.numpy()
    sum = 0.0
    for i in range(5200):   # 26000 * 0.8 20800
        if (pred[i] == y_test[i]):
            sum = sum + 1
    # np.savetxt('dongna/Y_test.csv', y_test, delimiter=',')
    # np.savetxt('dongna/Y_test_pred.csv', pred, delimiter=',')
    outputfile = open('dongna/for/covbagging_3c_'+str(t)+'.txt', "a")
    sys.stdout = outputfile
    print("\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(sum, 5200, 100. * sum / 5200))
    print("acc:", accuracy_score(y_test, pred))
    print("precision", precision_score(y_test, pred, average='macro'))
    print("recall", recall_score(y_test, pred, average='micro'))
    print("F1", f1_score(y_test, pred, average='macro'))
    print(classification_report(y_test, pred ))
    return pred, accuracy_score(y_test, pred), precision_score(y_test, pred, average='macro'), recall_score(y_test, pred, average='micro'), f1_score(y_test, pred, average='macro')

def bagging_by_DGP(dataMat, labelMat, fileName, t = 30):#默认并行生成十个基学习器
    test_data,test_label = loadDataSet(fileName)  #获取测试样本与标签   
    predict_list = []
    accuracy=[]
    precisi=[]
    recal=[]
    f1=[]
    for i in range(t):#并行生成T个
        train_data,train_label = rand_train(dataMat,labelMat)#自主采样1得到样本
        tr_x = torch.from_numpy(train_data)
        tr_y = torch.from_numpy(train_label)
        deepgp = DeepGP(tr_x, tr_y, num_classes=13)
        # epoch iter batch learningr
        deepgp.train(60, 128, 512, 0.0001)

        y_pred, a, p, r, f  = test(test_data, test_label, deepgp, t)
        # print('pred', y_pred)
        predict_list.append(y_pred), accuracy.append(a), precisi.append(p), recal.append(r), f1.append(f)
        print('mean:', np.mean(accuracy), np.mean(precisi), np.mean(recal), np.mean(f1))
    print('meta-learning : t', t)
    return predict_list,test_label

def calc_acc(predict_list,test_label):#计算准确率
    pred = []
    # print('calc::', predict_list,len(predict_list))
    predict_list = list(map(list, zip(*predict_list)))
    for i in range(len(predict_list)):
        pred.append(np.argmax(np.bincount(predict_list[i])))
    sum = 0.0
    print(len(pred), pred[0:15], test_label[0:15])
    # np.savetxt('dongna/main/pred30.csv', pred, delimiter=',')
    for i in range(5200):   # 26000 * 0.8 20800
        if (pred[i] == test_label[i]):
            sum = sum + 1
    print("\nlast Accuracy: {}/{} ({:.2f}%)\n".format(sum, 5200, 100. * sum / 5200))
    print("acc:", accuracy_score(test_label, pred))
    print("precision", precision_score(test_label, pred, average='macro'))
    print("recall", recall_score(test_label, pred, average='micro'))
    print("F1", f1_score(test_label, pred, average='macro'))
    print(classification_report(test_label, pred))
    return float(sum / 5200.0)

if __name__ == "__main__":
    filetrain = 'dongna/main/cv_traindata1.csv'
    filetest  = 'dongna/main/cv_testdata1.csv'
    dataMat, labelMat = loadDataSet(filetrain)
    # train_data, train_label = rand_train(dataMat, labelMat)
    for i in range(24, 31, 3):
        predict_list, test_label = bagging_by_DGP(dataMat, labelMat, filetest, i)
        print("Bagging: ", calc_acc(predict_list, test_label))