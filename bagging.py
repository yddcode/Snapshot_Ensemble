from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse, random
from sklearn.svm import SVC
from models.DGP4 import DeepGP
import pandas as pd
from sklearn.metrics import f1_score, classification_report, recall_score, precision_score, accuracy_score, r2_score
import sys, time, torch
from sklearn.model_selection import train_test_split, KFold
import csv
import pandas as pd


def loadDataSet(fileName):#读取数据
    # numFeat = len(open(fileName).readline().split('\t')) 
    # dataMat = []
    # labelMat = []
    # fr = open(fileName)
    # for line in fr.readlines():
    #     lineArr =[]
    #     curLine = line.strip().split('\t')
    #     for i in range(numFeat-1):#添加数据
    #         lineArr.append(float(curLine[i]))
    #     dataMat.append(lineArr)
    #     labelMat.append(float(curLine[-1]))#添加数据对应标签
    data_train = np.loadtxt(open(fileName, encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    return X_train, y_train

def rand_train(dataMat,labelMat):#自助法采样
    len_train = len(labelMat)#获取样本1数
    index = []
    for i in range(len_train):#抽取样本数次样本
        index.append(random.randint(0,len_train-1)) #随机生成样本索引
    #     train_data.append(dataMat[index])#添加对应数据与标签
    #     train_label.append(labelMat[index])
    # print(dataMat[1])
    # print(train_data.shape)
    return dataMat[index],labelMat[index]#返回训练集与训练集标签

def test(X_test, y_test, gpmodule):
    X_test = torch.from_numpy(X_test)
    pred = gpmodule(X_test)
    pred = pred.numpy()
    # print(pred[:20])
    # print(y_test[:20])
    # print(pred.shape, y_test.shape)
    sum = 0.0
    for i in range(20000):   # 26000 * 0.8 20800
        if (pred[i] == y_test[i]):
            sum = sum + 1
    # np.savetxt('dongna/Y_test.csv', y_test, delimiter=',')
    # np.savetxt('dongna/Y_test_pred.csv', pred, delimiter=',')
    outputfile = open("dongna/main/models/doc_covbagging_7c9.txt", "a")
    sys.stdout = outputfile
    print("\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(sum, 20000, 100. * sum / 20000))
    print("acc:", accuracy_score(y_test, pred))
    print("precision", precision_score(y_test, pred, average='macro'))
    print("recall", recall_score(y_test, pred, average='micro'))
    print("F1", f1_score(y_test, pred, average='macro'))
    print(classification_report(y_test, pred ))
    return pred, accuracy_score(y_test, pred), precision_score(y_test, pred, average='macro'), recall_score(y_test, pred, average='micro'), f1_score(y_test, pred, average='macro')

def bagging_by_DGP(dataMat,labelMat,fileName,t=9):#默认并行生成十个基学习器
    test_data,test_label = loadDataSet(fileName)  #获取测试样本与标签   
    predict_list = []
    accuracy=[]
    precisi=[]
    recal=[]
    f1=[]
    for i in range(t):#并行生成T个
        train_data,train_label = rand_train(dataMat,labelMat)#自主采样1得到样本
        # clf = tree.DecisionTreeClassifier()#初始化决策树模型
        # clf.fit(train_data,train_label)#训练模型
        tr_x = torch.from_numpy(train_data)
        tr_y = torch.from_numpy(train_label)
        deepgp = DeepGP(tr_x, tr_y, num_classes=13)
        # epoch iter batch learningr
        deepgp.train(2, 2, 512, 0.0001)

        y_pred, a, p, r, f  = test(test_data, test_label, deepgp)
        # print('pred', y_pred)
        predict_list.append(y_pred), accuracy.append(a), precisi.append(p), recal.append(r), f1.append(f)
        print('mean:', np.mean(accuracy), np.mean(precisi), np.mean(recal), np.mean(f1))
        # print('predict_list', predict_list)
        # total = []
        # y_predicted = clf.predict(test_data)#预测数据
        # total.append(y_predicted)
        # predict_list.append(total)#结果添加到预测列表中
    return predict_list,test_label


# csv_file = open('dongna/compar/test-3c.csv')
# csv_reader_lines = csv.reader(csv_file)
# most = []
# for one_line in csv_reader_lines:
#     print(one_line)
#     print(np.bincount(one_line))
#     print(np.argmax(np.bincount(one_line)))
#     most.append(np.argmax(np.bincount(one_line)))

# most = pd.DataFrame(data=most)
# most.to_csv('dongna/compar/most-3c.csv')

# data = pd.read_csv('dongna/compar/ff-3c.csv') 
# data = np.array(data.loc[:,:]) #去除第一行
# # dat=df.drop(['None'],axis=1) #去除‘id’列
# data = list(map(lambda x:x[1:], data))

# pred = [np.argmax(one_hot)for one_hot in data]
# print(len(pred))

def calc_error(predict_list,test_label):#计算错误率
    pred = []
    # print('calc::', predict_list,len(predict_list))
    predict_list = list(map(list, zip(*predict_list)))
    print(predict_list, len(predict_list))
    # f = f.detach().numpy()
    # print(f,f[1])

    for i in range(len(predict_list)):
        print(predict_list[i])
        # jj = predict_list[i]
        # for j in range(len(jj)):
        # max = np.bincount(predict_list[i])
        print(np.argmax(np.bincount(predict_list[i])))
        pred.append(np.argmax(np.bincount(predict_list[i])))
    sum = 0.0
    print(len(pred), pred[0:15], test_label[0:15])
    np.savetxt('dongna/main/doc_pred9.csv', pred, delimiter=',')
    for i in range(20000):   # 26000 * 0.8 20800
        if (pred[i] == test_label[i]):
            sum = sum + 1
    print("\nlast Accuracy: {}/{} ({:.2f}%)\n".format(sum, 20000, 100. * sum / 20000))
    print("acc:", accuracy_score(test_label, pred))
    print("precision", precision_score(test_label, pred, average='macro'))
    print("recall", recall_score(test_label, pred, average='micro'))
    print("F1", f1_score(test_label, pred, average='macro'))
    print(classification_report(test_label, pred))
    # m,n,k = shape(predict_list)#提取预测集信息
    # predict_label = sum(predict_list,axis = 0)
    # predict_label = sign(predict_label)
    # for i in range(len(predict_label[0])):
    #     if predict_label[0][i] == 0:#如果票数相同，则随机生成一个标签
    #         tip = random.randint(0,1)
    #         if tip == 0:
    #             predict_label[0][i] = 1
    #         else:
    #             predict_label[0][i] =-1
    # error_count = 0#初始化预测错误数
    # for i in range(k):
    #     if predict_label[0][i] != test_label[i]:#判断预测精度
    #         error_count += 1
    # error_rate = error_count/k
    return float(sum / 20000.0)

if __name__ == "__main__":
    # filetrain = 'dongna/main/cv_traindata1.csv'
    # filetest  = 'dongna/main/cv_testdata1.csv'
    filetrain = 'D:/vscode/vscodework/zangwen/doc7feature_train.csv'
    filetest  = 'D:/vscode/vscodework/zangwen/doc7feature_test.csv'
    dataMat, labelMat = loadDataSet(filetrain)
    # train_data, train_label = rand_train(dataMat, labelMat)
    predict_list, test_label = bagging_by_DGP(dataMat, labelMat, filetest)
    print("Bagging: ", calc_error(predict_list, test_label))