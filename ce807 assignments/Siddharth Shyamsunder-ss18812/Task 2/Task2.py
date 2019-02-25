'''
Created by ss18812
on 16/2/2019
'''
import pandas as pd
import csv
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 
#Reading the training data
data=open('aij-wikiner-en-wp2',encoding='utf-8')
data=data.read()
data=data.split('|')
data=" ".join(data)
data=data.split()
count=0
for i in range(len(data)):
    count+=1
block=[]
count=30000
print('Length of the training dataset:',count)
for i in range(count):
    block.append([])
i=0
#print(block)
for j in range(0,(3*count)):
    if(j==(3*count)):
        break
    if(j%3==0 and j!=0):
        i+=1

    block[i].append(data[j])


data="Word POS NER".split()
csvdata=data
with open("ML.csv",'w',encoding='utf-8') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerow(csvdata)
csvFile.close()


csvdata=block

with open("ML.csv","a",encoding='utf-8') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerows(csvdata)
csvFile.close()
#Reading the test data
data2=open('wikigold.conll.txt',encoding='utf-8')
data2=data2.read()
data2=data2.split('|')
data2=" ".join(data2)
data2=data2.split()
count2=0
for i in range(len(data2)):
    count2+=1
block1=[]
count2=count2//2
print('Length of the test dataset:',count2)
for i in range(count2):
    block1.append([])
i=0
for j in range(0,(2*count2)):
    if(j==(2*count2)):
        break
    if(j%2==0 and j!=0):
        i+=1

    block1[i].append(data2[j])

data2="Word NER".split()
csvdata=data2
with open("MLOP.csv",'w',encoding='utf-8') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerow(csvdata)
csvFile.close()


csvdata=block1

with open("MLOP.csv","a",encoding='utf-8') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerows(csvdata)
csvFile.close()

data=pd.read_csv('ML.csv')

data2=pd.read_csv('MLOP.csv')
#Encoding features and output
frames=[data,data2]
result=pd.concat(frames,sort=True)
result=result[result.NER!='B-MISC']
result=result[result.NER!='B-PER']
data3=result.Word
data4=result.NER

data3=pd.get_dummies(data3)
X=data3[:count]

X2=data3[count:]

encoder=preprocessing.LabelEncoder()
Y=encoder.fit_transform(result.iloc[0:count,0].astype(str))
Y1=encoder.fit_transform(result.iloc[count:,0].astype(str))

listt=list(encoder.classes_)
#Performing Perceptron Algorithm
clf=Perceptron(tol=1e-3,random_state=0)
dt=clf.fit(X,Y)

scores = cross_val_score(dt, X, Y, cv=5)
print("Accuracy of a Perceptron Algorithm is:",round(np.mean((scores*100)),2),"+/-",round(np.std(scores*100),0))
