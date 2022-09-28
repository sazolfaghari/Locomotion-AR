import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold, KFold
from tensorflow.keras import backend as K
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix,classification_report
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, AveragePooling2D, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
import glob
import os
import numpy as np
import itertools
import operator
#from keras.utils import np_utils
from tensorflow.keras.regularizers import l2
import random
import time
import imageio
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import VGG19, ResNet50, ResNet50V2,InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121
#from alexnet import AlexNet
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import cv2


#===================Images are resized to 224x224 pixels==================
#=========================================================================  
def resizeFun(npfile):
  img=load_img(npfile)  
  img_array = img_to_array(img)
  old_size = img_array.shape[:2] # old_size is in (height, width) format

  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])

  #new_size should be in (width, height) format

  im = cv2.resize(img_array, (new_size[1], new_size[0]))

  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)

  color = [200, 200, 200]
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

  return new_im

desired_size = 224 #ResNet, VGG,DenseNet,MobileNet
#desired_size = 299 #InceptionV3, InceptionResNetV2
rootfiles = './Images/'
all_files = glob.glob(os.path.join(rootfiles , '*.png'))
all_files.sort(key=lambda x: int(os.path.splitext(x)[0].split("/")[-1].split("_")[0]))

for idx, files in enumerate(all_files):
  filename= rootfiles + files.split("/")[-1]
  new_im = resizeFun(files)
  save_img(filename, new_im)


# ===================Read image labels==========================
labelfilename = 'section_all.csv'
readlabelfile = open(labelfilename, "r")
labeldata, finalLabels, npLabels = [],[],[]
acceptable_labels = [1,3,4,7,8,11,13,15]
for line in readlabelfile:
    Type = line.split(",")
    #if (np.isin(float(Type[1]),acceptable_labels)) :
    labeldata.append(np.array(list(map(float,Type))))
npLabels = np.asarray(labeldata, dtype=np.int16)
print(npLabels.shape)
#id_imgs,class, id_p 
# ===================Load and Read images==========================
# Ordinamento immagini per id_immagine
rootImages = './Images/'
all_files = glob.glob(os.path.join(rootImages, '*.png'))
all_files.sort(key=lambda x: int(os.path.splitext(x)[0].split("/")[-1].split("_")[0]))
data, labels = [], []
for  files in all_files:
    imgID = int(files.split("/")[-1].split("_")[0])
    if imgID in npLabels[:, 0]:
        a = np.where(npLabels[:, 0] == imgID)
        labels.append(int(npLabels[int(a[0]),1]))
        # load the input images 
        img = load_img(files)
        img_array = img_to_array(img)
        #np_array = img_array.reshape(-1)
        data.append(img_array)


#data = np.array(data)
data = np.stack(data)
print(data.shape)
# ===================Convert Labels of different categories in numeric order==========================
#finalLabels[:, 1] = np.sort(finalLabels[:, 1],axis=None)
#print(finalLabels[:, 1])
temp = {i: j for j, i in enumerate(set(labels))}
labels = [temp[i] for i in labels]
labels = np.asarray(labels, dtype=np.int16)   
print(labels.shape) 
 
# ===================================================================================
# ===================Measure calculation==========================
# ===================================================================================
    
def measures_calculation(num_classes, confusion_matrix):
    precision = []
    recall = []
    f1_score = []
    for label in range(num_classes):
      col = confusion_matrix[:, label]
      row = confusion_matrix[label, :]
      prec = confusion_matrix[label, label] / row.sum()
      rec = confusion_matrix[label, label] / col.sum()
      precision.append(prec)
      recall.append(rec)
      f1_score.append(((2 * prec * rec) / (prec + rec)))

    precision= np.nan_to_num(precision)
    recall= np.nan_to_num(recall)
    f1_score= np.nan_to_num(f1_score)

    precision_macro_average = np.sum(precision) / num_classes
    recall_macro_average = np.sum(recall) / num_classes
    f1_score_macro_average = np.sum(f1_score) / num_classes
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = np.sum(confusion_matrix)
    accuracy =  diagonal_sum / sum_of_all_elements 

    return accuracy,precision,recall,f1_score,precision_macro_average,recall_macro_average,f1_score_macro_average

# ============Initialization===========================
# =====================================================
BS = 32
cm = 0

kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
kfold.get_n_splits(data, labels)
count = 1

with open('results_all_pretrained.csv', 'a+') as csvfile:
    csvwriter = csv.DictWriter(csvfile, ['pre_trained','classifier','accuracy', 'f1score', 'precision', 'recall', 'precision-macro',
                                        'recall-macro', 'fmeasure-macro','confusion_matrix'])
    csvwriter.writeheader()
csvfile.close()


preTrained_names = ["VGG19","ResNet50","ResNet50V2","MobileNet", "MobileNetV2", "DenseNet121"]   #"InceptionV3", "InceptionResNetV2",
# 

preTrained_classifiers = [
    VGG19(weights='imagenet',include_top=False),#7,7,512
    ResNet50(weights='imagenet',include_top=False),#7, 7, 2048
    ResNet50V2(weights='imagenet',include_top=False),#7, 7, 2048
    #InceptionV3(weights='imagenet',include_top=False), #8, 8, 2048
    #InceptionResNetV2(weights='imagenet',include_top=False), #8, 8, 1536
    MobileNet(weights='imagenet',include_top=False),
    MobileNetV2(weights='imagenet',include_top=False),
    DenseNet121(weights='imagenet',include_top=False)  
    ]   

names = ["NB","KNN","DT","SVM","RF","Neural Net","AdaBoost","QDAs"]
#
classifiers = [
    GaussianNB(),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    SVC(kernel="poly", C=1.0),  
    RandomForestClassifier(n_estimators=100, max_features=None, random_state=42),
    MLPClassifier(alpha=1, max_iter=500),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis()
    ]

#===============================================
#============Main Program==================
#===============================================
for preTrained_name, preclf in zip(preTrained_names, preTrained_classifiers):
    print('Name of pre-trained classifier:{}'.format(preTrained_name))
    for name, clf in zip(names, classifiers):
        print('Name of classifier:{}'.format(name))
        conf_matrix = 0
        for train_ix, test_ix in kfold.split(data, labels):

            train_X, test_X = data[train_ix], data[test_ix]
            train_y, test_y = labels[train_ix],labels[test_ix]        

            train_X = train_X.astype('float32')
            test_X = test_X.astype('float32')
            num_classes = 16 #len(np.unique(train_y))


            train_X /= 255.0
            test_X /= 255.0

            print(train_X.shape)
            print(train_y.shape)
            print(test_X.shape)
            print(test_y.shape)

            train_feat = preclf.predict(train_X, batch_size=BS) 
            test_feat = preclf.predict(test_X, batch_size=BS)
            
            print(train_feat.shape)
            print(test_feat.shape)
            
            height = train_feat.shape[1]
            width = train_feat.shape[2]
            channels = train_feat.shape[3]
            
            
            train_feat = np.reshape(train_feat,(-1,height*width*channels)) 
            test_feat = np.reshape(test_feat,(-1,height*width*channels)) 
            
            
            clf.fit(train_feat, train_y)     
            predictions = clf.predict(test_feat)
           
            cm = confusion_matrix(test_y,predictions, labels=list(range(0,num_classes)))
            conf_matrix = conf_matrix + cm

            print('Fold {}:'.format(count))
            print("classification_report:\n" , classification_report(test_y,predictions))
            print('===============================================================')
            print('Confusion Martrix:')
            print(cm)
            print('===============================================================')
            count +=1


        accuracy,precision,recall,fMeasure,precision_macro_average,recall_macro_average,fmeasure_macro_average = measures_calculation(num_classes, conf_matrix)
        csvdic = {'pre_trained':preTrained_name,'classifier':name, 'accuracy': accuracy, 'f1score': fMeasure, 'precision': precision, 'recall': recall,
              'precision-macro': precision_macro_average, 'recall-macro': recall_macro_average,
              'fmeasure-macro': fmeasure_macro_average,'confusion_matrix':conf_matrix}
        with open('results_all_pretrained.csv', 'a+') as csvfile:
            writer = csv.DictWriter(csvfile, ['pre_trained','classifier','accuracy', 'f1score', 'precision', 'recall', 'precision-macro',
                                            'recall-macro', 'fmeasure-macro','confusion_matrix'])
            writer.writerow(csvdic)
        csvfile.close()
        count = 1
        print('Final measures for Calssifier:{}'.format(name))
        print('----------------------------------------------------------------')
        print('accuracy: {}'.format(accuracy))
        print('precision:{}'.format(precision))
        print('recall:{}'.format(recall))
        print('fmeasure:{}'.format(fMeasure))

        print('precision-macro: {}'.format(precision_macro_average))
        print('recall-macro:{}'.format(recall_macro_average))
        print('fmeasure-macro:{}'.format(fmeasure_macro_average))
        print('----------------------------------------------------------------')
        print('Confusion Martrix:\n')
        print(conf_matrix)
