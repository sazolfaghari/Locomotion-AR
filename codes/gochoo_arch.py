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
import csv,cv2
# ===================Read image labels==========================
labelfilename = 'gochoo_sectionall_new_overlap.csv'
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
rootImages = './gochoo_images/'
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

def DCNN(xTrain,num_classes):
  model = Sequential()
  model.add(Conv2D(32, (5, 5), padding='same',input_shape=xTrain.shape[1:]))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(64, (5, 5), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(128, (5, 5), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Flatten())

  model.add(Dense(254))
  model.add(Activation('relu'))


  model.add(Dense(128))
  model.add(Activation('relu'))


  model.add(Dense(32))
  model.add(Activation('relu'))

  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  opt = Adam(lr=0.0001)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

  return model

  
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
EPOCHS = 27

kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
kfold.get_n_splits(data, labels)
count = 1

with open('results_gochoo_DCNN_all_actions_overlap.csv', 'a+') as csvfile:
    csvwriter = csv.DictWriter(csvfile, ['classifier','accuracy', 'f1score', 'precision', 'recall', 'precision-macro',
                                        'recall-macro', 'fmeasure-macro','confusion_matrix'])
    csvwriter.writeheader()
csvfile.close()

conf_matrix = 0
for train_ix, test_ix in kfold.split(data, labels):

    train_X, test_X = data[train_ix], data[test_ix]
    train_y, test_y = labels[train_ix],labels[test_ix]        

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    num_classes =  16 #len(np.unique(train_y))
    


    train_X /= 255.0
    test_X /= 255.0

    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)
    
    train_y =  keras.utils.to_categorical(train_y, num_classes)
    
    model = DCNN(train_X, num_classes)
    history = model.fit(train_X ,train_y, batch_size = BS, verbose=0, epochs=EPOCHS )
    
    '''train_feat = model.predict(train_X, batch_size=BS) 
    test_feat = preclf.predict(test_X, batch_size=BS)
    
    print(train_feat.shape)
    print(test_feat.shape)
    
    height = train_feat.shape[1]
    width = train_feat.shape[2]
    channels = train_feat.shape[3]
    
    
    train_feat = np.reshape(train_feat,(-1,height*width*channels)) 
    test_feat = np.reshape(test_feat,(-1,height*width*channels)) 
    
    
    clf.fit(train_feat, train_y)     '''
    predictions = model.predict(test_X)
    predictions = predictions.argmax(axis=1)  
    
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
csvdic = {'classifier':'DCNN', 'accuracy': accuracy, 'f1score': fMeasure, 'precision': precision, 'recall': recall,
      'precision-macro': precision_macro_average, 'recall-macro': recall_macro_average,
      'fmeasure-macro': fmeasure_macro_average,'confusion_matrix':conf_matrix}
with open('results_gochoo_DCNN_all_actions_overlap.csv', 'a+') as csvfile:
    writer = csv.DictWriter(csvfile, ['classifier','accuracy', 'f1score', 'precision', 'recall', 'precision-macro',
                                    'recall-macro', 'fmeasure-macro','confusion_matrix'])
    writer.writerow(csvdic)
csvfile.close()
count = 1
print('Final measures for Calssifier: DCNN')
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



