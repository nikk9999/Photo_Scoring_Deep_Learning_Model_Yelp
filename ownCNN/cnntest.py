#TensorFlow and Keras
import tensorflow as tf 
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt 

import cv2
import os
import random
import glob
from glob import glob
import pandas as pd
import csv
import cv2

import matplotlib.image as mpimg
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras import backend as K
import time
import sklearn
from sklearn.metrics import confusion_matrix

n = 64
x = []
y = []
xl = []
yl = []
ycat = []
yinfo = []
yquality = []
yname = []
tt = []
ttname = []
tt2 = []
ttname2 = []
tt3 = []
ttname3 = []

count = 0
i = 0

# img_dir = "/labeledYelp"
# data_path = os.path.join(img_dir,'*g')
# files = glob.glob(data_path)
# data = []

# img_dir = "test data yelp/Buttermilk Kitchen/"
# data_path = os.path.join(img_dir,'*g')
# files = glob.glob(data_path)
# data = []
# tt = []
# ttname = []

# for image in files:
# 	im = cv2.imread(image)
# 	ttname.append(image)
# 	tt.append(cv2.resize(im,(n,n), interpolation=cv2.INTER_CUBIC))

# with open('buttermilk.csv') as testfile:
# 	reader = csv.reader(testfile,delimiter=',')
# 	for row in reader:
# 		im = "test data yelp/Buttermilk Kitchen/" + row[3] + '.jpg'
# 		image = cv2.imread(im)
# 		tt.append(cv2.resize(image,(n,n), interpolation=cv2.INTER_CUBIC)) 
# 		ttname.append(im)
# print(len(tt))

# with open('fatmatts.csv') as testfile2:
# 	reader = csv.reader(testfile2,delimiter=',')
# 	for row in reader:
# 		im = "test data yelp/Fat Matt's Rib Shack/" + row[0] + '.jpg'
# 		image = cv2.imread(im)
# 		tt2.append(cv2.resize(image,(n,n), interpolation=cv2.INTER_CUBIC)) 
# 		ttname2.append(im)
# print(len(tt2))

# with open('homegrown.csv') as testfile2:
# 	reader = csv.reader(testfile2,delimiter=',')
# 	for row in reader:
# 		im = "test data yelp/Home Grown/" + row[0] + '.jpg'
# 		image = cv2.imread(im)
# 		tt3.append(cv2.resize(image,(n,n), interpolation=cv2.INTER_CUBIC)) 
# 		ttname3.append(im)
# print(len(tt3))

with open('Final_Train_Nov29.csv') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
    	if i==0:
            i = i+1
    	else:
    		i = i + 1
    		im = "finalTrainPhotos/" + row[1] + '.jpg'
    		image = cv2.imread(im)
    		# print(im)
    		# if image.any():
    			
    		# h, w, c = image.shape
    		# print("here")
    		x.append(cv2.resize(image,(n,n), interpolation=cv2.INTER_CUBIC))
    		# print((cv2.resize(image,(n,n), interpolation=cv2.INTER_CUBIC)).shape)
    		xl.append(row[4])

# imtest = cv2.imread("trainingYelp/_1AVl8QmfnDx1pmTxkFh7A.jpg")
# print(imtest)
i = 0
with open('finaltestinglabelsnew.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		if i==0:
			i = i+1
		else:
			i = i+1
			im = "labeledYelp/" + row[1] + '.jpg'
			image = cv2.imread(im)
			y.append(cv2.resize(image,(n,n), interpolation=cv2.INTER_CUBIC))
			# if (row[9]==1) & (row[10]==1):
			# 	yl.append(1)
			# else:
			# 	yl.append(0)
			yl.append(row[7])
			ycat.append(row[4])
			yinfo.append(row[6])
			yquality.append(row[5])
			yname.append(row[1])



train = np.array(x)
print(train.shape)
test = np.array(y)
trainlabels = np.array(xl)
testlabels = np.array(yl)
ynamenp = np.array(yname)

# ttnp = np.array(tt)
# ttnamenp = np.array(ttname)

# ttnp2 = np.array(tt2)
# ttnamenp2 = np.array(ttname2)

# ttnp3 = np.array(tt3)
# ttnamenp3 = np.array(ttname3)

yinfonp = np.array(yinfo)
yqualitynp = np.array(yquality)
# np.savetxt("train.csv",train,delimiter=',')
# np.savetxt("test.csv",test,delimiter=',')
# np.savetxt("trainlabels.csv",trainlabels,delimiter=',')
# np.savetxt("testlabels.csv",testlabels,delimiter=',')
# train, x_valid, trainlabels, x_validlabels = train_test_split(x_train, x_trainlabels, test_size = 0.2, random_state = 100)
# trainlabels = np.array([0,0,0,0,1,1,1,1])
# testlabels = np.array([0,1])
classnames = ['good', 'bad']


train = train/255.0
test = test/255.0
e = 4

##############################################################################################################################
start1 = time.time()
model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(n, n, 3)),
    # keras.layers.Dense(128, activation=tf.nn.relu),
    # keras.layers.Dense(2, activation=tf.nn.softmax)
    keras.layers.Conv2D(32,(3,3), input_shape=(n,n,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64,(3,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(128,(3,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(128,(3,3)),
    keras.layers.Activation("relu"),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),

    keras.layers.Dense(2, activation=tf.nn.softmax)
    # keras.layers.Activation("softmax")


])

# model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, trainlabels, validation_split=0.2, epochs=e, batch_size = 10)

print("Time to train:", time.time()-start1)
start2 = time.time()


eval_loss1, eval_acc1 = model.evaluate(train, trainlabels)
# print("Time to Evaluate Train Accuracy:", time.time()-start2)



start3 = time.time()
test_loss1, test_acc1 = model.evaluate(test, testlabels)
# print("Time to Evaluate Test Accuracy:", time.time()-start3)

# print('Test accuracy 1:', test_acc1)
lossi1, acci1 = model.evaluate(test,yinfonp)
lossq1, accq1 = model.evaluate(test,yqualitynp)

print("Info acc:", acci1, "Qual acc:",  accq1)

predict_test = model.predict(test)
# predict_tt = model.predict(ttnp)
# print(predict_tt)
# # for i in range(len(predict_tt)):
# # 	if predict_tt[i][0]>=0.5:
# # 		print(ttnamenp[i],0)
# # 	if predict_tt[i][0]<0.5:
# # 		print(ttnamenp[i],1)

# predict_tt2 = model.predict(ttnp2)
# print(predict_tt2)

# predict_tt3 = model.predict(ttnp3)
# print(predict_tt3)

tp = 0
tn = 0
fp = 0
fn = 0

food_tp = 0
food_tn = 0
food_fp = 0
food_fn = 0

drink_tp = 0
drink_tn = 0
drink_fp = 0
drink_fn = 0

outside_tp = 0
outside_tn = 0
outside_fp = 0
outside_fn = 0

inside_tp = 0
inside_tn = 0
inside_fp = 0
inside_fn = 0

tplab = []
tnlab = []
fplab = []
fnlab = []
for i in range(len(predict_test)):
	if ycat[i] == 'food':

		if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			food_tp = food_tp + 1
		if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
			food_fp = food_fp + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
			food_fn = food_fn + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
			food_tn = food_tn + 1

	if ycat[i] == 'drink':
		if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			drink_tp = drink_tp + 1
		if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
			drink_fp = drink_fp + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
			drink_fn = drink_fn + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
			drink_tn = drink_tn + 1
	if ycat[i] == 'outside':
		if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			outside_tp = outside_tp + 1
		if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
			outside_fp = outside_fp + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
			outside_fn = outside_fn + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
			outside_tn = outside_tn + 1
	if ycat[i] == 'inside':
		if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			inside_tp = inside_tp + 1
		if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
			inside_fp = inside_fp + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
			inside_fn = inside_fn + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
			inside_tn = inside_tn + 1
	if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			tp = tp + 1
			tplab.append(ynamenp[i])
	if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
		fp = fp + 1
		fplab.append(ynamenp[i])
	if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
		fn = fn + 1
		fnlab.append(ynamenp[i])
	if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
		tn = tn + 1
		tnlab.append(ynamenp[i])

print(food_tp, food_tn, food_fp, food_fn)
# print("e =",e)
print("acc1food_ =",(food_tp+food_tn)/(food_tp+food_tn+food_fp+food_fn))
print("prec1food_ =",(food_tp)/(food_tp+food_fp))
# print('Train Accuracy 1:',eval_acc1)

print(drink_tp, drink_tn, drink_fp, drink_fn)
# print("e =",e)
print("acc1drink_ =",(drink_tp+drink_tn)/(drink_tp+drink_tn+drink_fp+drink_fn))
print("prec1drink_ =",(drink_tp)/(drink_tp+drink_fp))
# print('Train Accuracy 1:',eval_acc1)

print(outside_tp, outside_tn, outside_fp, outside_fn)
# print("e =",e)
print("acc1outside_ =",(outside_tp+outside_tn)/(outside_tp+outside_tn+outside_fp+outside_fn))
print("prec1outside_ =",(outside_tp)/(outside_tp+outside_fp))
# print('Train Accuracy 1:',eval_acc1)

print(inside_tp, inside_tn, inside_fp, inside_fn)
# print("e =",e)
print("acc1inside_ =",(inside_tp+inside_tn)/(inside_tp+inside_tn+inside_fp+inside_fn))
print("prec1inside_ =",(inside_tp)/(inside_tp+inside_fp))
# print('Train Accuracy 1:',eval_acc1)

print(tp, tn, fp, fn)
# print("e =",e)
print("acc1 =",(tp+tn)/(tp+tn+fp+fn))
print("prec1 =",(tp)/(tp+fp))
print('Train Accuracy 1:',eval_acc1)
# print("False Positives:")
# print(fplab)
# print("False Negatives:")
# print(fnlab)




###################################################################################################################################

#################################################################################################

start21 = time.time()
model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(n, n, 3)),
    # keras.layers.Dense(128, activation=tf.nn.relu),
    # keras.layers.Dense(2, activation=tf.nn.softmax)
    keras.layers.Conv2D(32,(3,3), input_shape=(n,n,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64,(3,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(128,(3,3)),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),

    keras.layers.Dense(2, activation=tf.nn.softmax)
    # keras.layers.Activation("softmax")


])

# model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train, trainlabels, validation_split=0.2, epochs=e, batch_size = 10)
print("run time of 2:", time.time()-start21)

eval_loss2, eval_acc2 = model.evaluate(train, trainlabels)



test_loss2, test_acc2 = model.evaluate(test, testlabels)

# print('Test accuracy 2:', test_acc2)

predict_test = model.predict(test)
predict_test = model.predict(test)

# predict_tt = model.predict(ttnp)
# print(predict_tt)

tp = 0
tn = 0
fp = 0
fn = 0
food_tp = 0
food_tn = 0
food_fp = 0
food_fn = 0

drink_tp = 0
drink_tn = 0
drink_fp = 0
drink_fn = 0

outside_tp = 0
outside_tn = 0
outside_fp = 0
outside_fn = 0

inside_tp = 0
inside_tn = 0
inside_fp = 0
inside_fn = 0
for i in range(len(predict_test)):
	if ycat[i] == 'food':

		if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			food_tp = food_tp + 1
		if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
			food_fp = food_fp + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
			food_fn = food_fn + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
			food_tn = food_tn + 1

	if ycat[i] == 'drink':
		if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			drink_tp = drink_tp + 1
		if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
			drink_fp = drink_fp + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
			drink_fn = drink_fn + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
			drink_tn = drink_tn + 1
	if ycat[i] == 'outside':
		if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			outside_tp = outside_tp + 1
		if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
			outside_fp = outside_fp + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
			outside_fn = outside_fn + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
			outside_tn = outside_tn + 1
	if ycat[i] == 'inside':
		if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
			inside_tp = inside_tp + 1
		if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
			inside_fp = inside_fp + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
			inside_fn = inside_fn + 1
		if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
			inside_tn = inside_tn + 1

print(food_tp, food_tn, food_fp, food_fn)
# print("e =",e)
print("acc1food_ =",(food_tp+food_tn)/(food_tp+food_tn+food_fp+food_fn))
print("prec1food_ =",(food_tp)/(food_tp+food_fp))
# print('Train Accuracy 1:',eval_acc1)

print(drink_tp, drink_tn, drink_fp, drink_fn)
# print("e =",e)
print("acc1drink_ =",(drink_tp+drink_tn)/(drink_tp+drink_tn+drink_fp+drink_fn))
print("prec1drink_ =",(drink_tp)/(drink_tp+drink_fp))
# print('Train Accuracy 1:',eval_acc1)

print(outside_tp, outside_tn, outside_fp, outside_fn)
# print("e =",e)
print("acc1outside_ =",(outside_tp+outside_tn)/(outside_tp+outside_tn+outside_fp+outside_fn))
print("prec1outside_ =",(outside_tp)/(outside_tp+outside_fp))
# print('Train Accuracy 1:',eval_acc1)

print(inside_tp, inside_tn, inside_fp, inside_fn)
# print("e =",e)
print("acc1inside_ =",(inside_tp+inside_tn)/(inside_tp+inside_tn+inside_fp+inside_fn))
print("prec1inside_ =",(inside_tp)/(inside_tp+inside_fp))

print('Train Accuracy 2:',eval_acc2)

###############################################################################################################################################
# start31 = time.time()
# model = keras.Sequential([
#     # keras.layers.Flatten(input_shape=(n, n, 3)),
#     # keras.layers.Dense(128, activation=tf.nn.relu),
#     # keras.layers.Dense(2, activation=tf.nn.softmax)
#     keras.layers.Conv2D(32,(3,3), input_shape=(n,n,3)),
#     keras.layers.Activation("relu"),
#     keras.layers.Conv2D(64,(3,3)),
#     keras.layers.Activation("relu"),
#     keras.layers.MaxPooling2D((2,2)),

#     keras.layers.Conv2D(128,(3,3)),
#     keras.layers.Activation("relu"),
#     keras.layers.Conv2D(128,(3,3)),
#     keras.layers.Activation("relu"),
#     keras.layers.MaxPooling2D((2,2)),

#     keras.layers.Flatten(),

#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(64, activation=tf.nn.relu),
#     # keras.layers.Dense(32, activation=tf.nn.relu),

#     keras.layers.Dense(2, activation=tf.nn.softmax)
#     # keras.layers.Activation("softmax")


# ])

# # model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(train, trainlabels, validation_split=0.2, epochs=e, batch_size = 10)

# print("run time of 3:", time.time()-start31)
# eval_loss3, eval_acc3 = model.evaluate(train, trainlabels)



# test_loss3, test_acc3 = model.evaluate(test, testlabels)

# # print('Test accuracy 3:', test_acc3)

# predict_test = model.predict(test)

# tp = 0
# tn = 0
# fp = 0
# fn = 0
# for i in range(len(predict_test)):
# 	if (predict_test[i][1]>=0.5) & (testlabels[i]=='1'):
# 		tp = tp + 1
# 	if(predict_test[i][1]>=0.5) & (testlabels[i]=='0'):
# 		fp = fp + 1
# 	if(predict_test[i][1]<0.5) & (testlabels[i]=='1'):
# 		fn = fn + 1
# 	if(predict_test[i][1]<0.5) & (testlabels[i]=='0'):
# 		tn = tn + 1
# print(tp, tn, fp, fn)
# print("acc3 =",(tp+tn)/(tp+tn+fp+fn))
# print("prec3 =",(tp)/(tp+fp))
# print('Train Accuracy 3:',eval_acc3)

# print('Train Accuracy 1:',eval_acc1)
# print('Test accuracy 1:', test_acc1)

# print('Train Accuracy 2:',eval_acc2)
# print('Test accuracy 2:', test_acc2)

# print('Train Accuracy 3:',eval_acc3)
# print('Test accuracy 3:', test_acc3)