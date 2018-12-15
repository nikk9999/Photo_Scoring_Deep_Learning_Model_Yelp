from imutils import paths
import argparse
import cv2
import os
import glob
import csv
import pandas as pd
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from PIL import Image
import math
import numpy as np
# import shutil

i = 0
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
with open('NimaAnalysis.csv') as csv_file:
	reader = csv.reader(csv_file,delimiter = ',')
	for row in reader:
		# print(row[1])
		if i == 0:
			i = i + 1
			continue
		else:
		
			i = i+1
			src = 'labeledYelp/'+row[12]+'.jpg'
			ourlabel = int(row[9])
			if (float(row[2]) >= 0.5):
				modellabel = 1
			if (float(row[2]) < 0.5):
				modellabel = 0

			if modellabel == 0:
				try:
					n = 128
					image = cv2.imread(src)
					imageresize = cv2.resize(image,(n,n), interpolation=cv2.INTER_CUBIC)
					gray = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
					# print("here")
					temp = gray
					hist = np.bincount(temp.ravel(),minlength=256)+1
					hist=hist/float(np.sum(hist))
					entropy=np.sum(-hist*np.log2(hist))

					brightness = 0
					min_bright = 0
					max_bright = 255*128*128
					for i in range(128):
						for j in range(128):
							brightness = brightness+gray[i][j]

					fm = cv2.Laplacian(gray, cv2.CV_64F).var()

					if(fm > 1000) & (brightness > 1500000):
						modellabel = 1
					else:
						modellabel = 0
				except:
					modellabel = 0

			dest = ""
			if row[5] == 'food':
				if ourlabel == 1:
					if modellabel == 1:
						food_tp = food_tp + 1
						dest = 'NimaPhotos/newcombined/tp/'+row[12]+'.jpg'
						# os.rename(src, dest)
					if modellabel == 0:
						food_fn = food_fn + 1
						dest = 'NimaPhotos/newcombined/fn/'+row[12]+'.jpg'
						# os.rename(src, dest)
				if ourlabel == 0:
					if modellabel == 1:
						food_fp = food_fp + 1
						dest = 'NimaPhotos/newcombined/fp/'+row[12]+'.jpg'
						# os.rename(src, dest)
					if modellabel == 0:
						food_tn = food_tn + 1
						dest = 'NimaPhotos/newcombined/tn/'+row[12]+'.jpg'
					# os.rename(src, dest)

			if row[5] == 'drink':
				if ourlabel == 1:
					if modellabel == 1:
						drink_tp = drink_tp + 1
						dest = 'NimaPhotos/newcombined/tp/'+row[12]+'.jpg'
						# os.rename(src, dest)
					if modellabel == 0:
						drink_fn = drink_fn + 1
						dest = 'NimaPhotos/newcombined/fn/'+row[12]+'.jpg'
						# os.rename(src, dest)
				if ourlabel == 0:
					if modellabel == 1:
						drink_fp = drink_fp + 1
						dest = 'NimaPhotos/newcombined/fp/'+row[12]+'.jpg'
						# os.rename(src, dest)
					if modellabel == 0:
						drink_tn = drink_tn + 1
						dest = 'NimaPhotos/newcombined/tn/'+row[12]+'.jpg'

			if row[5] == 'outside':
				if ourlabel == 1:
					if modellabel == 1:
						outside_tp = outside_tp + 1
						dest = 'NimaPhotos/newcombined/tp/'+row[12]+'.jpg'
						# os.rename(src, dest)
					if modellabel == 0:
						outside_fn = outside_fn + 1
						dest = 'NimaPhotos/newcombined/fn/'+row[12]+'.jpg'
						# os.rename(src, dest)
				if ourlabel == 0:
					if modellabel == 1:
						outside_fp = outside_fp + 1
						dest = 'NimaPhotos/newcombined/fp/'+row[12]+'.jpg'
						# os.rename(src, dest)
					if modellabel == 0:
						outside_tn = outside_tn + 1
						dest = 'NimaPhotos/newcombined/tn/'+row[12]+'.jpg'

			if row[5] == 'inside':
				if ourlabel == 1:
					if modellabel == 1:
						inside_tp = inside_tp + 1
						dest = 'NimaPhotos/newcombined/tp/'+row[12]+'.jpg'
						# os.rename(src, dest)
					if modellabel == 0:
						inside_fn = inside_fn + 1
						dest = 'NimaPhotos/newcombined/fn/'+row[12]+'.jpg'
						# os.rename(src, dest)
				if ourlabel == 0:
					if modellabel == 1:
						inside_fp = inside_fp + 1
						dest = 'NimaPhotos/newcombined/fp/'+row[12]+'.jpg'
						# os.rename(src, dest)
					if modellabel == 0:
						inside_tn = inside_tn + 1
						dest = 'NimaPhotos/newcombined/tn/'+row[12]+'.jpg'

			# if dest!='':
			# 	try:
			# 		os.rename(src, dest)
			# 	except:
			# 		continue

print("For Food")
print(food_tp, food_fp, food_tn, food_fn)
print("Accuracy is:", (food_tp+food_tn)/(food_tp+food_tn+food_fp+food_fn))
print("Precision is:", (food_tp)/(food_tp+food_fp))
print("Number of False Positives are:", food_fp)

print("For Drink")
print(drink_tp, drink_fp, drink_tn, drink_fn)
print("Accuracy is:", (drink_tp+drink_tn)/(drink_tp+drink_tn+drink_fp+drink_fn))
print("Precision is:", (drink_tp)/(drink_tp+drink_fp))
print("Number of False Positives are:", drink_fp)

print("For Inside")
print(inside_tp, inside_fp, inside_tn, inside_fn)
print("Accuracy is:", (inside_tp+inside_tn)/(inside_tp+inside_tn+inside_fp+inside_fn))
print("Precision is:", (inside_tp)/(inside_tp+inside_fp))
print("Number of False Positives are:", inside_fp)

print("For Outside")
print(outside_tp, outside_fp, outside_tn, outside_fn)
print("Accuracy is:", (outside_tp+outside_tn)/(outside_tp+outside_tn+outside_fp+outside_fn))
print("Precision is:", (outside_tp)/(outside_tp+outside_fp))
print("Number of False Positives are:", outside_fp)

print("Total")
tp = outside_tp + inside_tp + food_tp + drink_tp
tn = outside_tn + inside_tn + food_tn + drink_tn
fp = outside_fp + inside_fp + food_fp + drink_fp
fn = outside_fn + inside_fn + food_fn + drink_fn
print(tp,fp,tn,fn)
print("Accuracy is:", (tp+tn)/(tp+tn+fp+fn))
print("Precision is:",(tp)/(tp+fp))



