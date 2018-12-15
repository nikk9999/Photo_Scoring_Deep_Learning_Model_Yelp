from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from keras.models import Model


model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)
model.summary()

df1 = pd.read_csv("../outAlex1.csv")
import glob
gl1 = glob.glob("/home/nihira/Documents/CS6220BigData/Project/neural-image-assessment-master/trainingYelp/*.jpg")

filenames1 = []
for glo in gl1:
    filenames1.append(glo.split("/")[8].split(".")[0])
vgg16_feature_list1 = []
# subdir = "labelledYelp/"
for img_path in gl1:        # process the files under the directory 'dogs' or 'cats'
        # ...

        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list1.append(vgg16_feature_np.flatten())
        
vgg16_feature_list_np1 = np.array(vgg16_feature_list1)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(vgg16_feature_list_np)
df1 = pd.read_csv('../trainLabelsVGG.csv')
# df1.columns = ["no", "photoId", "semi", "score", "equal", "value"]
photos = pd.read_csv("../samplePhots.csv")

scores1 = []
labels1 = []
category = []
for file in filenames1:
    f = df1.loc[df1['photoIdOrig'] == file]
    scores1.append(df1.loc[df1['photoIdOrig'] == file].value.item())
    labels1.append(f.norm.item())
    category.append(photos.loc[photos['photo_id'] == file].label.item())

scores1 = np.array(scores1)
labels1 = np.array(labels1)
category = np.array(category)

all_data1 = np.column_stack([vgg16_feature_list_np1,scores1,category])

clf1 = SVC(gamma='auto')
clf1.fit(all_data1, labels1) 

from sklearn.metrics import accuracy_score, precision_score
y_pred1 = clf1.predict(all_data)
accuracy_score(labels, y_pred1)

from sklearn.metrics import accuracy_score, precision_score
precision_score(labels, y_pred1)