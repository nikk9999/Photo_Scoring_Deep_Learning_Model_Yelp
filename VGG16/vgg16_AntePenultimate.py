from datetime import datetime
start=datetime.now()
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.externals import joblib
#print("Hi\n")
model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)
#model.summary()
import glob
gl1 = glob.glob("trainingYelp/*.jpg")
filenames1 = []
#print(gl1)
for glo in gl1:
    filenames1.append(glo.split("/")[1].split(".")[0])
vgg16_feature_list1 = []
# subdir = "labelledyelp/"
#print(gl1)
for img_path in gl1:        # process the files under the directory 'dogs' or 'cats'
        # ...
#        print("Entered" + img_path)
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        vgg16_feature = model.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list1.append(vgg16_feature_np.flatten())
vgg16_feature_list_np1 = np.array(vgg16_feature_list1)
#np.savez("foo.npz",vgg16_feature_list_np1)
df1 = pd.read_csv('Final_Train_Nov29.csv')
# df1.columns = ["no", "photoId", "semi", "score", "equal", "value"]
scores1 = []
labels1 = []
category = []
for file in filenames1:
    f = df1.loc[df1['photo_id'] == file]
    if len(f)>0:
            scores1.append(f.withmetrics.item())
            labels1.append(f.label.item())
            category.append(f.clarifai.item())
scores1 = np.array(scores1)
labels1 = np.array(labels1)
category = np.array(category)
your_pca = PCA(n_components=15000)
vgg16_feature_list_np1 = your_pca.fit_transform(vgg16_feature_list_np1)
print(your_pca.explained_variance_)
print(your_pca.explained_variance_ratio_)
print(your_pca.explained_variance_ratio_.cumsum())
#np.save("vgg16_feature_list_np1",vgg16_feature_list_np1)
all_data1 = np.column_stack([vgg16_feature_list_np1,scores1,category])
np.save('all_data1',all_data1)
clf1 = SVC(gamma='auto')
clf1.fit(all_data1, labels1)
print(datetime.now()-start)
filename = 'classifier1.joblib.pkl'
_ = joblib.dump(clf1, filename, compress=9)
from sklearn.metrics import accuracy_score, precision_score
y_pred1 = clf1.predict(all_data1)
print(accuracy_score(labels1, y_pred1))
from sklearn.metrics import accuracy_score, precision_score
print(precision_score(labels1, y_pred1))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(labels1, y_pred1))
                                                            