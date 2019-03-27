import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xlwt
import xlrd
from scipy.spatial.distance import cdist

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image



def progress(percent,width=50):
    if percent >= 100:
        percent=100
  
    show_str=('[%%-%ds]' %width) %(int(width * percent/100)*"#") 
    print('\r%s %d%%' %(show_str,percent),end='')

k=10
img_paths=sorted(glob.glob('./images/*.jpg'))#get the image paths
Features=[]#store the features
print('Extracting features......')
vgg16=VGG16(weights='imagenet')
model=Model(input=vgg16.input, output=vgg16.get_layer('flatten').output)

Features=[]
for i,path in enumerate(img_paths):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    Features.append(model.predict(x).reshape((1,-1)))
    progress(i/len(img_paths)*100,width=50)

# Use PCA to reduce the dim
print('\nPCA......')
Features=np.concatenate(Features,axis=0)
pca=PCA(n_components=0.99)#keep 99% of the information
Features=pca.fit_transform(Features)
print('The reduced dimension={}'.format(Features.shape[1]))
# After reducing the dimension,the dim = 4390

# use the kmeans to cluster
print('\nKmeans Clustering......')
km=KMeans(n_clusters=k)
km=km.fit(Features)
labels=km.labels_
centers=km.cluster_centers_
L=np.sort(np.unique(labels))

shows=[]#store the nearest images to the corresponding cluster center
for l in range(k):
    ind_label=np.nonzero(labels==L[l])[0]
    dist=cdist(centers[l:l+1,:],Features[ind_label,:])
    ind=np.argsort(dist,axis=1)[0]
    if ind.shape[0]>=6:#if there are at least 6 pictures
        shows.append(ind_label[ind[:6]])
    else:#there are only less than 6 pictures
        shows.append(ind_label[ind])

# save the result into the Sample_submission.csv
print('Typing the results......')
book = xlwt.Workbook()
sheet=book.add_sheet('Kmeans',cell_overwrite_ok=True)
for i in range(len(L)):
    sheet.write(0,L[i],'Cluster {}'.format(L[i]+1))
    ind=np.where(labels==L[i])[0]
    for j in range(len(ind)):
        sheet.write(j+1,L[i],"\'"+img_paths[ind[j]].split('\\')[-1].split('.')[0]+"\'")

book.save('A3_hchenbw_12233738_prediction.csv')
#show some examples
book=xlrd.open_workbook('A3_hchenbw_12233738_prediction.csv')
sheet=book.sheet_by_name('Kmeans')

#save the images
for i in range(k):
    plt.figure(num='Cluster {}'.format(i+1),figsize=(30,20))
    for j in range((shows[i].shape[0])):
        plt.subplot(2,3,j+1)
        plt.imshow(plt.imread(img_paths[shows[i][j]]))

    plt.savefig('Cluster{}.jpg'.format(i+1))