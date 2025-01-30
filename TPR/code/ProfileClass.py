import time
import sys
import warnings
import numpy as np
import seaborn as sns
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import IsolationForest
from more_itertools import tabulate
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from scipy.stats import multivariate_normal
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    cObs,cFea=oClass.shape
    colors = ['g', 'r', 'b', 'y']  
    for i in range(nObs):
        if i < cObs: 
            plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    plt.close()


def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))


def printAdvancedStats(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (recall * precision) / (recall + precision) if (recall + precision) != 0 else 0

    table = [["Metric", "Value"],
             ["Accuracy", "{:.2%}".format(accuracy)],
             ["Precision", "{:.2%}".format(precision)],
             ["Recall", "{:.2%}".format(recall)],
             ["F1-Score", "{:.2f}".format(f1_score)]]

    print(tabulate(table, tablefmt="fancy_grid"))

    def plot_confusion_matrix(tp, tn, fp, fn, classes):
        cm = [[tn, fp],
            [fn, tp]]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
    classes = ["Normal", "Anomaly"]
    plot_confusion_matrix(tp, tn, fp, fn, classes)


Classes={0:'Client',1:"Attacker"}
nfig=1

NormalFeatures_c1=np.loadtxt("../Pkt_Features/aluno_features_m2_w[120]_s60.dat")       # normal user
NormalFeatures_c2=np.loadtxt("../Pkt_Features/professor_features_m2_w[120]_s60.dat")       # normal user
NormalFeatures_botBasic=np.loadtxt("../Pkt_Features/basic_features_m2_w[120]_s60.dat")     # isolated bot basic
NormalFeatures_botAdvanced=np.loadtxt("../Pkt_Features/advanced_features_m2_w[120]_s60.dat")     # isolated bot advanced

oClass_client1=np.ones((len(NormalFeatures_c1),1))*0
oClass_client2=np.ones((len(NormalFeatures_c2),1))*0
oClass_bot=np.ones((len(NormalFeatures_botAdvanced),1))*1


features=np.vstack((NormalFeatures_c1,NormalFeatures_c2))
oClass=np.vstack((oClass_client1, oClass_client2))
features1=np.vstack((NormalFeatures_c1,NormalFeatures_c2,NormalFeatures_botAdvanced))
oClass1=np.vstack((oClass_client1, oClass_client2, oClass_bot))
print('Train Stats Features Size:',features.shape)
print('Classes Size: ', oClass.shape)

def plot_features(features, labels, f1index, f2index, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    nObs, nFea = features.shape
    colors = ['g', 'r', 'b', 'y']  # g: green, r: red, b: blue, y: yellow
    for i in range(nObs):
        plt.scatter(features[i, f1index], features[i, f2index], color=colors[int(labels[i])])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


percentage = 0.5
pC1 = int(len(NormalFeatures_c1) * percentage)
pC2 = int(len(NormalFeatures_c2) * percentage)
pC3 = int(len(NormalFeatures_botAdvanced) * percentage)


trainFUser = np.vstack((NormalFeatures_c1[:pC1, :], NormalFeatures_c2[:pC2, :]))    # TRAIN DATASET FOR ANOMALY DETECTION WITH THE FIRST 70% OF BOTH CLIENTS
trainCUser = np.vstack((oClass_client1[:pC1], oClass_client2[:pC2]))

testFUser = np.vstack((NormalFeatures_c1[pC1:, :], NormalFeatures_c2[pC2:, :]))
testCUser = np.vstack((oClass_client1[pC1:], oClass_client2[pC2:]))        # TEST DATASET FOR ANOMALY DETECTION AND TRAFFIC CLASSIFICATION WITH THE LAST 30% OF BOTH CLIENTS AND THE BOT

testFBot = NormalFeatures_botAdvanced[pC3:, :] 
testCBot = oClass_bot


#############----Feature Normalization----#############
scaler = MaxAbsScaler().fit(trainFUser)

trainFeaturesUser=scaler.transform(trainFUser)
testFeaturesUser=scaler.transform(testFUser)
testFeaturesBot=scaler.transform(testFBot)

#############----PCA----#############
pca = PCA(n_components=28)

train_pca=pca.fit(trainFeaturesUser)
trainPCA = train_pca.transform(trainFeaturesUser)
testPCA_User = train_pca.transform(testFeaturesUser)
testPCA_Bot = train_pca.transform(testFeaturesBot)


cumulative_variance_ratio = np.cumsum(train_pca.explained_variance_ratio_)
print("Cumulative explained variance ratio:", cumulative_variance_ratio)
print("Explained variance ratio:", train_pca.explained_variance_ratio_)


"""
#############----Anomaly Detection based on centroids distances----#############
centroids={}
pClass=(trainCUser==0).flatten()
centroids.update({0:np.mean(trainFeaturesUser[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)


tp = 0 #True Positive
tn = 0 #True Negative
fp = 0 #False Positive
fn = 0 #False Negative

AnomalyThreshold=10
print('\n-- Anomaly Detection based on Centroids Distances --')
nObsTest,nFea=testFeaturesBot.shape
for i in range(nObsTest):
    x=testFeaturesBot[i]
    dists=[distance(x,centroids[0])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        #True Positive
        tp += 1
    else:
        result="OK"
        #False Negative
        fn += 1
    # print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testCBot[i][0]],*dists,result))

nObsTest,nFea=testFeaturesUser.shape
for i in range(nObsTest):
    x=testFeaturesUser[i]
    dists=[distance(x,centroids[0])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        #False Positive
        fp += 1
    else:
        result="OK"
        #True Negative
        tn += 1
    # print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testCUser[i][0]],*dists,result))

printAdvancedStats(tp, tn, fp, fn)


#############----Anomaly Detection based on centroids distances (PCA Features)----#############
centroids={}
pClass=(trainCUser==0).flatten() 
centroids.update({0:np.mean(trainPCA[pClass,:],axis=0)})


tp = 0 #True Positive
tn = 0 #True Negative
fp = 0 #False Positive
fn = 0 #False Negative

AnomalyThreshold=10
print('\n-- Anomaly Detection based on Centroids Distances (PCA Features)--')
nObsTest,nFea=testPCA_Bot.shape

for i in range(nObsTest):
    x=testPCA_Bot[i]
    dists=[distance(x,centroids[0])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        #True Positive
        tp += 1
    else:
        result="OK"
        #False Negative
        fn += 1
    # print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testCBot[i][0]],*dists,result))

nObsTest,nFea=testPCA_User.shape
for i in range(nObsTest):
    x=testPCA_User[i]
    dists=[distance(x,centroids[0])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        #False Positive
        fp += 1
    else:
        result="OK"
        #True Negative
        tn += 1
    # print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testCUser[i][0]],*dists,result))

printAdvancedStats(tp, tn, fp, fn)



#############----Anomaly Detection based on One Class Support Vector Machines (PCA Features)---#############
print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainPCA)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainPCA)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainPCA)  

tpL, tnL, fpL, fnL = 0, 0, 0, 0
tpRBF, tnRBF, fpRBF, fnRBF = 0, 0, 0, 0
tpP, tnP, fpP, fnP = 0, 0, 0, 0


L1=ocsvm.predict(testPCA_Bot)
L2=rbf_ocsvm.predict(testPCA_Bot)
L3=poly_ocsvm.predict(testPCA_Bot)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=testFeaturesBot.shape
for i in range(nObsTest):    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        tpL += 1
    else:
        fnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        tpRBF += 1
    else:
        fnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        tpP += 1
    else:
        fnP += 1

    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testCBot[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


L1=ocsvm.predict(testPCA_User)
L2=rbf_ocsvm.predict(testPCA_User)
L3=poly_ocsvm.predict(testPCA_User)
testFeaturesUser
nObsTest,nFea=testFeaturesUser.shape
for i in range(nObsTest):    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        fpL += 1
    else:
        tnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        fpRBF += 1
    else:
        tnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        fpP += 1
    else:
        tnP += 1

    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testCUser[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


print("\nKernel Linear Statistics")
printAdvancedStats(tpL, tnL, fpL, fnL)
print("\nKernel RBF Statistics")
printAdvancedStats(tpRBF, tnRBF, fpRBF, fnRBF)
print("\nKernel Poly Statistics")
printAdvancedStats(tpP, tnP, fpP, fnP)



#############----Anomaly Detection based on One Class Support Vector Machines----#############
print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesUser)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesUser)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesUser)  

tpL, tnL, fpL, fnL = 0, 0, 0, 0
tpRBF, tnRBF, fpRBF, fnRBF = 0, 0, 0, 0
tpP, tnP, fpP, fnP = 0, 0, 0, 0


L1=ocsvm.predict(testFeaturesBot)
L2=rbf_ocsvm.predict(testFeaturesBot)
L3=poly_ocsvm.predict(testFeaturesBot)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=testPCA_Bot.shape
for i in range(nObsTest):    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        tpL += 1
    else:
        fnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        tpRBF += 1
    else:
        fnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        tpP += 1
    else:
        fnP += 1

    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testCBot[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


L1=ocsvm.predict(testFeaturesUser)
L2=rbf_ocsvm.predict(testFeaturesUser)
L3=poly_ocsvm.predict(testFeaturesUser)

nObsTest,nFea=testFeaturesUser.shape
for i in range(nObsTest):    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        fpL += 1
    else:
        tnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        fpRBF += 1
    else:
        tnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        fpP += 1
    else:
        tnP += 1


    # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testCUser[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


print("\nKernel Linear Statistics")
printAdvancedStats(tpL, tnL, fpL, fnL)
print("\nKernel RBF Statistics")
printAdvancedStats(tpRBF, tnRBF, fpRBF, fnRBF)
print("\nKernel Poly Statistics")
printAdvancedStats(tpP, tnP, fpP, fnP)


######################----K means----######################
print("############################################")
print("K-Means Clustering Statistics")

kmeans = KMeans(n_clusters=3, random_state=0).fit(trainFeaturesUser)
centroids = kmeans.cluster_centers_

# Define a function to calculate distance to the nearest centroid
def distance_to_centroid(point, centroids):
    return np.min([np.linalg.norm(point - c) for c in centroids])

distances = [distance_to_centroid(point, centroids) for point in trainFeaturesUser]

# Set your threshold for anomaly detection
kmeans_threshold = 0.5

# Testing with PCA features
tp, tn, fp, fn = 0, 0, 0, 0
for point in testFeaturesBot:
    if distance_to_centroid(point, centroids) > kmeans_threshold:
        tp += 1
    else:
        fn += 1

for point in testFeaturesUser:
    if distance_to_centroid(point, centroids) > kmeans_threshold:
        fp += 1
    else:
        tn += 1

print("\nK-Means Clustering Statistics")
printAdvancedStats(tp, tn, fp, fn)


######################----ISOLATION FOREST (PCA features)----######################
print("############################################")
print("Isolation Forest(PCA Features)")

# Isolation Forest with PCA features
iso_forest = IsolationForest(random_state=0).fit(trainPCA)

# Testing with PCA features
tp, tn, fp, fn = 0, 0, 0, 0
iso_forest_labels_bot = iso_forest.predict(testPCA_Bot)
iso_forest_labels_user = iso_forest.predict(testPCA_User)


# Counting TP, TN, FP, FN
tp = np.sum(iso_forest_labels_bot == -1)
fn = len(iso_forest_labels_bot) - tp
fp = np.sum(iso_forest_labels_user == -1)
tn = len(iso_forest_labels_user) - fp


# print("\nIsolation Forest Statistics (PCA Features)")
printAdvancedStats(tp, tn, fp, fn)



print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
means={}
pClass=(trainCUser==0).flatten() 
covs={}
pClass=(trainCUser==0).flatten()

means.update({0: np.mean(trainPCA[pClass,:], axis=0)})
covs.update({0: np.cov(trainPCA[pClass,:], rowvar=False) + np.eye(trainPCA.shape[1]) * 1e-6})  # Regularization

all_probs = np.array([multivariate_normal.logpdf(x, means[0], covs[0]) for x in trainPCA])
AnomalyThreshold = np.percentile(all_probs, 5)  # Set threshold at the 5th percentile

# Initialize Confusion Matrix Variables
tp, tn, fp, fn = 0, 0, 0, 0

# Test on Bot Data
for i in range(testPCA_Bot.shape[0]):
    x = testPCA_Bot[i,:]
    prob = multivariate_normal.logpdf(x, means[0], covs[0])
    
    if prob < AnomalyThreshold:
        tp += 1  # True Positive (Anomaly correctly detected)
    else:
        fn += 1  # False Negative (Missed Anomaly)

# Test on Normal User Data
for i in range(testPCA_User.shape[0]):
    x = testPCA_User[i,:]
    prob = multivariate_normal.logpdf(x, means[0], covs[0])
    
    if prob < AnomalyThreshold:
        fp += 1  # False Positive (Normal incorrectly flagged)
    else:
        tn += 1  # True Negative (Normal correctly classified)

# Print statistics
printAdvancedStats(tp, tn, fp, fn)

"""