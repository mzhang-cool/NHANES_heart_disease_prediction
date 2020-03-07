import matplotlib
import pandas as pd
import numpy as np
from docutils.nodes import inline
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pydotplus
from sklearn import tree
import collections
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
import seaborn as sns
import pickle


df = pd.read_csv("/Users/Tim/Desktop/scor_test/TESTCODE/all_years.csv", encoding='utf8',index_col = 0)

#Using Pearson Correlation
cor = df.corr()
#Correlation with output variable
cor_target = abs(cor["MCQ160C"])
#Selecting highly correlated features & Checking if there's any highly correlated features which need to be removed
relevant_features = cor_target[cor_target>0.3]
relevant_features

# clean data, remove the columns whose percentage of missing value is greater than 20%
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
print(missing_value_df)
for index, row in missing_value_df.iterrows():
    if (row['percent_missing'] >20) & (row['column_name'] != 'MCQ160C'):
        df = df.drop(columns=row['column_name'])
        
df=df.drop(columns=['RXDDRGID'])
df=df[pd.notnull(df['MCQ160C'])]

UniqueSEQN=df.drop_duplicates(subset='SEQN', keep='first')
UniqueSEQN.to_csv("/Users/Tim/Desktop/scor_test/TESTCODE/UniqueSEQN.csv")

UniqueSEQN = pd.read_csv("/Users/Tim/Desktop/scor_test/TESTCODE/UniqueSEQN.csv", encoding='utf8',index_col = 0)

#Remove redundant variables
UniqueSEQN=UniqueSEQN.drop(columns=['DMDCITZN'])
UniqueSEQN=UniqueSEQN.drop(columns=['PHDSESN'])
UniqueSEQN=UniqueSEQN.drop(columns=['SDDSRVYR'])
UniqueSEQN = UniqueSEQN[UniqueSEQN['SEQN'].notnull()]

categorical_features = ['SEQN','DIQ050', 'DIQ010', 'HSQ520', 'HOQ065', 'HSAQUEX', 'HSQ500','PHQ060','HUQ090',
                        'DMDHRGND', 'BPXPULS', 'RIDSTATR', 'PHQ040', 'PHQ050', 'RIDEXMON', 'PHQ020',
                        'PHQ030', 'SDMVPSU', 'DMDHRMAR', 'HUQ010', 'RIAGENDR','IMQ020','MCQ092','HIQ210','HUQ020',
                       'MCQ160C','MCQ010','HSQ510','MCQ053','HUQ030','RIDRETH1']
numerical_features = ['SEQN','DMDHREDU','BMXWT','BMXWAIST','LBXNEPCT','LBXLYPCT','BMXARMC','LBXBAPCT','LBDNENO','RIDAGEYR',
                      'HOD050','DMDHHSIZ','LBDLYMNO','URXUMA','BMXBMI','BMXARML','SDMVSTRA','LBXEOPCT',
                      'INDFMPIR','DMDHRAGE','BMXHT','WTINT2YR','LBXMPSI','LBXMCHSI','LBXPLTSI','LBDMONO','LBXRDW',
                      'LBXWBCSI','LBXMOPCT','LBDEONO','LBXRBCSI','WTMEC2YR','LBXMCVSI','LBDBANO','LBXHGB']

# Missing Value 
# Remove the missing value of the categorical_features
for feature in categorical_features:
    UniqueSEQN = UniqueSEQN[UniqueSEQN[feature].notnull()]
    print ('Dropped missing records in %s' % feature)
# Fill missing value of numerical_features with KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
UniqueSEQN[numerical_features] = imputer.fit_transform(UniqueSEQN[numerical_features])

# Change the categorical_features into dummies
UniqueSEQN2 = UniqueSEQN[categorical_features]
UniqueSEQN2 = UniqueSEQN2[categorical_features].astype('category')
print(UniqueSEQN2.info())
SEQN2 = pd.DataFrame(UniqueSEQN2['SEQN'])
MCQ160C = pd.DataFrame(UniqueSEQN2['MCQ160C'])
del UniqueSEQN2['SEQN']
del UniqueSEQN2['MCQ160C']
UniqueSEQN2colname=UniqueSEQN2.columns

UniqueSEQN2=pd.get_dummies(UniqueSEQN2) 
SEQN2 = SEQN2.join(MCQ160C)
UniqueSEQN2 = SEQN2.join(UniqueSEQN2)

# Remove Outlier of the numerical_features
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(UniqueSEQN[numerical_features]))
UniqueSEQN1 = UniqueSEQN[numerical_features][(z < 3).all(axis=1)]
SEQN1 = pd.DataFrame(UniqueSEQN1['SEQN'])
del UniqueSEQN1['SEQN']
UniqueSEQN1colname=UniqueSEQN1.columns

#Standardize te numerical variables
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
UniqueSEQN1_scaled = min_max_scaler.fit_transform(UniqueSEQN1)
UniqueSEQN1 = pd.DataFrame(UniqueSEQN1_scaled, columns=UniqueSEQN1colname)

UniqueSEQN1 = SEQN1.join(UniqueSEQN1)

# get the correlated info
corr_UniqueSEQN1 = UniqueSEQN1.corr()
# use seaborn for the visualization
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline
# make the edge
plt.figure(figsize=(8,6))
# draw the heatmap
sns.heatmap(corr_UniqueSEQN1, cmap='YlGnBu')


#Remove the highly correlated variables
del UniqueSEQN1['BMXWT']
del UniqueSEQN1['BMXARMC']
del UniqueSEQN1['DMDHRAGE']
del UniqueSEQN1['LBDNENO']
del UniqueSEQN1['LBDEONO']
del UniqueSEQN1['LBXRBCSI']
del UniqueSEQN1['WTMEC2YR']
del UniqueSEQN1['LBXMCVSI']
del UniqueSEQN1['BMXWAIST']
del UniqueSEQN1['LBDBANO']
del UniqueSEQN1['BMXARML']
print(UniqueSEQN1.info())

#Combine two variable type
newdftest = pd.merge(UniqueSEQN1, UniqueSEQN2, how='outer',on='SEQN')
newdf = newdftest[newdftest['BMXBMI'].notnull()]

#Re-define the target variables
newdf['MCQ160C']=newdf['MCQ160C'].replace(2.0, "No")
newdf['MCQ160C']=newdf['MCQ160C'].replace(1.0, "Yes")
newdf = newdf[newdf['MCQ160C']!=9.0]
newdf = newdf[newdf['MCQ160C']!=7.0]
newdf["MCQ160C"].value_counts()

# Undersampling to avoid the Data Leakage problem 
np.random.seed(666)
remove_n = 220000
df1 = newdf[newdf["MCQ160C"]!="No"]
df2 = newdf[newdf["MCQ160C"]=="No"]
drop_indices = np.random.choice(df2.index, remove_n, replace=False)
df2 = df2.drop(drop_indices)
selected_df=pd.concat([df1,df2],axis=0,join='inner')
selected_df["MCQ160C"].value_counts()
del selected_df['SEQN']
selected_df.to_csv("/Users/Tim/Desktop/scor_test/TESTCODE/finaldf.csv")
# & Oversampling with SMOTE
# Apply smote
#X_train = selected_df.drop(['MCQ160C'],axis=1)
#y_train = selected_df['MCQ160C']
#smt = SMOTE(random_state=666)
#X_train, y_train = smt.fit_resample(X_train, y_train)
#selected_dfaftersmote=pd.concat([y_train,X_train],axis=1,join='inner')
#selected_dfaftersmote["MCQ160C"].value_counts()

# Prepare the training & testing Dataset 
X_train = selected_df.drop(['MCQ160C'],axis=1)
y_train = selected_df['MCQ160C']
#creat your train and test set
X = X_train
Y = y_train
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=0.2,random_state=666,shuffle=True,stratify=Y)
# Use Random forest to do features selection
model = RandomForestClassifier()
model = model.fit(X_train, Y_train)
y_pred = model.predict(X_validation)

feature_imp = pd.Series(model.feature_importances_,index=list(X.columns)).sort_values(ascending=False)
list(feature_imp[0:20].keys())

#Manually checking
my_list=['RIAGENDR_2.0','LBXPLTSI','BMXHT','LBXBAPCT', 'BMXBMI',
 'LBXMOPCT','RIDAGEYR', 'LBXMCHSI','LBXRDW','URXUMA','LBXWBCSI',
 'LBXEOPCT','LBXLYPCT','LBXNEPCT','MCQ160C','DMDHREDU']
selected_df2 = selected_df[my_list]
X_train = selected_df2.drop(['MCQ160C'],axis=1)
y_train = selected_df2['MCQ160C']
#creat your train and test set
X = X_train
Y = y_train
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=0.2,random_state=666,shuffle=True,stratify=Y)

#Build Model / Compare the models with training set and testing set
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('MNB', MultinomialNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('SVM',SVC(gamma='auto',max_iter=3000,probability=True)))
# evaluate each model in turn
results = []
names = []
print('Testing Data')
for name, model in models:
    model = model.fit(X_train, Y_train)
    y_pred = model.predict(X_validation)
    precision, recall, fscore, train_support = score(Y_validation, y_pred, pos_label='Yes', average='binary')
    accuracy = acs(Y_validation,y_pred)
    results.append(recall)
    results.append(fscore)
    results.append(accuracy)
    names.append(name)
    msg = "%s: %f %f %f" % (name, recall.mean(), fscore.mean(), accuracy.mean())
    print(msg)

print('\n')

print('Training Data')
for name, model in models:
    model = model.fit(X_train, Y_train)
    trainpred = model.predict(X_train)
    precision, recall, fscore, train_support = score(Y_train, trainpred, pos_label='Yes', average='binary')
    accuracy = acs(Y_train,trainpred)
    results.append(recall)
    results.append(fscore)
    results.append(accuracy)
    names.append(name)
    msg = "%s: %f %f %f" % (name, recall.mean(), fscore.mean(), accuracy.mean())
    print(msg)

#ROC Curve
LR = LogisticRegression(solver='liblinear',multi_class='ovr')
KNN = KNeighborsClassifier()
CART = DecisionTreeClassifier()
RF = RandomForestClassifier(n_estimators=10)
MNB = MultinomialNB()
LDA = LinearDiscriminantAnalysis()
SVM = SVC(gamma='auto',max_iter=3000,probability=True)

LR = LR.fit(X_train, Y_train)
LRprobs = LR.predict_proba(X_validation)
LRpreds = LRprobs[:,1]
LRfpr, LRtpr, LRthreshold = metrics.roc_curve(Y_validation, LRpreds,pos_label='Yes')
LRroc_auc = metrics.auc(LRfpr, LRtpr)

KNN = KNN.fit(X_train, Y_train)
KNNprobs = KNN.predict_proba(X_validation)
KNNpreds = KNNprobs[:,1]
KNNfpr, KNNtpr, KNNthreshold = metrics.roc_curve(Y_validation, KNNpreds,pos_label='Yes')
KNNroc_auc = metrics.auc(KNNfpr, KNNtpr)

CART = CART.fit(X_train, Y_train)
CARTprobs = CART.predict_proba(X_validation)
CARTpreds = CARTprobs[:,1]
CARTfpr, CARTtpr, CARTthreshold = metrics.roc_curve(Y_validation, CARTpreds,pos_label='Yes')
CARTroc_auc = metrics.auc(CARTfpr, CARTtpr)

RF = RF.fit(X_train, Y_train)
RFprobs = RF.predict_proba(X_validation)
RFpreds = RFprobs[:,1]
RFfpr, RFtpr, RFthreshold = metrics.roc_curve(Y_validation, RFpreds,pos_label='Yes')
RFroc_auc = metrics.auc(RFfpr, RFtpr)

MNB = MNB.fit(X_train, Y_train)
MNBprobs = MNB.predict_proba(X_validation)
MNBpreds = MNBprobs[:,1]
MNBfpr, MNBtpr, MNBthreshold = metrics.roc_curve(Y_validation, MNBpreds,pos_label='Yes')
MNBroc_auc = metrics.auc(MNBfpr, MNBtpr)

LDA = LDA.fit(X_train, Y_train)
LDAprobs = LDA.predict_proba(X_validation)
LDApreds = LDAprobs[:,1]
LDAfpr, LDAtpr, LDAthreshold = metrics.roc_curve(Y_validation, LDApreds,pos_label='Yes')
LDAroc_auc = metrics.auc(LDAfpr, LDAtpr)

SVM = SVM.fit(X_train, Y_train)
SVMprobs = SVM.predict_proba(X_validation)
SVMpreds = SVMprobs[:,1]
SVMfpr, SVMtpr, SVMthreshold = metrics.roc_curve(Y_validation, SVMpreds,pos_label='Yes')
SVMroc_auc = metrics.auc(SVMfpr, SVMtpr)

#plt ROC curve
plt.title('ROC curve')
plt.plot(LRfpr, LRtpr, 'blue', label = 'LR, AUC = %0.2f' % LRroc_auc)
plt.plot(KNNfpr, KNNtpr, 'red', label = 'KNN AUC = %0.2f' % KNNroc_auc)
plt.plot(CARTfpr, CARTtpr, 'yellow', label = 'CART AUC = %0.2f' % CARTroc_auc)
plt.plot(RFfpr, RFtpr, 'green', label = 'RF AUC = %0.2f' % RFroc_auc)
plt.plot(MNBfpr, MNBtpr, 'gray', label = 'MNB AUC = %0.2f' % MNBroc_auc)
plt.plot(LDAfpr, LDAtpr, 'pink', label = 'LDA AUC = %0.2f' % LDAroc_auc)
plt.plot(SVMfpr, SVMtpr, 'purple', label = 'SVM AUC = %0.2f' % SVMroc_auc)


plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#In this case, the cost of False Negative(The person have the high risk to have heart deasease but we predict that he/she is safe) is more serious than False Positive, therefore, the algorithm which can give the smallest amount of False Negative is the best for this problem. 
#As above testing result, the Random Forest give us the highest Recall score with the highest accuracy, which means that the amount of False Negative is smallest.

#Prepare the model for the best performance
model = RandomForestClassifier(n_estimators=10,max_depth=18)
model = model.fit(X_train, Y_train)
y_pred = model.predict(X_validation)


precision, recall, fscore, train_support = score(Y_validation, y_pred, pos_label='Yes', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(round(precision, 3), round(recall, 3), round(fscore,3), round(acs(Y_validation,y_pred), 3)))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_validation, y_pred)
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# save the model to disk
filename = '/Users/Tim/Desktop/scor_test/TESTCODE/finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
# load the model from disk (testing)
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_validation, Y_validation)
print(result)
