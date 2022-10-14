#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report  
from sklearn import tree


# In[ ]:


#Use decision trees to prepare a model on fraud data 
#treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
#Data Description :
#Undergrad : person is under graduated or not
#Marital.Status : marital status of a person
#Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
#Work Experience : Work experience of an individual person
#Urban : Whether that person belongs to urban area or not


# In[2]:


df_C = pd.read_csv("C:\\Users\\HP\\Documents\\Excel R data\\Fraud_check.csv")
df_C.head()


# In[3]:


df = df_C.rename({'Undergrad':'UG','Marital.Status':'MS','Taxable.Income':'TI' ,'City.Population':'CP', 'Work.Experience':'WE' }, axis=1)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[13]:


df.describe()


# In[7]:


sns.pairplot(df)


# In[8]:


# converting sales value in catagorical data
TI_F = []
for TI in df["TI"]:
    if TI <=30000:
        TI_F.append("Risky")
    else:
        TI_F.append("Good")
df["TI_F"]= TI_F
df1 = df
df1.head()


# In[9]:


print(df1.UG.value_counts())
print(df1.MS.value_counts())
print(df1.Urban.value_counts())
print(df1.TI_F.value_counts())


# In[10]:


sns.countplot(df1.UG)


# In[11]:


sns.countplot(df1.MS)


# In[12]:


sns.countplot(df1.Urban)


# In[13]:


sns.countplot(df1.TI_F)


# In[14]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df1["UG"]= LE.fit_transform(df["UG"])
df1["Urban"] = LE.fit_transform(df["Urban"])
df1["MS"] = LE.fit_transform(df["MS"])
print(df1.head())
print(df1.shape)


# In[15]:


# Correlation analysis for data
corr = df1.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 6))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='coolwarm',linewidths = 1,linecolor="y", annot=True, fmt=".4f")
plt.show()


# # Selection of X and Y variable

# In[16]:


X = df1.drop(['TI', 'TI_F'], axis = 1)
print(list(X))


# In[17]:


Y = df1["TI_F"]
Y


# In[18]:


pd.crosstab(Y,Y)


# # Entropy method

# In[19]:


#Entropy 

# SPLITING DATA IN TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 40)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(pd.crosstab(Y_train,Y_train))
print(pd.crosstab(Y_test,Y_test))

# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
ET.fit(X_train, Y_train) 

# Predicted Y, Confusion Matrix and Accuracy score for TRAINNING data
#Y1_pred = ET.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,Y1_pred)
#print(cm)
#ac = accuracy_score(Y_train,Y1_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
Y_pred = ET.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
ac = accuracy_score(Y_test,Y_pred)
print(ac)


# In[20]:


pd.crosstab(Y_test,Y_test)


# # Classification Report

# In[21]:


from sklearn.metrics import classification_report  
print(classification_report(Y_pred, Y_test)) 


# In[22]:


# Using KFold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state = 7, shuffle = True)
from sklearn.tree import DecisionTreeClassifier
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
from sklearn.model_selection import cross_val_score
results = cross_val_score(ET, X, Y, cv=kfold)
print(results)
print(np.mean(abs(results)))
# Predicted Y, Confusion Matrix and Accuracy score for TEST data
from sklearn.model_selection import KFold, cross_val_predict
Y1_pred = cross_val_predict(ET, X, Y, cv=kfold)
print(pd.crosstab(Y,Y))
print(pd.crosstab(Y1_pred,Y1_pred))
print(pd.crosstab(Y,Y1_pred))


# In[23]:


from sklearn.metrics import classification_report  
print(classification_report(Y1_pred, Y))


# In[24]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50) # lr = 0.1, est = 100

gbc.fit(X_train,Y_train)

YG1_pred = gbc.predict(X_test)
pd.crosstab(YG1_pred, YG1_pred)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YG1_pred = gbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YG1_pred)
print(cm)
ac = accuracy_score(Y_test,YG1_pred)
print(ac)


# In[25]:


# Ada Boost Classifier
from sklearn.tree import DecisionTreeClassifier
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
from sklearn.ensemble import AdaBoostClassifier
adbc = AdaBoostClassifier(base_estimator=ET,n_estimators=50) 
adbc.fit(X_train,Y_train)
YA_pred = adbc.predict(X_test)
pd.crosstab(YA_pred, YA_pred)

#YAT_pred = adbc.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,YAT_pred)
#print(cm)
#ac = accuracy_score(Y_train,YAT_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YA_pred = adbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YA_pred)
print(cm)
ac = accuracy_score(Y_test,YA_pred)
print(ac)


# In[26]:


from sklearn.metrics import classification_report  
print(classification_report(YA_pred, Y_test))


# # Tree Graph

# In[27]:


from sklearn import tree
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
ET.fit(X_train,Y_train)
tree.plot_tree(ET) 


# In[28]:


import matplotlib.pyplot as plt
fn=['UG', 'MS', 'CP', 'WE', 'Urban']
cn=['Good', 'Risky']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,6), dpi=600)
tree.plot_tree(ET,
               feature_names = fn, 
               class_names=cn,
               filled = True);   


# # Gini

# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(pd.crosstab(Y_train,Y_train))
print(pd.crosstab(Y_test,Y_test))

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
DT.fit(X_train,Y_train)

# Predicted Y, Confusion Matrix and Accuracy score for TRAINNING data
#Y1_pred = DT.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,Y1_pred)
#print(cm)
#ac = accuracy_score(Y_train,Y1_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
Y_pred = DT.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
ac = accuracy_score(Y_test,Y_pred)
print(ac)


# In[30]:


from sklearn.metrics import classification_report  
print(classification_report(Y_pred, Y_test)) 


# In[31]:


# Using KFold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state = 7, shuffle = True)
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
from sklearn.model_selection import cross_val_score
results = cross_val_score(DT, X, Y, cv=kfold)
print(results)
print(np.mean(abs(results)))
print(pd.crosstab(Y,Y))

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
from sklearn.model_selection import KFold, cross_val_predict
Y1_pred = cross_val_predict(DT, X, Y, cv=kfold)
print(pd.crosstab(Y1_pred,Y1_pred))
print(pd.crosstab(Y,Y1_pred))


# In[32]:


from sklearn.metrics import classification_report  
print(classification_report(Y1_pred, Y)) 


# In[33]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50) # lr = 0.1, est = 100

gbc.fit(X_train,Y_train)

YG1_pred = gbc.predict(X_test)
pd.crosstab(YG1_pred, YG1_pred)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YG1_pred = gbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YG1_pred)
print(cm)
ac = accuracy_score(Y_test,YG1_pred)
print(ac)


# In[34]:


from sklearn.metrics import classification_report  
print(classification_report(YG1_pred, Y_test))


# In[36]:


# Ada Boost Classifier
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
from sklearn.ensemble import AdaBoostClassifier
adbc = AdaBoostClassifier(base_estimator=DT,n_estimators=50) 
adbc.fit(X_train,Y_train)
YA_pred = adbc.predict(X_test)
pd.crosstab(YA_pred, YA_pred)

#YAT_pred = adbc.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,YAT_pred)
#print(cm)
#ac = accuracy_score(Y_train,YAT_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YA_pred = adbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YA_pred)
print(cm)
ac = accuracy_score(Y_test,YA_pred)
print(ac)


# In[37]:


from sklearn.metrics import classification_report  
print(classification_report(YA_pred, Y_test))


# In[39]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
DT.fit(X_train,Y_train)
tree.plot_tree(DT)


# In[40]:


import matplotlib.pyplot as plt
fn=['UG', 'MS', 'CP', 'WE', 'Urban']
cn=[ 'Good','Risky']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,6), dpi=600)
tree.plot_tree(DT,
               feature_names = fn, 
               class_names=cn,
               filled = True);        

