
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:

data = pd.read_csv("C:/Users/saga54/Desktop/cs-training.csv")


# In[5]:

data.describe()


# In[6]:

data=data.drop('Unnamed: 0', axis = 1)


# In[7]:

data.describe()


# In[8]:


data.columns


# In[9]:

cleanCol = []
for i in range(len(data.columns)):
    cleanCol.append(data.columns[i].replace('-', ''))


# In[10]:

cleanCol


# In[11]:


data.columns = cleanCol


# In[12]:

data.describe()


# In[13]:

data.head(5)


# In[14]:

data.isnull().sum()


# In[15]:

data['age'].hist(bins=100)


# In[16]:

data.age.describe()


# In[17]:

for i in range(0,110):
    print i, len(data[data.age == i])


# In[18]:

"""age should be between a defined range, 0-109 makes less sense, should be between 22-91"""


# In[19]:

np.median(data.age)
np.mean(data.age)


# In[20]:

mean_age=np.mean(data.age)
ageNew=[]
for val in data.age:
    if val < 22 or val >91:
        ageNew.append(mean_age)
    else:
        ageNew.append(val)
        


# In[21]:

ageNew


# In[22]:

data.age = ageNew


# In[23]:

"""RevolvingUtilizationOfUnsecuredLines = Total balance on credit cards and personal lines of credit 
except real estate and no installment debt like car loans divided by the sum of credit limits"""


# In[24]:

data.RevolvingUtilizationOfUnsecuredLines.describe()


# In[25]:

len(data[data.RevolvingUtilizationOfUnsecuredLines >1])


# In[26]:


"""the value here should be between 0-1 [implies 0 to 100%], but few values are more than 1 [implying more than 100%], so all those values must be a data entry error and should be changed to the value/100"""


# In[27]:

for val in data.RevolvingUtilizationOfUnsecuredLines:
    if val >1:
        data['RUUL_indicator']=1
    else:
        data['RUUL_indicator']=0


# In[28]:

RUULNew=[]

for val in data.RevolvingUtilizationOfUnsecuredLines:
    if val <=10:
        RUULNew.append(val/10)
    elif val >10 and val <=100:
        RUULNew.append(val/100)
    elif val >100 and val <=1000:
        RUULNew.append(val/1000)
    elif val >1000 and val <=10000:
        RUULNew.append(val/10000)
    elif val >10000 and val <=100000:   
        RUULNew.append(val/100000)
    else:
        RUULNew.append(val)
        


# In[29]:

data.RevolvingUtilizationOfUnsecuredLines = RUULNew


# In[30]:

"""NumberOfTime3059DaysPastDueNotWorse"""


# In[31]:

data.NumberOfTime3059DaysPastDueNotWorse.describe()


# In[32]:

for i in range(0,100):
    print i, len(data[data.NumberOfTime3059DaysPastDueNotWorse == i])


# In[33]:

"""looks 96 and 98 are outliers"""


# In[34]:

New = []
meanNOTT = data.NumberOfTime3059DaysPastDueNotWorse.mean()
for val in data.NumberOfTime3059DaysPastDueNotWorse:
    if ((val == 98) | (val == 96)):
        New.append(meanNOTT)
    else:
        New.append(val)

data.NumberOfTime3059DaysPastDueNotWorse = New


# In[36]:

"""DebtRatio""" 


# In[37]:

data.DebtRatio.describe()


# In[38]:

len(data[data.DebtRatio > 1])


# In[39]:

len(data[data.DebtRatio >0])


# In[40]:

New = []
medianNOTT = data.DebtRatio.median()
for val in data.NumberRealEstateLoansOrLines:
    if val>1:
        New.append(medianNOTT)
    else:
        New.append(val)

data.DebtRatio = New


# In[ ]:

"""NumberOfOpenCreditLinesAndLoans"""


# In[41]:

data.NumberOfOpenCreditLinesAndLoans.describe()


# In[42]:

data['NumberOfOpenCreditLinesAndLoans'].hist(bins=100)


# In[ ]:

"""NumberOfTimes90DaysLate"""


# In[43]:

data.NumberOfTimes90DaysLate.describe()


# In[44]:

for i in range(0,100):
    print i, len(data[data.NumberOfTimes90DaysLate == i])


# In[45]:

New = []
meanNOTT = data.NumberOfTimes90DaysLate.mean()
for val in data.NumberOfTimes90DaysLate:
    if ((val == 98) | (val == 96)):
        New.append(meanNOTT)
    else:
        New.append(val)

data.NumberOfTimes90DaysLate = New


# In[46]:

"""NumberRealEstateLoansOrLines"""


# In[47]:

data.NumberRealEstateLoansOrLines.describe()


# In[48]:

for i in range(0,55):
    print i, len(data[data.NumberRealEstateLoansOrLines == i])


# In[49]:

New = []
meanNOTT = data.NumberRealEstateLoansOrLines.mean()
for val in data.NumberRealEstateLoansOrLines:
    if val>50:
        New.append(meanNOTT)
    else:
        New.append(val)

data.NumberRealEstateLoansOrLines = New


# In[50]:

data.NumberRealEstateLoansOrLines.describe()


# In[51]:

"""NumberOfTime6089DaysPastDueNotWorse"""


# In[52]:

data.NumberOfTime6089DaysPastDueNotWorse.describe()


# In[53]:

New = []
meanNOTT = data.NumberOfTime6089DaysPastDueNotWorse.mean()
for val in data.NumberOfTime6089DaysPastDueNotWorse:
    if ((val == 98) | (val == 96)):
        New.append(meanNOTT)
    else:
        New.append(val)

data.NumberOfTime6089DaysPastDueNotWorse = New


# In[54]:

data.NumberOfTime6089DaysPastDueNotWorse.describe()


# In[55]:

"""NumberOfDependents"""


# In[56]:

data.NumberOfDependents.describe()


# In[57]:

for i in range(0,25):
    print i, len(data[data.NumberOfDependents == i])


# In[58]:

"""having more than 10 dependents looks weird"""


# In[59]:

New = []
meanNOTT = data.NumberOfDependents.mean()
for val in data.NumberOfDependents:
    if val>10:
        New.append(meanNOTT)
    else:
        New.append(val)

data.NumberOfDependents = New


# In[60]:

data.NumberOfDependents.isnull().sum()


# In[61]:

data['NumberOfDependents'] = data['NumberOfDependents'].fillna(0)


# In[62]:

data.NumberOfDependents.describe()


# In[63]:

"""MonthlyIncome"""


# In[64]:

data.MonthlyIncome.describe()


# In[65]:

train = data[data.MonthlyIncome.isnull() == False]
test = data[data.MonthlyIncome.isnull() == True]


# In[66]:

train.shape, test.shape


# In[67]:

X_train = train.drop(['MonthlyIncome', 'SeriousDlqin2yrs'], axis=1)
y_train = train.MonthlyIncome
X_test = test.drop(['MonthlyIncome', 'SeriousDlqin2yrs'], axis=1)


# In[68]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[69]:

lmMod = LinearRegression(fit_intercept=True, normalize=True).fit(X_train, y_train)


# In[70]:

lmMod.coef_


# In[71]:

pred = lmMod.predict(X_test)


# In[72]:

predNoZero = []
for val in pred:
    if val >= 0:
        predNoZero.append(val)
    else:
        predNoZero.append(0.)


# In[73]:

testFull = data[data.MonthlyIncome.isnull() == True]


# In[74]:

testFull['MonthlyIncome'] = predNoZero


# In[76]:

monNew = []
for index in data.index:
    if data.MonthlyIncome[index].is_integer() == True:
        monNew.append(data.MonthlyIncome[index])
    else:
        monNew.append(testFull.MonthlyIncome[index])


# In[77]:

testFull.MonthlyIncome.isnull().sum()


# In[78]:


data.MonthlyIncome = monNew


# In[79]:

"""FEATURE ENGINEERING"""


# In[80]:

data.MonthlyIncome.describe()


# In[81]:

"""No Income Variable Indicator"""


# In[82]:

for val in data.MonthlyIncome:
    if val <=0:
        data['NoIncome_MI_indicator']=1
    else:
        data['NoIncome_MI_indicator']=0


# In[83]:

"""Zero Debt Ratio Indicator"""


# In[84]:

for val in data.DebtRatio:
    if val <=0:
        data['No_DebtRatio_indicator']=1
    else:
        data['No_DebtRatio_indicator']=0


# In[85]:

"""Monthly Income is Zero, But Debt Ratio is non-zero = 1""" 


# In[86]:

data['MIz_DRnz']=map(lambda x,y: 1 if (x==0 and y>0) else 0, data['MonthlyIncome'], data['DebtRatio'])


# In[87]:

"""Monthly Income is Zero, But Debt Ratio is zero = 1""" 


# In[88]:

data['MIz_DRz']=map(lambda x,y: 1 if (x==0 and y==0) else 0, data['MonthlyIncome'], data['DebtRatio'])


# In[89]:

"""Monthly Income is Non-Zero, But Debt Ratio is zero = 1""" 


# In[90]:

data['MInz_DRz']=map(lambda x,y: 1 if (x>0 and y==0) else 0, data['MonthlyIncome'], data['DebtRatio'])


# In[91]:

"""Zero Revolving Utilization when Revolving Utilization Of Unsecured Lines == 0"""


# In[92]:

for val in data.RevolvingUtilizationOfUnsecuredLines:
    if val <=0:
        data['ZeroRevolvingUtilization']=1
    else:
        data['ZeroRevolvingUtilization']=0


# In[93]:

"""debtRatio * Monthly Income = DR_MI"""


# In[94]:

#All 0 MI to 1 so that DR dont become 0 
for val in data.MonthlyIncome:
    if val ==0:
        MIZ=1
    else:
        MIZ=val


# In[95]:

data['DR_MI']=MIZ*data['DebtRatio']


# In[96]:

data.DR_MI.describe()


# In[97]:

from math import *


# In[98]:

data['Log_DR_MI']=np.log(data.DR_MI)


# In[99]:

"""Log of DebtRatio*MonthlyIncome"""


# In[101]:

data_new2=data


# In[102]:

data['Log_DR_MI']=data.Log_DR_MI.replace([np.inf, -np.inf], 0)


# In[103]:

"""  RevolvingLines = NumberOfOpenCreditLinesAndLoans - NumberRealEstateLoansOrLines
"""


# In[104]:

data['RevolvingLines']=data['NumberOfOpenCreditLinesAndLoans']-data['NumberRealEstateLoansOrLines']


# In[105]:

"""HasRealEstateLoans = NumberRealEstateLoansOrLines > 0)"""


# In[106]:

HRSL=[]
for val in data.NumberRealEstateLoansOrLines:
    if val >0:
        HRSL.append(1)
    else:
        HRSL.append(0)
        
data['HasRealEstateLoans']=HRSL


# In[107]:

"""HasMultipleRealEstateLoans = NumberRealEstateLoansOrLines > 2"""


# In[108]:

MHRSL=[]
for val in data.NumberRealEstateLoansOrLines:
    if val >2:
        MHRSL.append(1)
    else:
        MHRSL.append(0)
        
data['HasMultipleRealEstateLoans ']=MHRSL


# In[109]:

"""DisposableIncome = (1 - DebtRatio) * MonthlyIncome"""


# In[110]:

data['DisposableIncome']=(1-data['DebtRatio'])*data['MonthlyIncome']


# In[111]:

"""RevolvingToRealEstate  = RevolvingLines / (1 + NumberRealEstateLoansOrLines)"""


# In[112]:

data['RevolvingToRealEstate']=data['RevolvingLines'] / (1+data['NumberRealEstateLoansOrLines'])


# In[113]:

"""FullUtilization = RevolvingUtilizationOfUnsecuredLines == 1)
  ExcessUtilization = RevolvingUtilizationOfUnsecuredLines > 1)"""


# In[114]:

FU=[]
for val in data.RevolvingUtilizationOfUnsecuredLines:
    if val ==1:
        FU.append(1)
    else:
        FU.append(0)
        
data['FullUtilization']=FU


# In[115]:

EU=[]
for val in data.RevolvingUtilizationOfUnsecuredLines:
    if val >1:
        EU.append(1)
    else:
        EU.append(0)
        
data['ExcessUtilization']=EU


# In[116]:

"""
  RevolvingLinesPerPerson = RevolvingLines / (1 + NumberOfDependents)
  RealEstateLoansPerPerson = NumberRealEstateLoansOrLines / (1 + NumberOfDependents)
  IncomePerDependent = 1+NumberOfDependents/MonthlyIncome+1
  """


# In[117]:

data['RevolvingLinesPerPerson'] = data['RevolvingLines'] / (1+data['NumberOfDependents'])
data['RealEstateLoanPerPerson'] = data['NumberRealEstateLoansOrLines'] / (1+data['NumberOfDependents'])


# In[118]:

data['IncomePerDependent']=(1+data['NumberOfDependents']) / (1+data['MonthlyIncome'])


# In[119]:

"""NumberOfTimePastDue"""


# In[120]:

data['NumberOfTimePastDue']=data['NumberOfTime3059DaysPastDueNotWorse']+data['NumberOfTime6089DaysPastDueNotWorse']+data['NumberOfTimes90DaysLate']


# In[121]:

"""DelinquenciesPerLine  = NumberOfTimesPastDue / NumberOfOpenCreditLinesAndLoans"""


# In[122]:

data['DelinquenciesPerLine']=data['NumberOfTimePastDue'] /data['NumberOfOpenCreditLinesAndLoans']


# In[123]:

data_new3=data


# In[124]:







data['DelinquenciesPerLine']=data.DelinquenciesPerLine.replace([np.inf, -np.inf], np.NaN)
data.DelinquenciesPerLine[np.isnan(data.DelinquenciesPerLine)] = 0
data.DelinquenciesPerLine[np.isinf(data.DelinquenciesPerLine)] = 0
data['DelinquenciesPerLine']=data['DelinquenciesPerLine'].fillna(0)


# In[125]:

data['DelinquenciesPerLine']=data.DelinquenciesPerLine.replace([np.inf, -np.inf], 0)


# In[126]:

data.DelinquenciesPerLine.isnull().sum()


# In[127]:

"""DelinquenciesPerRevolvingLine  = NumberOfTimesPastDue / RevolvingLines"""


# In[128]:

data['DelinquenciesPerRevolvingLine'] = data['NumberOfTimePastDue'] / data['RevolvingLines']


# In[131]:

data['DelinquenciesPerRevolvingLine']=data.DelinquenciesPerRevolvingLine.replace([np.inf, -np.inf], np.NaN)
data.DelinquenciesPerRevolvingLine[np.isnan(data.DelinquenciesPerRevolvingLine)] = 0
data.DelinquenciesPerRevolvingLine[np.isinf(data.DelinquenciesPerRevolvingLine)] = 0
data['DelinquenciesPerRevolvingLine']=data['DelinquenciesPerRevolvingLine'].fillna(0)


# In[132]:

from sklearn.cross_validation import train_test_split


# In[133]:

data_new4=data


# In[134]:

X = data.drop('SeriousDlqin2yrs', axis=1)
y = data.SeriousDlqin2yrs


# In[135]:

#np.savetxt("C:/Users/saga54/Desktop/foo.csv", data, delimiter=",")


# In[137]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[144]:

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier


# In[145]:

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)


# In[146]:

y_pred = clf.fit(X_train, y_train).predict(X_test)


# In[147]:

from sklearn.metrics import confusion_matrix


# In[148]:

confusion_matrix(y_test, y_pred)


# In[149]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[150]:

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[151]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


# In[152]:

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
metrics.auc(fpr, tpr)


# In[153]:

""""colsample_bytree": 0.41,
      "gamma": 0.643,
      "max_depth": 5,
      "max_delta_step": 1.78,
      "min_child_weight": 10.0,
      "objective": "binary:logistic",
      "subsample": 0.801,
      "learning_rate": 0.027,
      "silent": false,
      "nthread": 7,
      "n_estimators": 295,
      "seed": 2"""


# In[155]:

data.to_csv("C:/Users/saga54/Desktop/gmc.csv")


# In[ ]:

"""RANDOM FOREST CLASSIFIER"""


# In[200]:

from sklearn.ensemble import RandomForestClassifier
clfRF = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)


# In[201]:


y_pred = clfRF.fit(X_train, y_train).predict(X_test)


# In[202]:

confusion_matrix(y_test, y_pred)


# In[203]:


accuracy_score(y_test, y_pred)


# In[204]:

clfRF.feature_importances_


# In[205]:

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[206]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


# In[207]:

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
metrics.auc(fpr, tpr)


# In[ ]:

"""ADA BOOSTING CLASSIFIER"""


# In[208]:

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

clfADA = AdaBoostClassifier(n_estimators=100)



# In[209]:

y_pred = clfADA.fit(X_train, y_train).predict(X_test)


# In[210]:

confusion_matrix(y_test, y_pred)


# In[211]:

accuracy_score(y_test, y_pred)


# In[212]:

clfADA.feature_importances_


# In[214]:

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[215]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


# In[216]:

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
metrics.auc(fpr, tpr)


# In[ ]:

"""ENSEMBLE 1"""


# In[218]:

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


clf1 = GradientBoostingClassifier(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')



# In[219]:

for clf, label in zip([clf1, clf2, clf3, eclf], ['Gradient Boosting Classifier', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# In[220]:

"""Ensemble 2"""


# In[221]:

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=1)
clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
clf3 = AdaBoostClassifier(n_estimators=100)
X = X_train
y = y_train
eclf1 = VotingClassifier(estimators=[('gb', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X, y)

y_pred1=eclf1.predict(X_test)


eclf2 = VotingClassifier(estimators=[('gb', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft')
eclf2 = eclf2.fit(X, y)

y_pred2=eclf2.predict(X_test)

eclf3 = VotingClassifier(estimators=[('gb', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[2,1,1])
eclf3 = eclf3.fit(X, y)
y_pred3=eclf3.predict(X_test)


# In[222]:

confusion_matrix(y_test, y_pred1)

accuracy_score(y_test, y_pred1)


# In[223]:

confusion_matrix(y_test, y_pred2)

accuracy_score(y_test, y_pred2)


# In[224]:

confusion_matrix(y_test, y_pred3)

accuracy_score(y_test, y_pred3)


# In[225]:

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred1, target_names=target_names))


# In[226]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred1)


# In[227]:

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred1)
metrics.auc(fpr, tpr)


# In[228]:

"""Ensemble 3"""


# In[229]:

from sklearn.model_selection import GridSearchCV
clf1 = GradientBoostingClassifier(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = AdaBoostClassifier()
eclf = VotingClassifier(estimators=[('gb', clf1), ('rf', clf2), ('ab', clf3)], voting='soft')

params = {'rf__n_estimators': [20, 200],}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(X_train, y_train)


# In[230]:

y_pred=grid.predict(X_test)


# In[231]:

confusion_matrix(y_test, y_pred)


# In[232]:

accuracy_score(y_test, y_pred)


# In[233]:

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[234]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


# In[235]:

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
metrics.auc(fpr, tpr)


# In[ ]:



