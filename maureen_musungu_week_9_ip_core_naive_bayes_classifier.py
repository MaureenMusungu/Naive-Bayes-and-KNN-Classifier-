# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
#importing the neccessary libraries
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Reading the data

##
##
data = pd.read_csv('spambase Data.csv')

# Previewing the top of the data
data.head()

# Prevoewing the bottom of the data
data.tail()

# Checking information about the data
data.info()

# Getting the number of rows and columns
data.shape

## Tidying the Data"""

# Checking the columns
data.columns

#Checking for null values
data.isnull().sum()

# There are no null values in the data

# Checking for duplicates

data.duplicated().sum()

# Dropping the duplicate entries

data.drop_duplicates(inplace=True)
data.duplicated().sum()

# Renaming columns

data.rename(columns={'0':'word_freq_make','0.64':'word_freq_address','0.64.1':'word_freq_all', '0.1':'word_freq_3d',
                   '0.32':'word_freq_our','0.2':'word_freq_over','0.3':'word_freq_remove','0.4':'word_freq_internet',
                   '0.5':'word_freq_order','0.6':'word_freq_mail','0.7':'word_freq_receive','0.64.2':'word_freq_will', 
                   '0.8':'word_freq_people','0.9':'word_freq_report','0.10':'word_freq_address',
                   '0.32.1':'word_freq_free','0.11':'word_freq_business', '1.29':'word_freq_email',
                   '1.93':'word_freq_you','0.12':'word_freq_credit','0.96':'word_freq_your','0.13':'word_freq_font',
                   '0.14':'word_freq_000','0.15':'word_freq_money','0.16':'word_freq_hp','0.17':'word_freq_hpl',
                   '0.18':'word_freq_george','0.19':'word_freq_650','0.20':'word_freq_lab','0.21':'word_freq_labs',
                   '0.22':'word_freq_telnet','0.23':'word_freq_857','0.24':'word_freq_data','0.25':'word_freq_415',
                   '0.26':'word_freq_85','0.27':'word_freq_technology','0.28':'word_freq_1999','0.29':'word_freq_parts',
                   '0.30':'word_freq_pm','0.31':'word_freq_direct','0.32.2':'word_freq_cs','0.33':'word_freq_meeting',
                   '0.34':'word_freq_original','0.35':'word_freq_project', '0.36':'word_freq_re','0.37':'word_freq_edu',
                   '0.38':'word_freq_table','0.39':'word_freq_conference','0.40':'char_freq_%3B','0.41':'char_freq_%28',
                   '0.42':'char_freq_%5B','0.778':'char_freq_%21','0.43':'char_freq_%24','0.44':'char_freq_%23',
                   '3.756':'capital_run_length_average','61':'capital_run_length_longest','278':'capital_run_length_total', '1':'class'},inplace=True)

data.columns

## Exploratory Data Analysis"""

# Printing the statistical summaries
data.describe()

# Importing the relevant libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Bar plot to compare the status of emails as either spam or not
plt.figure(figsize=(5,5))
sns.countplot(x='class', data=data)
plt.title('Type of Email',fontsize=15)
plt.xlabel('Class',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()

#* Many of the emails were not considered to be spam. """

# Checking for multicollinearity using a correlation table
data.corr()

## Modelling"""

# Commented out IPython magic to ensure Python compatibility.
# Importing the relevant libraries

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from scipy.stats import norm

# Checking the data distribution

x = np.linspace(-5, 5)
y = norm.pdf(x)
plt.plot(x, y)
#plt.vlines(ymin=0, ymax=0.4, x=1, colors=['red'])

#* From the plot, the data looks gaussian, hence we use the gaussian Naive Bayes classifier.

### 80:20 Segmentation


# importing the required libraries

from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
import numpy as np
from sklearn.naive_bayes import GaussianNB

# preprocessing the data

X = data.drop('class',axis=1).values
y = data['class'].values

# Splitting our data into a training set and a test set
# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training our model
clf = GaussianNB()  
model = clf.fit(X_train, y_train)

import numpy as np

# Predicting our test predictors
pred = model.predict(X_test)
print(np.mean(pred == y_test))

# Predicting the test set results

y_pred = clf.predict(X_test)

#Classification metrices
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

#* The accuracy score of this model was 81%.

### 70:30 Segmentation


# Splitting our data into a training set and a test set
# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Training our model
# 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()  
model = clf.fit(X_train, y_train)

import numpy as np
# Predicting our test predictors
predicted = model.predict(X_test)
print(np.mean(predicted == y_test))

#Classification metrices
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

#* The accuracy score of this model was 82%.

### 60:40 Segmentation


# Splitting our data into a training set and a test set
# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Training our model
# 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()  
model = clf.fit(X_train, y_train)

import numpy as np
# Predicting our test predictors
predicted = model.predict(X_test)
print(np.mean(predicted == y_test))

#Classification metrices
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

#* The accuracy score of this model was 83%.

## Results and Conclusion

#Below is a summary of the performance per model:

#80:20 - Model accuracy score: 0.8147

# 70:30 - Model accuracy score: 0.8250

# 60:40 - Model accuracy score: 0.8302

# From the three segments, the model that was more accurate was the one divided into 60:40 sets with a score of 83%.
