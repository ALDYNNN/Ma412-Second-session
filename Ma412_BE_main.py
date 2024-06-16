# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:49:04 2024
BEMa412
@author: MAGDELEINE Dylan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
#%%


# importation of excel data : Train.csv
Data_train = pd.read_csv('C:/Users/MAGDELEINE Dylan/Desktop/rattrapages/train.csv')
Data_train = Data_train.drop(Data_train.columns[[0, 1,]], axis=1) #remove ID and unamed:0


# split of all the column of the data (train)
Gender = Data_train['Gender']
Customer_Type = Data_train['Customer Type']
age = Data_train['Age']
Type_of_Travel = Data_train['Type of Travel']
Class = Data_train['Class']
Flight_Distance = Data_train['Flight Distance']
Inflight_wifi_service = Data_train['Inflight wifi service']
Departure_Arrival_time_convenient = Data_train['Departure/Arrival time convenient']
Ease_of_Online_booking = Data_train['Ease of Online booking']
Gate_location = Data_train['Gate location']
Food_and_drink = Data_train['Food and drink']
Online_boarding = Data_train['Online boarding']
Seat_comfort = Data_train['Seat comfort']
Inflight_entertainment = Data_train['Inflight entertainment']
On_board_service = Data_train['On-board service']
Leg_room_service = Data_train['Leg room service']
Baggage_handling = Data_train['Baggage handling']
Checkin_service = Data_train['Checkin service']
Inflight_service = Data_train['Inflight service']
Cleanliness = Data_train['Cleanliness']
Departure_Delay_in_Minutes = Data_train['Departure Delay in Minutes']
Arrival_Delay_in_Minutes = Data_train['Arrival Delay in Minutes']
satisfaction = Data_train['satisfaction']



# importation of excel data : Train.csv
Data_test = pd.read_csv('C:/Users/MAGDELEINE Dylan/Desktop/rattrapages/test.csv')
Data_test = Data_test.drop(Data_test.columns[[0, 1,]], axis=1) #remove ID and unamed:0



#%% 

#                   I. Data Analysis and Understanding

#%%

# Histogram of age
plt.figure(figsize=(10, 6))
plt.hist(age, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram of flight distance
plt.figure(figsize=(10, 6))
plt.hist(Flight_Distance, bins=20, color='green', edgecolor='black')
plt.title('Distribution of Flight Distance')
plt.xlabel('Flight Distance')
plt.ylabel('Frequency')
plt.show()



# Bar chart for satisfaction by type of travel
pd.crosstab(Data_train['Type of Travel'], Data_train['satisfaction']).plot(kind='bar')
plt.title('Satisfaction by Type of Travel')
plt.xlabel('Type of Travel')
plt.ylabel('Count')
plt.legend(title='Satisfaction')
plt.show()

# Bar chart for satisfaction by type of class
pd.crosstab(Data_train['Class'], Data_train['satisfaction']).plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Satisfaction by Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend(title='Satisfaction')
plt.show()

# Bar chart for type of travel by type of class
pd.crosstab(Data_train['Class'], Data_train['Type of Travel']).plot(kind='bar', color=['blue', 'green', 'blue'])
plt.title('Type of Travel by Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend(title='Type of Travel')
plt.show()


# Bar chart for satisfaction by age
pd.crosstab(Data_train['Age'], Data_train['satisfaction']).plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Satisfaction by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Satisfaction')
plt.show()


# Bar chart for age by type of class
pd.crosstab( Data_train['Age'],Data_train['Class']).plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('age')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend(title='age by type of class')
plt.show()


#%% 

#                       Principal Component Analysis (PCA)


#%%

# We are treating the data
features = Data_train.select_dtypes(include=[np.number])  # we choose only the numerical value
features = features.fillna(features.mean())  # Remplace the missing value by the mean

# Normalise the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


pca = PCA().fit(features_scaled)  # we compute the pca

# Explained variance
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(explained_variance_ratio >= 0.90) + 1  # 90% of Explained variance

print(f"Number of component for 90% of explained varience : {num_components}")



pca = PCA(n_components=num_components)  # Compute the pca for the number of component found below
principalComponents = pca.fit_transform(features_scaled) 

# Creation of a DataFrame for the result of the PCA (12 columns)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3','PC4', 'PC5', 'PC6','PC7', 'PC8', 'PC9','PC10', 'PC11', 'PC12'])

print('explained varience for each component :', pca.explained_variance_ratio_)


# Extraction of eigenvectors
loadings = pca.components_.T  

# Creation of a DataFrame for eigenvectors
loading_matrix = pd.DataFrame(data=loadings, index=features.columns, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12'])


#%% Visualisation of eigenvectors

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC1'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC1')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC2'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC2')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC3'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC3')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC4'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC4')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC5'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC5')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC6'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC6')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC7'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC7')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC8'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC8')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC9'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC9')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC10'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC10')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC11'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC11')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=loading_matrix.index, height=loading_matrix['PC12'])
plt.xlabel('Variables')
plt.ylabel('eigenvectors')
plt.title('Contribution of components for PC12')
plt.xticks(rotation=45)  
plt.show()

#%%

#               LOGISTIC REGRESSION


#%%


#Data train transformation string into numbers
Data_train['Gender'] = Data_train['Gender'].replace({'Male': 1, 'Female': 0})
Data_train['satisfaction'] = Data_train['satisfaction'].replace({'satisfied': 1, 'neutral or dissatisfied': 0})
Data_train['Type of Travel'] = Data_train['Type of Travel'].replace({'Business travel': 1, 'Personal Travel': 0})
Data_train['Class'] = Data_train['Class'].replace({'Business': 2, 'Eco': 1, 'Eco Plus':0})
Data_train['Customer Type'] = Data_train['Customer Type'].replace({'Loyal Customer': 1, 'disloyal Customer': 0})

#Data test transformation string into numbers
Data_test['Gender'] = Data_test['Gender'].replace({'Male': 1, 'Female': 0})
Data_test['satisfaction'] = Data_test['satisfaction'].replace({'satisfied': 1, 'neutral or dissatisfied': 0})
Data_test['Type of Travel'] = Data_test['Type of Travel'].replace({'Business travel': 1, 'Personal Travel': 0})
Data_test['Class'] = Data_test['Class'].replace({'Business': 2, 'Eco': 1, 'Eco Plus':0})
Data_test['Customer Type'] = Data_test['Customer Type'].replace({'Loyal Customer': 1, 'disloyal Customer': 0})


#Set X and y variables
X_train = Data_train.drop('satisfaction',axis=1)
y_train = Data_train['satisfaction']

X_test = Data_test.drop('satisfaction',axis=1)
y_test = Data_test['satisfaction']

# set the lacking variables 
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Normaliser les données
scaler = StandardScaler()
scld_X_train = scaler.fit_transform(X_train_imputed)
scld_X_test = scaler.transform(X_test_imputed)



#%%

Logic_regression = LogisticRegression()
Logic_regression.fit(scld_X_train, y_train)

y_pred = Logic_regression.predict(scld_X_test)
print(f'Model: Logistic regression')
print(f'Classification Report: {classification_report(y_test, y_pred)}')
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.grid(False)
plt.show()


#%%

#               Suport vector machine


#%%

SVM = SVC()
SVM.fit(scld_X_train, y_train)

y_pred = SVM.predict(scld_X_test)
print(f'Model: SVM')
print(f'Classification Report: {classification_report(y_test, y_pred)}')
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.grid(False)
plt.show()


#%%

#               Random Forest


#%%

random_forest = RandomForestClassifier()
random_forest.fit(scld_X_train, y_train)

y_pred = random_forest.predict(scld_X_test)
print(f'Model: Random Forest')
print(f'Classification Report: {classification_report(y_test, y_pred)}')
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.grid(False)
plt.show()


#%%

#               K neighbor


#%%

K_neighbor = KNeighborsClassifier(n_neighbors=3)
K_neighbor.fit(scld_X_train, y_train)

y_pred = K_neighbor.predict(scld_X_test)
print(f'Model: K_neighbor')
print(f'Classification Report: {classification_report(y_test, y_pred)}')
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot() 
plt.grid(False)
plt.show()
    
#%%

#               Selection of the best model and final analysis

#%%



coefficients = pd.DataFrame(index=X_train.columns,data=Logic_regression.coef_ .reshape(-1,1) ,columns=['Coefficient'])
plt.figure(figsize=(14,8),dpi=200)
sns.barplot(data=coefficients.sort_values('Coefficient'),x=coefficients.sort_values('Coefficient').index,y='Coefficient')
plt.title('Variable Coefficients Based on The Logistic Regression  Model')
plt.xticks(rotation=90);    


var_importance = pd.DataFrame(index=X_train.columns,data=random_forest.feature_importances_ .reshape(-1,1) ,columns=['Importance'])
plt.figure(figsize=(12,8),dpi=200)
sns.barplot(data=var_importance.sort_values('Importance'),x=var_importance.sort_values('Importance').index,y='Importance')
plt.title('Variable Importance Based on The Random Forest Model')
plt.xticks(rotation=90);






