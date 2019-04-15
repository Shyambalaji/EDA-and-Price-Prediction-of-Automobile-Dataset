import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression



print("Reading the dataset.........")
automobile_data = pd.read_csv("Automobile_data.csv")

print("Header of the dataset.......")
print(automobile_data.head())

print("Type of data type of each attributes.......")
print(automobile_data.dtypes)

print("Mean,Standard devation,min value,max value of each attribute of the dataset......")
print(automobile_data.describe())


print("Finding the unfilled values by replacing it with NULL.......")
print(automobile_data.isnull().sum())
data_temp=automobile_data.replace('?',np.NAN)
print(data_temp.isnull().sum())

print("    Cleaning the dataset     ")
print("      Cleaning the Normalized loss    ")
print(automobile_data[automobile_data['normalized-losses'] != '?'])
normalized_loss = automobile_data['normalized-losses'].loc[automobile_data['normalized-losses'] != '?'] 
normalized_loss_median = normalized_loss.astype(int).median()
#Relplacing the missing values to median
automobile_data['normalized-losses'] = automobile_data['normalized-losses'].replace('?',normalized_loss_median).astype(int)
print(automobile_data['normalized-losses'])

print("Cleaning the bore")
print(automobile_data[automobile_data['bore']!='?'])
bore = automobile_data['bore'].loc[automobile_data['bore']!='?']
bore_median = bore.astype(float).median()
automobile_data['bore']=automobile_data['bore'].replace('?',bore_median)
print(automobile_data['bore'])

print("Cleaning the stroke")
print(automobile_data[automobile_data['stroke']!='?'])
stroke=automobile_data['stroke'].loc[automobile_data['stroke']!='?']
stroke_median = stroke.astype(float).median()  
automobile_data['stroke']=automobile_data['stroke'].replace('?',stroke_median)
print(automobile_data['stroke'])

print("Cleaning the horsepower")
print(automobile_data[automobile_data['horsepower']!='?'])
horsepower = automobile_data['horsepower'].loc[automobile_data['horsepower']!='?']
horsepower_median=horsepower.astype(int).median()
automobile_data['horsepower']=automobile_data['horsepower'].replace('?',horsepower_median)
print(automobile_data['horsepower'])

print("Cleaning the peak_rpm")
print(automobile_data[automobile_data['peak-rpm']!='?'])
peak_rpm = automobile_data['peak-rpm'].loc[automobile_data['peak-rpm']!='?']
peak_rpm_median = peak_rpm.astype(int).median()
automobile_data['peak-rpm']=automobile_data['peak-rpm'].replace('?',peak_rpm_median)
print(automobile_data['peak-rpm'])

print("Cleaning the price")
print(automobile_data[automobile_data['price']!='?'])
price = automobile_data['price'].loc[automobile_data['price']!='?']
price_median = price.astype(int).median()
automobile_data['price']=automobile_data['price'].replace('?',price_median)
print(automobile_data['price'])
temp=automobile_data['num-of-doors'].map({'two':2,'four':4,'?':4})
automobile_data['num-of-doors']=temp

print("overall corelation between the attributes")
plt.figure(figsize=(15,13))
sns.heatmap(automobile_data.corr(),annot=True,cmap='summer')

automobile_data[['symboling','normalized-losses']].hist(figsize=(10,8),bins=6,color='b',linewidth='3',edgecolor='k')
plt.tight_layout()
plt.show()


#automobile_data[['wheel-base','peak-rpm']].hist(figsize=(5,8),bins=6,color='Y',linewidth='2',edgecolor='k')
#plt.tight_layout()
#plt.show()



automobile_data[['engine-size','compression-ratio']].hist(figsize=(5,8),bins=6,color='Y')
plt.tight_layout()
plt.show()


plt.figure(1)
plt.subplot(221)
automobile_data['engine-type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red')
plt.title("Number of Engine Type frequency diagram")
plt.ylabel('Number of Engine Type')
plt.xlabel('engine-type');


plt.subplot(222)
automobile_data['num-of-doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='green')
plt.title("Number of Door frequency diagram")
plt.ylabel('Number of Doors')
plt.xlabel('num-of-doors');

plt.subplot(223)
automobile_data['fuel-type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='purple')
plt.title("Number of Fuel Type frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('fuel-type');

plt.subplot(224)
automobile_data['body-style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='orange')
plt.title("Number of Body Style frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('body-style');
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax=sns.countplot(automobile_data['make'],palette='dark',edgecolor='k',linewidth=2)
plt.xticks(rotation='vertical')
plt.xlabel('Car Maker',fontsize=10)
plt.ylabel('Number of Cars',fontsize=10)
plt.title('Cars count by manufacturer',fontsize=10)
ax.tick_params(labelsize=15)
plt.show()

fig = plt.figure(figsize=(10, 10))
mileage=automobile_data.groupby(['make']).mean()
mileage['avg-mpg']=((mileage['city-mpg']+mileage['highway-mpg'])/2)
ax=mileage['avg-mpg'].sort_values(ascending=False).plot.bar(edgecolor='k',linewidth=2)
plt.xticks(rotation='vertical')
plt.xlabel('Car Maker',fontsize=10)
plt.ylabel('Number of Cars',fontsize=10)
plt.title('Fuel Economy of Car Makers',fontsize=20)
ax.tick_params(labelsize=20)
plt.show()
plt.show()

X=automobile_data[['curb-weight','engine-size','horsepower','width','highway-mpg']]
Y=automobile_data['price']
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y,train_size=0.8, test_size=0.2, random_state=0)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
predicted_values=prediction.astype(int)
print(predicted_values)

