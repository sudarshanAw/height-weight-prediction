
#importing libraries
import pandas as pnd
import matplotlib.pyplot as mplot

#dataset 
dataset = pnd.read_csv('weight-height.csv')

#dependent and independent variable
x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

mplot.scatter(x,y) 
mplot.xlabel('height')
mplot.ylabel('weight')

#splitting into training and testsets 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1 ,  random_state = 0);

#simple linear regression model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train)
y_predict = regressor.predict(x_test.reshape(-1,1))

# Visualising the Training set results
mplot.scatter(x_train, y_train, color = 'red')
mplot.plot(x_train, regressor.predict(x_train.reshape(-1,1)), color = 'blue')
mplot.title('Height Vs Weight (Training set)')
mplot.xlabel('Height')
mplot.ylabel('Weight')
mplot.show()

# Visualising the Test set results
mplot.scatter(x_test, y_test, color = 'red')
mplot.plot(x_train, regressor.predict(x_train.reshape(-1,1)), color = 'blue')
mplot.title('Height Vs Weight (Test set)')
mplot.xlabel('Height')
mplot.ylabel('Weight')
mplot.show()