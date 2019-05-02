import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("stores.csv",parse_dates=[1])

def findHighlyCorrelated():
    # I used this algorithm from https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/ to identify 
    # highly correlated features in the input.

    # Create correlation matrix
    corr_matrix = dataset.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print("Highly Correlated features:", to_drop)
    return to_drop

###
# Converting categorical data to continuous
###

dataset['dayOfWeek'] = dataset['Date'].dt.dayofweek #Split date into other features as they are more useful
dataset['dayOfMonth'] = dataset['Date'].dt.day
dataset['month'] = dataset['Date'].dt.month
dataset['dayOfYear'] = dataset['Date'].dt.dayofyear

###
# Removing categorical data
###

# dayOfWeek - removed because the original Date variable was the end of the week which were mostly Thursday with 3960 data points compared to Wednesday's 180
# Store - store numbers are categorical so they were removed
# Month - between month, dayOfMonth and dayOfYear, dayOfYear is most continuous so only it is use to reduce collinearity.
# Fuel_Price - upon running the linear regression model multiple times selecting random data points, the coefficient for this variable was
# changing a lot from being positive to negatively correlated so it was removed from analysis due to its variance.
# IsHoliday - for the same reasons as Fuel_Price and also it being too dichotomous.

dataset = dataset.drop(["Date","Store", "month", "dayOfMonth", "dayOfWeek", "Fuel_Price", "IsHoliday"], axis=1) #Remove date as we converted it into continuous data 

###
# Removing correlated data
###

# It was found that the features of "Population_over_65" and "Population_under_18" were highly correlated with "Population"
# Both were removed to reduce collinearity and improve the regression formula.

correlatedVariables = findHighlyCorrelated()
dataset = dataset.drop(correlatedVariables, axis=1)

###
# Potential interesting attributes
###

# If I was given an API or database I would find the store locations using the number and bin them into certain areas so that location
# could be also be used as an input when split into dummy variables (since it is categorical) and input into the linear model for potential further insights.

###
# Creating a linear regression model
###

X = dataset.drop(["Weekly_Sales"], axis=1) #Create a dataset for the model
target = dataset["Weekly_Sales"] #Set weekly sales as the model prediction target

#Splitting the data set into two sets to give data to test the linear model.

X_train, X_test, Y_train, Y_test = train_test_split(X, target)
lm = LinearRegression()
lm.fit(X_train, Y_train)
print("Model R^2 for training data", lm.score(X_train, Y_train)) #Model R^2 when used on training data (comprised of 25% of the dataset)
print("Model R^2 for test data", lm.score(X_test, Y_test)) #Model R^2 on test data (comprised of 75% of the dataset)
frame = pd.DataFrame(zip(X.columns, lm.coef_), columns = ['features', 'estimatedCoeff']) #Create a table of the fitted model
print(frame) #Print the table, the conclusion is at the bottom of the page

###
#Plotting and analysis
###

#Using matplotlib to plot the residual plot to see if the regression model centred around zero and not too biased.

plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c = 'b', s = 5, alpha = 0.5) #Plot the training data residuals in blue
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c='g', s=5) #Plot the test data residuals in blue
plt.hlines(y=0, xmin=0, xmax = 50)
plt.title('Residual plot')
plt.ylabel('Residuals')
plt.show()


# Single variable modelling of Unemployment against Weekly Sales, one of the more consistently correlated features of the dataset.

unemployment = dataset.filter(['Unemployment'], axis = 1) # Using only the unemployment attribute
lineBestFitModel = LinearRegression() # Create a new linear model to make a line of best fit for the scatter plot
lineBestFitModel.fit(unemployment, target) # Fit it to predict weekly sales
print("Unemployment only regression model R^2: ", lineBestFitModel.score(unemployment, target)) # Print the R^2 score
y = lineBestFitModel.predict(unemployment)
plt.title('Weekly Sales vs Unemployment')
plt.scatter(dataset.Unemployment, dataset.Weekly_Sales, c='b', s = 5, alpha = 0.5)
plt.ylabel('Weekly Sales')
plt.xlabel('Unemployment(%)')
plt.plot(unemployment,y, c='k') # Plot the line of best fit
plt.show()

# Through repeated testing with random samples of training data it was found that Temperature and Unemployment 
# was most strongly and consistently correlated with weekly sales. Both of them were negatively correlated with
# coefficients of -3300 for temperature and -5000 for unemployment which was consistent
# throughtout the entire data set. The linear regression model had a R^2 score of 0.89
# after removing collinear and categorical data indicating that the model fit the data well.
# Adding location as a field may further improve this R^2 score. 

# Plotting the residuals showed that the predictions were all randomly centered around zero which means that
# the regression model was not too biased (or had much collinearity).

# A scatter plot was made of Unemployment against Weekly Sales to see if the results above holds true. The line of best fit followed a trend which can be seen in the plot,
# the R^2 score for only Unemployment against Weekly Sales was only 0.012 suggesting it is only weakly correlated even though it is the strongest
# out of the attributes available.