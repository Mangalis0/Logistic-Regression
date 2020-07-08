# Logistic Regression on Credit Card Approval Dataset

**Completed by Mangaliso Makhoba.**

**Overview:** Build a model that will predict the Approval Status given features: Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the ApprovalStatus.

**Problem Statement:** Evaluate the performance of the Logistic Regression Model.

**Data:** [Credit Card Approval dataset](http://archive.ics.uci.edu/ml/datasets/credit+approval)

**Deliverables:** Write a function which calculates the Accuracy, Precision, Recall and F1 scores.

## Topics Covered

1. Machine Learning
3. Classification
4. Logistic Regression
5. Accuracy
6. Precision
7. Recall
8. F1 Score

## Tools Used
1. Python
1. Pandas
2. Scikit-learn
2. Jupyter Notebook

## Installation and Usage

Ensure that the following packages have been installed and imported.

```bash
pip install numpy
pip install pandas
pip install sklearn
```

#### Jupyter Notebook - to run ipython notebook (.ipynb) project file
Follow instruction on https://docs.anaconda.com/anaconda/install/ to install Anaconda with Jupyter. 

Alternatively:
VS Code can render Jupyter Notebooks

## Notebook Structure
The structure of this notebook is as follows:

 - First, we will start off by loading and viewing the dataset.
 - We will see that the dataset has a mixture of both numerical and non-numerical features; that it contains values from different ranges; and that it contains a number of missing entries.
 - Based upon the observations above, we will preprocess the dataset to ensure the machine learning model we choose can make good predictions.
 - After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
 - Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.



# Function 1: Data Cleaning
Write a function to clean the given data . The function should:
* Replace the '?'s with NaN.
* Impute the missing values with mean imputation.
* Impute the missing values of non-numeric columns with the most frequent values as present in the respective columns.

_**Function Specifications:**_
* Should take a pandas Dataframe and column name as input and return a list as an output.

* The list should be a count of unique values in the column

_**Expected Outputs:**_
    

```python
data_cleaning(df, 0) == [480, 210]
data_cleaning(df, 9) == [395, 295]
```


# Function 2: Data Preprocessing

Write a function to pre-process the data so that we can run it through the classifier. The function should:
* Convert the non-numeric data into numeric using sklearn's ```labelEncoder``` 
* Drop the features 11 and 13 and convert the DataFrame to a NumPy array
* Split the data into features and labels
* Standardise the features using sklearn's ```MinMaxScaler```
* Split the data into 80% training and 20% testing data.
* Use the `train_test_split` method from `sklearn` to do this.
* Set random_state to equal 42 for this internal method. 

_**Function Specifications:**_
* Should take a dataframe as input.
* Should return two `tuples` of the form `(X_train, y_train), (X_test, y_test)`.

_**Expected Outputs:**_

```python
(X_train, y_train), (X_test, y_test) = data_preprocess(df)
print(X_train[:1])
print(y_train[:1])
print(X_test[:1])
print(y_test[:1])
```

```python    
[[1.         0.25787966 0.48214286 1.         1.         0.42857143
  0.33333333 0.         0.         0.         0.         0.
  0.        ]]
[1.]
[[0.5        1.         0.05357143 0.66666667 0.33333333 0.42857143
  0.33333333 0.         0.         1.         0.02985075 0.
  0.00105   ]]
[1.]
```

# Function 3: Training the model

Now that we have formatted our data, we can fit a model using sklearn's `LogisticRegression` class with solver 'lbfgs'. Write a function that will take as input `(X_train, y_train)` that we created previously, and return a trained model.

_**Function Specifications:**_
* Should take two numpy `arrays` as input in the form `(X_train, y_train)`.
* The returned model should be fitted to the data.

_**Expected Outputs:**_

```python
lm = train_model(X_train, y_train)
print(lm.intercept_[0])
print(lm.coef_)
```
```
1.5068926456005878
[[ 0.25237869 -0.22847881 -0.01779302  2.00977742  0.23903441 -0.29504922
  -0.08952344 -0.83468871 -3.48756932 -1.07648711 -0.83688921  0.07860585
  -1.3077735 ]]
```
# Function 4: Testing the model

AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes. Write a function which returns the roc auc score of your trained model when tested with the test set.

_**Function Specifications:**_
* Should take the fitted model and two numpy `arrays` `X_test, y_test` as input.
* Should return a `float` of the roc auc score of the model. This number should be between zero and one.


_**Expected Outputs:**_
    
```python
print(roc_score(lm,X_test,y_test))
```
```python
0.8865546218487395
```

# Function 5: Accuracy, Precision, Recall and F1 scores

Write a function which calculates the Accuracy, Precision, Recall and F1 scores.

_**Function Specifications:**_
* Should take the fitted model and two numpy `arrays` `X_test, y_test` as input.
* Should return a tuple in the form (`Accuracy`, `Precision`, `Recall`, `F1-Score`)

_**Expected Outputs:**_
```python
(accuracy, precision, recall, f1) = scores(lm,X_test,y_test)
    
print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1 score: %f' % f1)
```
```python
Accuracy: 0.833333
Precision: 0.846154
Recall: 0.808824
F1 score: 0.827068
```

## Conclusion
All the statistics of performance are at least 80%. The base model of the Logistic Regression is performing reasonably well, we can look to improve these statistics by tuning the hyperparameter 'C' of the Logistic Regression model. The F1 score is the most important statistic compared to the "Accuracy" as it takes into account both the recall and the precision which would then account for imbalances which may exist in the data. Accuracy on the other hand looks at the overall proportion labels that the model got right, this would give a false confidence if we were would rely on it as a may be very good at predicting the majority label, but very poor at poor at predicting the minority, therefore accuracy does not evenly weigh the data, unlike the F1 which will always work better. 

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## Contributing Authors
**Authors:** Mangaliso Makhoba, Explore Data Science Academy

**Contact:** makhoba808@gmail.com

## Project Continuity
This is project is complete


## License
[MIT](https://choosealicense.com/licenses/mit/)