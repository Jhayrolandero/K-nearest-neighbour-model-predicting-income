import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Reading the data file
df = pd.read_csv("adult.data")

# Machine can't read strings so we will convert it into integer
encoder = preprocessing.LabelEncoder()

# Converting all attributes to integer
age = encoder.fit_transform(list(df["age"]))
workclass = encoder.fit_transform(list(df["workclass"]))
fnlwgt = encoder.fit_transform(list(df["fnlwgt"]))
education = encoder.fit_transform(list(df["education"]))
education_num = encoder.fit_transform(list(df["education-num"]))
marital_status = encoder.fit_transform(list(df["marital-status"]))
occupation = encoder.fit_transform(list(df["occupation"]))
relationship = encoder.fit_transform(list(df["relationship"]))
race = encoder.fit_transform(list(df["race"]))
sex = encoder.fit_transform(list(df["sex"]))
capital_gain = encoder.fit_transform(list(df["capital-gain"]))
capital_loss = encoder.fit_transform(list(df["capital-loss"]))
hours_per_week = encoder.fit_transform(list(df["hours-per-week"]))
native_country = encoder.fit_transform(list(df["native-country"]))
income = encoder.fit_transform(list(df["income"]))

predict = "income"


# Making dataset 
X = list(zip(age, workclass, fnlwgt, education, education_num, marital_status, occupation, 
             relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country))

y = list(income)

# Splitting our dataset intro two, one: is the Training set, Two: Test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.1)

# Making our model using K-nearest neighbour
model = KNeighborsClassifier(n_neighbors=7)

# time to fit the training set
model.fit(x_train, y_train)

# Testing our model accuracy
acc = model.score(x_test, y_test)
print(f"Accuracy: {acc}")

# Testing the prediction of our model

income_var = [">50k", "<=50k"]

prediction = model.predict(x_test) 

for x in range(len(prediction)):
    print(f"Prediction: {income_var[prediction[x]]}\nActual Income: {income_var[y_test[x]]}")
    print("")