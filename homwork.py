#importing the necessary libaries

import pandas as pd
import IPython
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

#importing the dataset
wine_data = pd.read_csv("./Data/wine.data",header = None)
wine_data.columns  = ["Class Label","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines", "Proline"]
print(wine_data["Class Label"].nunique())
print(wine_data["Class Label"].value_counts())

#seperating the target variable from the rest
"""
    The dataset is relatively balanced, The number of data points per class are not so distant from each other.

"""

X = wine_data.drop('Class Label', axis=1)
y = wine_data['Class Label']

print(y.shape, X.shape)

#preparing the dataset for a machine learning algorithm
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 7, test_size = 0.25)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#scaling the data set
scaler = MinMaxScaler()
sc_x=scaler.fit(x_train)
x_train_scaled=sc_x.transform(x_train.values)
x_test_scaled=sc_x.transform(x_test.values)
print(x_train_scaled.shape)

'''
We are going to run random forest on both the scaled data and the unscaled data to see which performs better
before we go onto tuning parameters
'''
rClassifier_1 = RandomForestClassifier()
rClassifier_2 = RandomForestClassifier()

rClassifier_1.fit(x_train,y_train)
y_preds = rClassifier_1.predict(x_test)

rClassifier_2.fit(x_train_scaled,y_train)
y_preds1 = rClassifier_2.predict(x_test_scaled)

print('The accuracy score for Random Forest Unscaled is', accuracy_score(y_test,y_preds)*100,"%")
print('The accuracy score for Random Forest scaled is', accuracy_score(y_test,y_preds1)*100,"%")

#visualizing the  most important features
feat_importances = pd.Series(rClassifier_1.feature_importances_, index=X.columns)
feat_importances.nlargest(13).plot(kind='barh')

#visulaizing  a single treee
estimator = rClassifier_1.estimators_[10]
target_names = ["Class 1", "Class 2", "Class 3"]

export_graphviz(estimator, out_file='tree.dot',
                feature_names = X.columns,
                class_names = target_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')