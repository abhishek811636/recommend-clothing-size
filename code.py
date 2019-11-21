# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Load the dataframe
df = pd.read_json(path, lines=True)

# replace the space with _
df.columns=df.columns.str.strip().str.lower().str.replace(' ', '_')

# Check the percentage of missing value
missing_data = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().sum()/82790)*100})
print(missing_data)


# drop the the columns
df.drop(['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'], axis=1, inplace=True)



#code starts here

# store independent variable
X = df.drop(['fit'],axis=1)

# store dependent variable
y = df['fit']

# spliting the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.33,random_state=6)



# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here
# plot the bar chart
g_by_category = df.groupby('category')
cat_fit = g_by_category['fit'].value_counts()
cat_fit = cat_fit.unstack()
plot_barh(cat_fit, 'fit')

# Code ends here


# --------------
# Code starts here
cat_len=g_by_category['length'].value_counts()
cat_len=cat_len.unstack()
plot_barh(cat_len,'length')

# Code ends here


# --------------
# function to to convert feet to inches

def get_cms(x):
    if type(x) == type(1.0):
        return
    #print(x)
    try: 
        return (int(x[0])*30.48) + (int(x[4:-2])*2.54)
    except:
        return (int(x[0])*30.48)

# apply on train data    
X_train.height = X_train.height.apply(get_cms)

# apply on testing set
X_test.height = X_test.height.apply(get_cms)


# --------------
print(X_train.isnull().sum())

# drop missing values
X_train.dropna(subset=['height','length','quality'],inplace=True)
X_test.dropna(subset=['height','length','quality'],inplace=True)

y_train=y_train[X_train.index]
y_test=y_test[X_test.index]

# fill missing values with mean
X_train['bra_size'].fillna((X_train['bra_size'].mean()), inplace=True)
X_test['bra_size'].fillna((X_test['bra_size'].mean()), inplace=True)

X_train['hips'].fillna((X_train['hips'].mean()), inplace=True)
X_test['hips'].fillna((X_test['hips'].mean()), inplace=True)

# fill missing value with mode
mode_1 = X_train['cup_size'].mode()[0]
mode_2 = X_test['cup_size'].mode()[0]
X_train['cup_size']=X_train['cup_size'].replace(np.nan,mode_1)
X_test['cup_size']=X_test['cup_size'].replace(np.nan,mode_2)

print(X_train.isnull().sum())




# --------------
# Code starts here

X_train=pd.get_dummies(data=X_train,columns=["category", "cup_size","length"],prefix=["category", "cup_size","length"])

X_test=pd.get_dummies(data=X_test,columns=["category", "cup_size","length"],prefix=["category", "cup_size","length"])
# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# Code starts here
model=DecisionTreeClassifier(random_state=6)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

score=accuracy_score(y_test,y_pred)

precision=precision_score(y_test,y_pred,average=None)
# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here

# instantiate Decision tree classifier model
model=DecisionTreeClassifier(random_state=6)

# grid search on logistic regression
grid = GridSearchCV(estimator=model, param_grid=parameters)
grid.fit(X_train, y_train)
# make predictions 
y_pred = grid.predict(X_test)

# accuracy check
accuracy= grid.score(X_test,y_test)
print(accuracy)




