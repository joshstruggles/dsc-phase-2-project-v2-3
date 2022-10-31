
## Final Project Submission

Please fill out:
* Student name: Joshua Ruggles
* Student pace: **self paced** / part time / full time
* Scheduled project review date/time: 
* Instructor name: Joe Comeaux
* Blog post URL:


# King County Housing Data

![09112020_2018sea_124804.webp](attachment:09112020_2018sea_124804.webp)

## Business Problem


King County is having a housing crisis! 
    
Okay, well not really. We have been asked by a real estate agency in the area to build a predictive model for the King County housing market; the agency's directive is to establish locations in King County that will return the most investment and have the most profitable features. 
****************

## Raw data 

# Let's reacquaint ourselves with the data. 

import pandas as pd 
df = pd.read_csv('data/kc_house_data.csv')
df7 = df
df.head()

df.info()

from statsmodels.formula.api import ols

# Let's preview the model with all columns that are integers or numbers. Since we are building new homes, we do not need to 
# know anything about yr_built or yr_renovated
outcome = 'price'
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'zipcode', 'sqft_living15', 
            'sqft_lot15']
predictors = '+'.join(features)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=df).fit()
model.summary()

<div class="alert alert-block alert-info">
Just having taken a glance at the data we can tell a few things: 
    <br><br>
    -There are categorical values in the dataset
    <br><br>
    -There is skew; we will need to normalize this data prior to creating our predictive model
    <br><br>
    -There is multicollinearity
</div>

## Multicollinearity

# Based on the information so far, we are interested in finding what affects the 'price' of a home. Let's separate 
# categorical columns from numerical columns. 
import numpy as np
import seaborn as sns 
from matplotlib import pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


af = df.corr()
af

sns.set(rc = {'figure.figsize': (15,5)})
sns.heatmap(af, annot = True);

<div class= "alert alert-block alert-info">Based on the correlation, the column 'price' seems to be most affected by 'sqft_living'. </div>

abs(af) > 0.75

df2 = af.abs().stack().reset_index().sort_values(0, ascending=False)

df2['pairs'] = list(zip(df2.level_0, df2.level_1))

df2.set_index(['pairs'], inplace = True)

df2.drop(columns=['level_1', 'level_0'], inplace = True)

# cc for correlation coefficient
df2.columns = ['cc']

df2.drop_duplicates(inplace=True)

df2[(df2.cc>.75) & (df2.cc<1)]


first = 'sqft_living'
second = 'sqft_above'
price = 'price'


fig, ax = plt.subplots()

ax.scatter(af['sqft_living'], af['price'], alpha=0.5)
ax.set_xlabel('sqft_living')
ax.set_ylabel("price")
ax.set_title("Most Correlated Feature vs. Price");

fig, ax = plt.subplots()

ax.scatter(af['sqft_above'], af['price'], alpha=0.5)
ax.set_xlabel('sqft_above')
ax.set_ylabel("Price")
ax.set_title("Second Most Correlated Feature vs. Price");

<div class="alert alert-block alert-info">
The two columns that are most associated with our target, price are sqft_living and sqft_above. Currently, our business recommendation would be to maximize this space, but that would not speak toward the issues we potentially found earlier with multicollinearity. Let's investigate further.     
    
</div>    

## Build a table for all data

<div class="alert alert-block alert-warning">
We will try modeling our data without the columns 'sqft_living', 'sqft_above' & 'sqft_living15'. We also will not need 'yr_built' (we are focused on brand new houses), 'yr_renovated' (same reason), 'zipcode' (categorical data), 'lat', and 'long'
</div>

df = df[['id', 'price', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'sqft_lot15','grade', 'condition', 'view', 
         'waterfront']]

## Dealing with null data and non-numbers

<div class="alert alert-block alert-info">
We need to split our dataset into numerical data, this means addressing columns grade, condition, view, and waterfront. 
</div>

df.head()

#Looking at the dataset there were three object columns not included in our correlation. Let's include them for now.  
dfa = df
dfa.head()

dfa.info()

dfa.isna().sum()

<div class="alert alert-block alert-info">There are some null values in the dataset. </div>

dfa['condition'].unique()

dfa['grade'].unique()

dfa['view'].unique()

dfa['waterfront'].unique()

<div class="alert alert-block alert-danger">
Unfortunatley, because there are non-numbers in our dataset, we cannot compute an R-squared number from these interactions nor can we test.     
    
</div>

## Object(ion) your honor

<div class="alert alert-block alert-info">
'condition'may prove to be useful in our pursuit of price since bthe condition of a home is an actionable item. We will need to make these categoricals into numbers however in order to see for certain.
</div>

print(dfa['condition'].value_counts())
print()
print(dfa['grade'].value_counts())
print()
print(dfa['view'].value_counts())
print()
print(dfa['waterfront'].value_counts())

<div class="alert alert-block alert-warning">
There are many conditions that a home can be in. That said, unfortunately we will have to place each condition into its own category. We will likely be able to make the 'grade' column into numbers, as well as 'view'. 
</div>

## You take that bad boy out on the water recently? 

<div class="alert alert-block alert-warning">
The two happiest days of a boatowner's life? The day he buys the boat and the day he sells it. Let's see if we are just as ecstatic once we have made this column into a boolean.      
</div>    

dfa['waterfront'].unique()

dfa['waterfront'] = dfa['waterfront'].fillna('NO')

dfa.head()

## One Hot Encoder

from sklearn.preprocessing import OneHotEncoder
df_action = dfa.copy()
cond = df_action[["condition"]]

ohe = OneHotEncoder(categories = 'auto', sparse = False, handle_unknown = 'ignore')
ohe.fit(cond)
ohe.categories_

condition = ohe.transform(cond)
condition

condition = pd.DataFrame(condition, columns = ohe.categories_[0], index = df_action.index)
condition

df_action.drop("condition", axis = 1, inplace = True)
df_action

df_action = pd.concat([df_action, condition], axis = 1)
df_action.head()

df_action.info()

<div class="alert alert-block alert-success">
Now that we have successfully made our 'condition' column into 5 separate features, let's see if we can't get 'grade' and 'view' to do the same.     
</div>

#Let's see again the special values inside of ['grade']
print(df_action['grade'].value_counts())

<div class="alert alert-block alert-info">
Looking at the value counts for 'grade', there are 11 values, 1 of which (poor) only has one value. Let's make this an even 10, since we will get little information from taking our home from 'average' grade to 'poor'.     
</div>

## Your house will be grade(d)

grade = df_action[["grade"]]

ohe = OneHotEncoder(categories = 'auto', sparse = False, handle_unknown = 'ignore')
ohe.fit(grade)
ohe.categories_

grade = ohe.transform(grade)
grade

grade = pd.DataFrame(grade, columns = ohe.categories_[0], index = df_action.index)
grade

df_action.drop("grade", axis = 1, inplace = True)
df_action

df_action = pd.concat([df_action, grade], axis = 1)
df_action.head()

<div class="alert alert-block alert-success">
Another success, though this doesn't make our table particularly readable, we have numbers in all but our last feature.     
</div>

## Have you seen the view from my house? 

print(df_action['view'].value_counts())

# We will have to change "NONE" to "N/A" in order for us to move forward. 

df_action['view'].isna().sum()

df_action['view'].unique()

# There are 63 values in df_action['view'] that are unaccounted for. We should suspect that a 'nan' variable will affect
# our dataset

df_action["view"] = df_action["view"].fillna("N/A")
df_action['view'].value_counts()

view = df_action[["view"]]

ohe = OneHotEncoder(categories = 'auto', sparse = False, handle_unknown = 'ignore')
ohe.fit(view)
ohe.categories_



view = ohe.transform(view)
view

df_action['view'].isna().sum()

view = pd.DataFrame(view, columns = ohe.categories_[0], index = df_action.index)
view

df_action.drop("view", axis = 1, inplace = True)
df_action

df_action = pd.concat([df_action, view], axis = 1)
df_action.head()

df_action = df_action.drop(['id'], axis = 1)
df_action.head()

<div class="alert alert-block alert-success">
Success! Now that all of our columns are numerical, it's time for us to split this data.
    
</div>

## Ordinal Encoder for boolean values. 
Did you sell that boat yet? You don't even live on the waterfront. 

from sklearn.preprocessing import OrdinalEncoder 

water = df_action[['waterfront']]

# (2) Instantiate an OrdinalEncoder
encoder_water = OrdinalEncoder()

# (3) Fit the encoder on street_train
encoder_water.fit(water)

# Inspect the categories of the fitted encoder
encoder_water.categories_[0]

water_encoded_train = encoder_water.transform(water)

# Flatten shape
water_encoded_train = water_encoded_train.flatten()

# Let's take a look
water_encoded_train

df_action['waterfront'].isna().sum()

df_action["waterfront"] = water_encoded_train

# Visually inspect X_train
df_action['waterfront'].head()

df_action['waterfront'].value_counts()

print(146/(21451 + 146))

<div class="alert alert-block alert-success">
Keep in mind, a value of 0 means that you do not in fact have a waterfront property. Enough with the boat puns I suppose as a lowly 0.6% of homes in the King County area have a waterfront view.      
</div>

## Separation Anxiety

<div class="alert alert-block alert-info">
Because our intentions with this data is to predict the prices of homes we should make that our y-value, and our x-values the remaining columns in df_action.
</div>

np.all(np.isfinite(df_action))

y = df_action['price']
X = df_action.drop("price", axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(model, X_train, y_train, cv=3)

model.fit(X_train, y_train)
model.score(X_test, y_test)

from sklearn.model_selection import cross_validate, ShuffleSplit
baseline_model = LinearRegression()
splitter = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)

baseline_scores = cross_validate(
    estimator = baseline_model, 
    X= X_train, 
    y = y_train, 
    return_train_score = True, 
    cv = splitter
)

print('Train score: ', baseline_scores['train_score'].mean())
print('Validation score: ', baseline_scores['test_score'].mean())

<div class="alert alert-block alert-info">
An R-squared score of 0.68 in our testing model compared to a 0.71 in our training model would suggest that this is an accurate train_test_split. We should consider making this even better however, by creating another model, this time without ordinal categories: ['condition', 'grade', 'view', 'waterfront']. Between the two we should pick the model with a better R-squared value.   
</div>

## The second model. Let's hope it improves! 

df_action2 = dfa.drop(['condition', 'grade', 'view', 'waterfront'], axis = 1).copy()

df_action2.head()

df_action2 = df_action2.drop(['id'], axis = 1).copy()
df_action2.head()

y = df_action2['price']
X = df_action2.drop("price", axis = 1)

# Run your second train_test_split on the new model created, df_action2
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, random_state=42)

model = LinearRegression()
model.fit(X_train2, y_train2)

cross_val_score(model, X_train2, y_train2, cv=3)

model.fit(X_train2, y_train2)
model.score(X_test2, y_test2)

# We should evaluate this second model against the first

second_model = LinearRegression()

second_model_scores = cross_validate(
    estimator = second_model, 
    X = X_train2,
    y= y_train2, 
    return_train_score = True,
    cv = splitter)
print("Current Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())

<div class="alert alert-block alert-danger">
Well, not exactly what we had in mind. Though we removed ordinal categories, our R-squared number suggests that we are explaining up to 10% less of the variance in our model. Let's use our ordinal categories and do some predictions. Also, because we are evaluating model performance, we should only concern ourselves with the validation score at this point.      
    
</div>

## Third model

<div class="alert alert-block alert-warning">
We will add back in our deleted categorical values as their dummy columns. 
</div>

significant_features = ['bedrooms', 'bathrooms',  'floors', 'sqft_lot','sqft_lot15', 'Average', 'Fair', 'Good', 'Poor', 'Very Good', '10 Very Good', 
                        '11 Excellent', '12 Luxury', '13 Mansion', 'AVERAGE', 'EXCELLENT', 'FAIR', 'GOOD',  
                        'waterfront']

third_model = LinearRegression()
X_train_third_model = X_train[significant_features]

third_model_scores = cross_validate(
    estimator = third_model,
    X = X_train_third_model, 
    y = y_train, 
    return_train_score = True, 
    cv= splitter
)

print("Current Model")
print("Train score:     ", third_model_scores["train_score"].mean())
print("Validation score:", third_model_scores["test_score"].mean())
print()
print("Second Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())

import statsmodels.api as sm
third_model = LinearRegression
model = sm.OLS(y_train, X_train_third_model).fit()
model.summary()

<div class="alert alert-block alert-success">
Though it isn't much of an improvement, our current model is running about the same as our baseline model. We have addressed the issue of having a feature that did not prove to be statistically significant, so this model is our best.      
    </div>

## Standard Scale

<div class="alert alert-block alert-info">
Sklearn has a feature selection option to give us insight as to what features we should use for our model. 
</div>

from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

# Importances are based on coefficient magnitude, so
# we need to scale the data to normalize the coefficients
X_train_for_RFECV = StandardScaler().fit_transform(X_train_third_model)

model_for_RFECV = LinearRegression()

# Instantiate and fit the selector
selector = RFECV(model_for_RFECV, cv=splitter)
selector.fit(X_train_for_RFECV, y_train)

# Print the results
print("Was the column selected?")
for index, col in enumerate(X_train_third_model.columns):
    print(f"{col}: {selector.support_[index]}")

df = df_action[significant_features]
target = df_action['price']

import statsmodels.api as sm
predictors = sm.add_constant(df)
predictors.head()

model = sm.OLS(target, df).fit()
model.summary()

predictor_scaled = (df - np.mean(df)) / np.std(df)

scaled = predictor_scaled
scaled.head()

predictors = sm.add_constant(predictor_scaled)
model = sm.OLS(target, predictors).fit()
model.summary()

<div class="alert alert-block alert-info">
Due to the coefficients not matching with the rest of the dataset after scaling, we will take out the following columns: 'Average', 'Fair', 'Good', 'Poor', and 'Very Good'    
    
</div>    

y = df_action['price']
X = scaled[['bedrooms', 'bathrooms',  'floors', 'sqft_lot','sqft_lot15', '10 Very Good', 
                        '11 Excellent', '12 Luxury', '13 Mansion', 'AVERAGE', 'EXCELLENT', 'FAIR', 'GOOD',  
                        'waterfront']]
X.head()

predictors = sm.add_constant(scaled)
model = sm.OLS(target, predictors).fit()
model.summary()

<div class="alert alert-block alert-success">
Great! Our data is now standardized. 
    </div>

## Train_test_split on our scaled data

X_train3, X_test3, y_train, y_test = train_test_split(X,y,random_state=42)

model = LinearRegression()
model.fit(X_train3, y_train)

model.fit(X_test3,y_test)

cross_val_score(model, X_train3, y_train, cv=3)

model.fit(X_train3, y_train)
model.score(X_test3, y_test)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, model.predict(X_test3), squared=False)

<div class="alert alert-block alert-danger">
Well, admittedly this doesn't look great. We can account for about 52% of the variance in our model, meaning that there is a standard deviation of around $256,130. We will need to have someone double-check these figures. 

</div>

fourth_model = LinearRegression()
X_train_fourth_model = X_train3

fourth_model_scores = cross_validate(
    estimator = fourth_model,
    X = X_train_fourth_model, 
    y = y_train, 
    return_train_score = True, 
    cv= splitter
)
print("Current Model")
print("Train score:     ", fourth_model_scores["train_score"].mean())
print("Validation score:", fourth_model_scores["test_score"].mean())
print()
print("Third Model")
print("Train score:     ", third_model_scores["train_score"].mean())
print("Validation score:", third_model_scores["test_score"].mean())
print()
print("Second Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())

fourth_model.fit(X_test3,y_test)

preds = fourth_model.predict(X_test3)
fig, ax = plt.subplots()

perfect_line = np.arange(y_test.min(), y_test.max())
ax.plot(perfect_line, linestyle="--", color="orange", label="Perfect Fit")
ax.scatter(y_test, preds, alpha=0.5)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.legend();

<div class="alert alert-block alert-success">
Based on our visual it looks like more than 50% of our variance actually falls UNDER our model's predicted pricing. Great if you're buying a home, good information to have if you're a seller.      
    </div>

import scipy.stats as stats

residuals = (y_test - preds)
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True);

<div class="alert alert-block alert-info">
'floors' looks to be the only non-normal distribution.    
</div>

## Independence Assumption

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_train3.values, i) for i in range(X_train3.shape[1])]
pd.Series(vif, index=X_train3.columns, name="Variance Inflation Factor")

![1_RnwCs5nzz4ggJEtdZsg9kw.png](attachment:1_RnwCs5nzz4ggJEtdZsg9kw.png)

<div class="alert alert-block alert-success">
Values are well below 5.0, highly correlated. Our model shows mild multicollinearity, allowable for our purposes. 
</div>

# Predictive Model

df = pd.read_csv('data/kc_house_data.csv')

continuous = ['sqft_lot', 'sqft_lot15']
categoricals = ['bedrooms', 'bathrooms', 'waterfront', 'view', 'grade', 'price']

## Continuous Features

df_continuous = df[continuous]

#log features 
log_names = [f'{column}_log' for column in df_continuous.columns]

df_log = np.log(df_continuous)
df_log.columns = log_names

#normalize (subtract mean and divide by std)

def normalize(feature): 
    return (feature - feature.mean()) / feature.std()
df_log_norm = df_log.apply(normalize)

df_log_norm.head()

## Categorical Features

# categoricals = ['bedrooms', 'bathrooms', 'waterfront_NO', 'waterfront_YES', 'view_AVERAGE', 'view_EXCELLENT', 'view_FAIR'
#               ,'view_GOOD', 'grade_10 Very Good', 'grade_11 Excellent', 'grade_12 Luxury', 'grade_13 Mansion']

df_ohe = pd.get_dummies(df[categoricals])
df_ohe.head()

df_ohe1 = df_ohe[['bedrooms', 'bathrooms', 'waterfront_NO', 'waterfront_YES', 'view_AVERAGE', 'view_EXCELLENT', 'view_FAIR'
               ,'view_GOOD', 'grade_10 Very Good', 'grade_11 Excellent', 'grade_12 Luxury', 'grade_13 Mansion']]

df_ohe1.head()

## Combine Features

preprocessed = pd.concat([df_log_norm, df_ohe1], axis = 1)
preprocessed.head()

## Linear Model with price_log as target in statsmodels

X = preprocessed
y = df_ohe['price']

X.info()

X_int = sm.add_constant(X)
model = sm.OLS(y, X_int).fit()
model.summary()

final = LinearRegression()
final.fit(X_int,y)
final.score(X_int,y)

print(pd.Series(final.coef_, index=X_int.columns, name="Coefficients"))
print()
print("Intercept:", final.intercept_)

mean_squared_error(y,final.predict(X_int), squared = False)

## Run model in scikit-learn

linreg= LinearRegression()
linreg.fit(X,y)
linreg.coef_

linreg.intercept_

categoricals.remove("price")

used_cols = [*continuous, *categoricals]
used_cols

new_row = pd.DataFrame(columns=used_cols)

new_row = new_row.append({
'bedrooms': 4, 
'bathrooms': 4, 
'waterfront': 'NO',
'view': "NONE",
'grade': '11 Excellent',
'sqft_lot': 1500, 
'sqft_lot15': 2500}, ignore_index = True)
new_row

new_row_cont = new_row[continuous]

log_names = [f'{column}_log' for column in new_row_cont.columns]

new_row_log = np.log(new_row_cont.astype(float)) #function will not work unless this is a float
new_row_log.columns = log_names

#normalizing
for col in continuous: 
    #normalize using mean and std from overall dataset
    new_row_log[f'{col}_log'] = (new_row_log[f'{col}_log'] - df[col].mean()) / df[col].std()
new_row_log


categoricals = ['waterfront', 'view', 'grade']

new_row_cat = new_row[categoricals]

new_row_ohe = pd.DataFrame(columns = df_ohe.columns)

ohe_dict = {}
for col_type in new_row_cat.columns: 
    col_list = [c for c in new_row_ohe.columns.to_list() if col_type in c]
    for x in col_list: 
        if new_row_cat[col_type][0] in x: 
            ohe_dict[x] = 1
        else: 
            ohe_dict[x] = 0
new_row_ohe = new_row_ohe.append(ohe_dict, ignore_index=True)
new_row_ohe

df_ohe2 = new_row_ohe[['bedrooms', 'bathrooms', 'waterfront_NO', 'waterfront_YES', 'view_AVERAGE', 'view_EXCELLENT', 'view_FAIR'
               ,'view_GOOD', 'grade_10 Very Good', 'grade_11 Excellent', 'grade_12 Luxury', 'grade_13 Mansion']]

df_ohe2.head()

new_row_processed = pd.concat([new_row_log, df_ohe2], axis = 1)
new_row_processed.info()

new_row_processed['bedrooms'] = 4

new_row_processed['bathrooms'] = 4

new_row_processed

new_row_pred_log = linreg.predict(new_row_processed)
new_row_pred_log

# Conclusion

<div class="alert alert-block alert-info">
Our first model was run on just the raw data in the King County housing dataset. This was meant to give us a snapshot of features that we may want to focus on, I included a heatmap and .corr function at the beginning of the project to give us better insight into what contributes to the price of a home. Unfortunately our second model cut out all non-numeric values and gave us our worst predictive model. 

After the failure of our secondary model we were able to break down our categorical columns, grade, condition, view, and waterfront. The first three we settled as arrays of the various categories, the waterfront data we were able to create boolean values. 
    
Our fourth model utilized standard scaling across the entirety of the dataset. We pulled our information on cost of features from this model. 
    
    
* Though this final model is not the best in regards to R-squared score, it includes the least amount of multicollinearity, giving us a clearer picture regarding what features contribute to the price of a home. 
    
    
    
</div>    

