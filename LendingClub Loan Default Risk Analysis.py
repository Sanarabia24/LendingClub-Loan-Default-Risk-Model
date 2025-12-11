#!/usr/bin/env python
# coding: utf-8

# In[108]:


# Upgrading Typing_extensions
pip install --upgrade typing_extensions


# In[104]:


# Installing Libraries
get_ipython().system('pip install catboost')
get_ipython().system('pip install shap')


# In[1]:


#--------------------------#
### Importing Libraries ### 
#--------------------------#
import pandas as pd
import numpy as np
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import shap
shap.initjs()

from catboost import CatBoostClassifier
from catboost import Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from collections import Counter

import joblib


# In[2]:


#--------------------------#
### Data Processing ### 
#--------------------------#


# In[9]:


# Importing the cleaned LendingClub loan data 
df = pd.read_csv('C:/Users/91988/OneDrive/Desktop/Data Analyst Portfolio/SQL Project/Lending Club Loan Data.csv')


# In[10]:


# Looking at Data
df.head()


# In[11]:


# Checking for the % of missing values in each column for a better understanding
missing_pct = (df.isna().mean() * 100).round(3).sort_values(ascending=False)
print("\nMissing % (top 20):\n", missing_pct.head(20))


# In[12]:


# Dropping columns with highest missing values which cant be used for ML

df.drop(columns=['mths_since_last_record', 'mths_since_last_major_derog'], inplace=True, errors='ignore')


# In[13]:


# Checking whether the data type changed while importing the data from SQL
df.info()
df.describe(include='all', datetime_is_numeric=True)


# In[14]:


#------------------------#
### Data Cleaning ###
#------------------------#


# In[15]:


# Converting all the date columns to datetime data type 
df['issue_d'] = pd.to_datetime(df['issue_d'])
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])


# In[16]:


df['issue_d'].head()


# In[17]:


###  Extracting numeric value ###


# In[18]:


# Separating 'months' from the numerical value from term column and converting it to INT 
df['term'] = df['term'].astype(str).str.extract('(\d+)').astype(int)

df['term'].dtype
df['term'].unique()


# In[19]:


# Extracting the numbers from Emp Length
df['emp_length'] = df['emp_length'].replace({
    '10+ years': 10,
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 year': 1,
    '< 1 year': 0,
    'n/a': None
}).astype('float')


# In[20]:


df['emp_length'].dtype
df['emp_length'].unique()


# In[21]:


# Extracting the first 3 numbers from the zip code column
df['zip_updated'] = df['zip_code'].astype(str).str[:3]


# In[22]:


df['zip_updated'].head()


# In[23]:


# Dropping the zip_code column
df = df.drop(columns=["zip_code"])


# In[24]:


# Extracting year and month into from the date columns into new columns
df['issue_year'] = df['issue_d'].dt.year
df['issue_month'] = df['issue_d'].dt.month

df['earliest_cr_year'] = df['earliest_cr_line'].dt.year
df['earliest_cr_month'] = df['earliest_cr_line'].dt.month


# In[25]:


# Replace NaT in year/month columns with np.nan (CatBoost can handle np.nan)
date_cols = ['issue_year', 'issue_month', 'earliest_cr_year', 'earliest_cr_month']

for col in date_cols:
    df[col] = df[col].astype('float')   # ensure numeric dtype
    df[col] = df[col].replace({pd.NaT: np.nan})


# In[26]:


#Filling the missing values in Earliest_cr_year and month to -1 to avoid interference when we run the model. 
#The model will treat '-1' as a separate category bin. This avoids creating artificial dates and data leakage 
df['earliest_cr_year'] = df['earliest_cr_year'].fillna(-1).astype(int)
df['earliest_cr_month'] = df['earliest_cr_month'].fillna(-1).astype(int)


# In[27]:


#Checking if we have missing values
df.isna().sum()[['earliest_cr_year','earliest_cr_month']]


# In[28]:


#Dropping the original date columns as there's missing values in them and would interfere when we run the model
df = df.drop(columns=["issue_d", "earliest_cr_line"])


# In[29]:


#---------------------------------#
### Checking for Outliers ###
#---------------------------------#


# In[30]:


# Choosing numeric columns to cap 
numeric_to_cap = ['loan_amnt','funded_amnt','installment','int_rate','annual_inc',
    'revol_bal','dti','revol_util','tot_coll_amt','tot_cur_bal','total_rev_hi_lim']


# In[31]:


# Saving a copy of the dataset before capping the amount columns
df_before = df[numeric_to_cap].copy()


# In[32]:


#Visualising the outliers in the amount columns before capping
plt.figure(figsize=(14, 6))
df_before.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Columns BEFORE Outlier Capping")
plt.show()


# In[33]:


for col in numeric_to_cap:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df_before[col].dropna())
    plt.title(f"Outliers in {col} (Before Capping)")
    plt.ylabel(col)
    plt.show()


# In[34]:


# Using winsorization by IQR for amount columns
def cap_outliers_iqr(series, lower_q=0.25, upper_q=0.75, factor=1.5):
    q1 = series.quantile(lower_q)
    q3 = series.quantile(upper_q)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)

numeric_to_cap = [c for c in numeric_to_cap if c in df.columns]

for c in numeric_to_cap:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c] = cap_outliers_iqr(df[c])
    


# In[35]:


# Comparing before vs After capping of amount columns
df_after = df[numeric_to_cap]

comparison = pd.DataFrame({
    'before_min': df_before.min(),
    'after_min':  df_after.min(),
    'before_max': df_before.max(),
    'after_max':  df_after.max()})

comparison


# In[36]:


# To get all rows where capping actually happened:
changes = df_before[col] != df[col]
df_before.loc[changes, col].head(), df.loc[changes, col].head()


# In[37]:


# Visually comparing before vs After capping of amount columns (for each column)
df_after = df[numeric_to_cap]

for col in numeric_to_cap:
    plt.figure(figsize=(10, 4))

    # Before
    plt.subplot(1, 2, 1)
    plt.boxplot(df_before[col].dropna())
    plt.title(f"{col} - Before")

    # After
    plt.subplot(1, 2, 2)
    plt.boxplot(df_after[col].dropna())
    plt.title(f"{col} - After")

    plt.tight_layout()
    plt.show()



# In[38]:


# Visualising amount columns after capping
plt.figure(figsize=(14, 6))
df_after.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Columns After Outlier Capping")
plt.show()


# In[39]:


# From the above graphs and table we can see that 'tot_coll_amt' has mostly 0 values along with the missing values
# and there's a major variation between the 0 values and the other amounts which show up as outliers. Hence we will
# be dropping the column as it is unusable for our modelling.


# In[40]:


# Dropping the 'tot_coll_amt'
df.drop('tot_coll_amt', axis=1, inplace=True)


# In[41]:


# Detecting outliers in all the count columns
count_cols = ['delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
    'total_acc', 'acc_now_delinq', 'collections_12_mths_ex_med']

def detect_count_outliers(df, count_cols):
    outlier_summary = {}

    for col in count_cols:
        if col not in df.columns:
            continue
        
        series = df[col].dropna()

        # Summary statistics
        summary = {
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'median': series.median(),
            'p95': series.quantile(0.95),
            'p99': series.quantile(0.99),
            'num_unique': series.nunique(),
            'value_counts_top10': series.value_counts().head(10).to_dict()}

        # Identify "rare" high values
        threshold = series.quantile(0.99)
        rare_high_outliers = df[df[col] > threshold][col]

        summary['num_rare_high'] = rare_high_outliers.shape[0]
        summary['rare_high_values'] = rare_high_outliers.unique().tolist()

        outlier_summary[col] = summary

    return outlier_summary

# Run it
count_outlier_report = detect_count_outliers(df, count_cols)
count_outlier_report


# In[42]:


# Since none of these count fields above show any problematic outliers we do not need to cap them.
# They are naturally skewed, but that’s expected for credit-related count data.


# In[43]:


# Visualising the outliers of all the count columns

# Keeping only columns that exist in the dataframe
count_cols = [c for c in count_cols if c in df.columns]

# Create layout for subplots
n_cols = 3
n_rows = int(np.ceil(len(count_cols) / n_cols))

plt.figure(figsize=(15, 5 * n_rows))

for i, col in enumerate(count_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    df[col].plot(kind='box')
    plt.title(f"Boxplot: {col}")
    plt.ylabel("Value")

plt.tight_layout()
plt.show()


# In[44]:


#----------------------------------------------#
### Determining Numeric Data's Linearity ###
#----------------------------------------------#


# In[45]:


# Cleaning/Mapping our target column - 'Loan_status'

# Mapping each option to good or bad status(0/1)
mapping = {"fully paid": 0,
    "current": 0,
    "in grace period": 0,           # borderline, will be treating as “not yet default”
    "late (16-30 days)": 1,
    "late (31-120 days)": 1,
    "charged off": 1,
    "default": 1,
    "does not meet the credit policy status:fully paid": 0,
    "does not meet the credit policy status:charged off": 1 }

# Mapping to binary
df['loan_default_flag'] = (
    df['loan_status']
    .str.lower()
    .str.replace(".", "", regex=False)
    .str.split(" status: ").str[-1]   # removing "does not meet credit policy..." wrapper
    .map(mapping))
               
               
# Preview
df[['loan_status', 'loan_default_flag']].head(20)


# In[46]:


#Checking if we have any missing values in the loan_default_flag column which is our target column 
df['loan_default_flag'].isna().sum()


# In[47]:


#Dropping the loan_status column since now we have the cleaned binary version. We drop this column as the
# CatBoost model may leak target information.
df = df.drop(columns=["loan_status"], errors="ignore")


# In[48]:


df.info()


# In[49]:


# Selecting numeric columns
numeric_cols_1 = ['annual_inc','revol_util','tot_cur_bal','total_rev_hi_lim',
    'delinq_2yrs','inq_last_6mths','open_acc','pub_rec','total_acc',
    'collections_12_mths_ex_med','acc_now_delinq']


# In[50]:


#Checking if our numeric data is linear or not through scatter plot

n_cols = 3                                # Number of plots per row
n_rows = int(np.ceil(len(numeric_cols_1) / n_cols))

plt.figure(figsize=(18, 5 * n_rows))

for i, col in enumerate(numeric_cols_1, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.scatterplot(
        data=df,
        x=col,
        y='loan_default_flag',
        hue='loan_default_flag',
        alpha=0.3,
        legend=False)
    
    plt.title(f"{col} vs loan_default_flag")
    plt.xlabel(col)
    plt.ylabel("loan_default_flag")

plt.tight_layout()
plt.show()


# In[51]:


#---------------------------------------------------------------------#
### Feature engineering for our ML model to perform better ###
#---------------------------------------------------------------------#


# In[52]:


# Cleaning revol_util
df["revol_util_clean"] = df["revol_util"].astype(float) / 100


# In[53]:


# Calculating Credit Utilization ratio
df["credit_util_ratio"] = df["revol_bal"] / df["total_rev_hi_lim"]


# In[54]:


# Calculating Income to Loan 
df["income_to_loan"] = df["annual_inc"] / (df["loan_amnt"] + 1)


# In[55]:


# Calculating loan Funded Ratio (Shows how much of loan was funded)
df["funded_ratio"] = df["funded_amnt"] / (df["loan_amnt"] + 1)


# In[56]:


# Creating DTI Buckets
#DTI is rarely linear. Bucketing improves signal.
df["dti_bucket"] = pd.cut(df["dti"],
    bins=[-1, 10, 20, 30, 40, 50, 1000],
    labels=["0-10", "11-20", "21-30", "31-40", "41-50", "50+"])


# In[57]:


# Calculating Numeric Interactions

#DTI x Interest Rate
df["dti_x_intrate"] = df["dti"] * df["int_rate"]


# In[58]:


# Loan Amount × Grade Score
grade_map = {"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7}
df["grade_num"] = df["grade"].map(grade_map)

df["loan_x_grade"] = df["loan_amnt"] * df["grade_num"]


# In[59]:


# Revolving Balance as % of Income
df["revol_bal_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)


# In[60]:


# Installment Burden (Installment ÷ Monthly Income)
df["installment_burden"] = df["installment"] / (df["annual_inc"]/12)


# In[61]:


# Creating credit_age from earliest_cr_year
df["credit_age"] = df["issue_year"] - df["earliest_cr_year"]


# In[62]:


# Adding months
df["credit_age_months"] = df["credit_age"] * 12 + (df["issue_month"] - df["earliest_cr_month"])


# In[63]:


# Creating Delinquency Binning for delinq_2yrs
df["delinq_2yrs_bin"] = pd.cut(df["delinq_2yrs"],
    bins=[-1, 0, 1, 3, 100],
    labels=["none", "low", "medium", "high"])


# In[64]:


# Useful One-hot Flags
df["purpose_small_business"] = (df["purpose"] == "small_business").astype(int)


# In[65]:


# Calculating debt-to-income balance ratio
df["cur_bal_to_income"] = df["tot_cur_bal"] / df["annual_inc"]


# In[66]:


# Calculating High inquiry flag
df["high_inq_flag"] = (df["inq_last_6mths"] >= 3).astype(int)


# In[67]:


# Creating bins for mths_since_last_delinq
col = "mths_since_last_delinq"

# Create bins (only for non-missing values)
df["mths_since_last_delinq_bin"] = pd.cut(
    df[col],
    bins=[-1, 12, 36, 60, 120, 9999],
    labels=["0-12m", "1-3yr", "3-5yr", "5-10yr", "10yr+"])

# Add category for NaN
df["mths_since_last_delinq_bin"] = df["mths_since_last_delinq_bin"].cat.add_categories(["No_Delinquency"])

# Assign NaN to "No_Delinquency"
df.loc[df[col].isna(), "mths_since_last_delinq_bin"] = "No_Delinquency"


# In[68]:


#-------------------------------#
### Using CatBoost ML Model ###
#-------------------------------#


# In[69]:


# 1. Selecting target + features
# ------------------------------------------
target = "loan_default_flag"

features = [c for c in df.columns if c != target]


# In[70]:


# 2. Identifying categorical columns
#    CatBoost handles categorical encodings internally
# ------------------------------------------
cat_features = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'verification_status', 'purpose', 'addr_state', 'application_type', 'zip_updated', 'dti_bucket',
    'delinq_2yrs_bin', 'mths_since_last_delinq_bin']


# In[71]:


#Ensuring proper data type
for c in cat_features:
    df[c] = df[c].astype(str)


# In[72]:


# 3. Train/Val/Test split 
#-----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    df[features], df[target],
    test_size=0.3,
    random_state=42,
    stratify=df[target])

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp)


# In[73]:


# 4. Calculate class weights
# ------------------------------

w0 = (y_train == 0).sum()
w1 = (y_train == 1).sum()

scale = w0 / w1  # weight for minority class 1


# In[68]:


# 5. Train CatBoost with class weights
# --------------------------------------

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    eval_metric='AUC',
    loss_function='Logloss',
    class_weights=[1, scale],  # Correct format
    random_seed=42,)

model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features,
    use_best_model=True, verbose=200)


# In[69]:


# 6. Evaluation
# -------------------------

# Validation scores
y_val_prob = model.predict_proba(X_val)[:, 1]

# Calculating Threshold using Youden’s J
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
youden = tpr - fpr
best_threshold = thresholds[np.argmax(youden)]

print("Best threshold:", best_threshold)

y_val_pred = (y_val_prob >= best_threshold).astype(int)
print("Validation AUC:", roc_auc_score(y_val, y_val_prob))


# In[70]:


# Test scores
y_test_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)
print("Test AUC:", roc_auc_score(y_test, y_test_prob))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))


# In[71]:


# Saving Model
model.save_model("model.cbm")


# In[72]:


# Loading the model
model = CatBoostClassifier()
model.load_model("model.cbm")


# In[73]:


# Checking probability distributions
plt.hist(y_test_prob[y_test==0], bins=50, alpha=0.6, label='Class 0')
plt.hist(y_test_prob[y_test==1], bins=50, alpha=0.6, label='Class 1')
plt.legend()
plt.show()


# In[74]:


#-------------------------------#
### Model Debugging Checks ###
#-------------------------------#


# In[75]:


# 1. Leakage Checks
#----------------------


# In[74]:


# Target Leakage From Future Information
leakage_keywords = ["pymnt", "recover", "out_prncp", "total", "last_pymnt", "next_pymnt", 
    "collection", "settlement"]

future_cols = [col for col in df.columns if any(k in col.lower() for k in leakage_keywords)]

future_cols


# In[75]:


# Leakage from Target Encoded Columns
[col for col in df.columns if "default" in col.lower() or "risk" in col.lower()]


# In[76]:


# Leakage From Date Columns
[col for col in df.columns if "date" in col.lower() or "_d" in col.lower()]


# In[77]:


# Dropping the below columns as these could cause leakage
df = df.drop(columns=["mths_since_last_delinq", "mths_since_last_delinq_bin"])


# In[78]:


# 2. Imbalance Check
#------------------------


# In[79]:


# 2a. CHECK BASIC IMBALANCE
# -------------------------
target = "loan_default_flag"

print("Class distribution:")
print(df[target].value_counts())
print("\nPercentage distribution:")
print(df[target].value_counts(normalize=True) * 100)


# In[82]:


# This is highly imbalance. It causes very poor precision for the minority class. 


# In[83]:


# Plotting imbalance
plt.figure(figsize=(6,4))
df[target].value_counts().plot(kind='bar')
plt.title("Target Imbalance")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# In[80]:


# 2b. TRAIN/TEST SPLIT
# -------------------------
features = [c for c in df.columns if c != target]

X_train, X_test, y_train, y_test = train_test_split(df[features],
    df[target],
    test_size=0.2,
    random_state=42,
    stratify=df[target])


# In[81]:


# 2c. CLASS WEIGHTS CALCULATION
# --------------------------------
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()

class_weights = {0: num_pos/num_neg, 1: num_neg/num_pos}
print("\nClass Weights Used:", class_weights)


# In[85]:


# 2.1a. BASELINE MODEL
# -------------------------
cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

base_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    class_weights=class_weights,
    verbose=0)

base_model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_features)


# In[86]:


# 2.1b. Evaluation 
# --------------------
pred_prob = base_model.predict_proba(X_test)[:,1]
print("\nBaseline AUC:", roc_auc_score(y_test, pred_prob))

threshold = 0.2
pred_class = (pred_prob >= threshold).astype(int)
print("\nClassification Report (Base Model):\n")
print(classification_report(y_test, pred_class))


# In[87]:


# This model used class weights instead of modifying the dataset. Here the AUC is a solid ~0.703. 
# It means CatBoost + class weights is learning useful patterns despite severe imbalance.


# In[88]:


# Saving the model
base_model.save_model("base_model.cbm")


# In[89]:


# Loading the model
base_model = CatBoostClassifier()
base_model.load_model("base_model.cbm")


# In[90]:


# 2.2a. OVERSAMPLING TEST
# -------------------------
df_train = pd.concat([X_train, y_train], axis=1)

minority = df_train[df_train[target] == 1]
majority = df_train[df_train[target] == 0]


# In[91]:


minority_oversampled = resample(minority, replace=True,
    n_samples=len(majority), random_state=42)

df_oversampled = pd.concat([majority, minority_oversampled])
X_os = df_oversampled[features]
y_os = df_oversampled[target]


# In[92]:


# 2.2b. OVERSAMPLING Model Fit
# -------------------------------
os_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=0)

os_model.fit(X_os, y_os, eval_set=(X_test, y_test),
    cat_features=cat_features)


# In[93]:


# 2.2c. Evaluation
# -------------------------
pred_prob_os = os_model.predict_proba(X_test)[:,1]
print("\nOversampled AUC:", roc_auc_score(y_test, pred_prob_os))

pred_os = (pred_prob_os >= threshold).astype(int)
print("\nClassification Report (Oversampled Model):\n")
print(classification_report(y_test, pred_os))


# In[94]:


# This is SMOTE or random oversampling. Oversampling reduced AUC (0.689). 
# This can harm tree-based models like CatBoost


# In[95]:


# Saving the model
os_model.save_model("os_model.cbm")


# In[96]:


# Loading the model
os_model = CatBoostClassifier()
os_model.load_model("os_model.cbm")


# In[97]:


# 2.3a. UNDERSAMPLING TEST
# -------------------------
minority = df_train[df_train[target] == 1]
majority_downsampled = resample(majority, replace=False,
    n_samples=len(minority), random_state=42)


# In[98]:


df_undersampled = pd.concat([majority_downsampled, minority])
X_us = df_undersampled[features]
y_us = df_undersampled[target]


# In[99]:


# 2.3b. UNDERSAMPLING Model Fit
# -------------------------------
us_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=0)

us_model.fit(X_us, y_us, eval_set=(X_test, y_test),
    cat_features=cat_features)


# In[100]:


# 2.3c. Evaluation
# --------------------
pred_prob_us = us_model.predict_proba(X_test)[:,1]
print("\nUndersampled AUC:", roc_auc_score(y_test, pred_prob_us))

pred_us = (pred_prob_us >= threshold).astype(int)
print("\nClassification Report (Undersampled Model):\n")
print(classification_report(y_test, pred_us))


# In[101]:


# Undersampling produced slightly better AUC (0.7056) than baseline. This means:
# Some majority samples are uninformative and the model benefits from focusing on "hard negatives".
# But undersampling reduces dataset size. This produces the risk of losing information.


# In[102]:


# Saving the model
us_model.save_model("us_model.cbm")


# In[103]:


# Loading the model
us_model = CatBoostClassifier()
us_model.load_model("us_model.cbm")


# In[104]:


# 2.4. COMPARING ALL MODELS
# -------------------------
print("\n=======================")
print("AUC SUMMARY")
print("=======================")
print("Baseline (Weighted):", roc_auc_score(y_test, pred_prob))
print("Oversampled:", roc_auc_score(y_test, pred_prob_os))
print("Undersampled:", roc_auc_score(y_test, pred_prob_us))


# In[105]:


# Class-weighted CatBoost model is currently the best clean-performing model so far with AUC 0.703. 
# This will be our final model for debugging. 
# The imbalance is NOT the main reason for the low recall or precision. 
# Our model performs as expected for credit-risk data:
# 1) High recall for minority class (defaults)
# 2) Lower precision
# 3) Strong separation ability (AUC > 0.70)


# In[106]:


#------------------------------#
### Feature Importance ###
#------------------------------#


# In[107]:


# PredictionValueChange Importance
feature_names = model.feature_names_

importances = model.get_feature_importance(type="PredictionValuesChange")

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances}).sort_values("importance", ascending=False)

print(fi_df.head(20))


# In[108]:


# Plot
plt.figure(figsize=(8, 12))
plt.barh(fi_df["feature"].head(25)[::-1], fi_df["importance"].head(25)[::-1])
plt.title("Top 25 Feature Importance (PredictionValueChange)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(r"C:\Users\91988\OneDrive\Desktop\Data Analyst Portfolio\Python Project\Plots\feature_importance.png",
            dpi=300, bbox_inches='tight')
plt.show()


# In[109]:


# a)From the above Feature importance plot and values we can see that member_id and id are
#   showing importance (5.140, 4.600) almost as strong as annual_inc. This doesn't make sense 
#   as any identifier should have zero relationship with loan default risk. This indicated leakage
#   or pattern memorization. Hence we need to drop both member_id and id columns. 
# b)On the other hand we have many traditional credit features showing high importance such as 
#   Int_rate, annual_inc, sub_grade, grade, dti etc


# In[82]:


# Dropping the identifier columns from our dataset
df = df.drop(columns=["member_id", "id"])


# In[111]:


#-----------------------------------------------------------------------#
### Re-running the CatBoost model after making the necessary changes
#-----------------------------------------------------------------------#


# In[83]:


# 1. Selecting target + features
# ------------------------------------------
target = "loan_default_flag"

# Excluding the earliest_cr_year from the model to avoid confusion
features = [c for c in df.columns if c not in [target, "earliest_cr_year"]]


# In[84]:


# 2. Identifying categorical columns
#    CatBoost handles categorical encodings internally
# ------------------------------------------------------
cat_features = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'verification_status', 'purpose', 'addr_state', 'application_type', 'zip_updated', 'dti_bucket',
    'delinq_2yrs_bin']


# In[85]:


#Ensuring proper data type
for c in cat_features:
    df[c] = df[c].astype(str)


# In[86]:


# 3. Train/Val/Test split 
#-----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(df[features], df[target],
    test_size=0.3,random_state=42, stratify=df[target])

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
    test_size=0.5,random_state=42,stratify=y_temp)


# In[87]:


# 4. Calculate class weights
# ------------------------------

w0 = (y_train == 0).sum()
w1 = (y_train == 1).sum()

scale = w0 / w1  # weight for minority class 1


# In[88]:


cat_indices = [X_train.columns.get_loc(col) for col in cat_features]


# In[118]:


# 5. Train CatBoost with class weights
# ---------------------------------------
model_1 = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    eval_metric='AUC',
    loss_function='Logloss',
    class_weights=[1, scale],  # Correct format
    random_seed=42,)

model_1.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_indices,
    use_best_model=True, verbose=200)


# In[119]:


# 6. Evaluation
# -------------------------

# Validation scores
y_val_prob = model_1.predict_proba(X_val)[:, 1]

# Calculating Threshold using Youden’s J
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
youden = tpr - fpr
best_threshold = thresholds[np.argmax(youden)]

print("Best threshold:", best_threshold)

y_val_pred = (y_val_prob >= best_threshold).astype(int)
print("Validation AUC:", roc_auc_score(y_val, y_val_prob))


# In[120]:


# Test scores
y_test_prob = model_1.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)
print("Test AUC:", roc_auc_score(y_test, y_test_prob))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))


# In[121]:


# The above model produces AUC = 0.7066 which means that the model has moderate discriminatory power
# (better than random, which is 0.50).
# Here the precision for Class 0 (Non-default) is 0.94 which means when the model predicts 0, it's correct
# 94% of the time. The recall is 0.63 which means it misses many 0's. For class 1 (Default) the precision
# is 0.19 which means when the model predicts default it is correct only 19% of the time. The recall here is 
# 0.67, meaning it catches 67% of actual defaults. This pattern is very typical for credit models.
# High recall but low precision for defaults means that the model finds most risky loans but it also adds
# a lot of false alarms.This is expected, because defaults are rare (imbalanced dataset).


# In[122]:


#Saving Model
model_1.save_model("model_1.cbm")


# In[89]:


# Loading model
model_1 = CatBoostClassifier()
model_1.load_model("model_1.cbm")


# In[90]:


# Generating probabilities again
y_prob_raw = model_1.predict_proba(X_test)[:, 1]


# In[124]:


#-----------------------------------------------#
### Feature Importance for our Final Model ###
#-----------------------------------------------#


# In[125]:


# PredictionValueChange Importance
importances = model_1.get_feature_importance(type="PredictionValuesChange")
feature_names = X_train.columns

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(fi_df)


# In[144]:


# Exporting Feature Importance to csv file for visualization in Power Bi
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model_1.get_feature_importance()})

# Export summary file
feature_importance.to_csv("Feature_Imp_Bi.csv", index=False)


# In[126]:


# Plot
plt.figure(figsize=(8, 12))
plt.barh(fi_df["feature"].head(25)[::-1], fi_df["importance"].head(25)[::-1])
plt.title("Top 25 Feature Importance (PredictionValueChange)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(r"C:\Users\91988\OneDrive\Desktop\Data Analyst Portfolio\Python Project\Plots\feature_importance2.png",
            dpi=300, bbox_inches='tight')
plt.show()


# In[127]:


#------------------------------------------------#
### Calculating SHAP values for the test set ###
#------------------------------------------------#


# In[128]:


# Converting test set into a CatBoost Pool
train_pool = Pool(data=X_train,label=y_train,cat_features=cat_indices)


# In[129]:


# Computing SHAP Values
shap_values = model_1.get_feature_importance(data=train_pool,type="ShapValues")


# In[130]:


# Removing the last column and extracting the actual SHAP values only (n_samples, n_features + 1)
shap_values_for_plot = shap_values[:, :-1]


# In[131]:


# Initialize SHAP explainer for CatBoost
explainer = shap.TreeExplainer(model_1)

# Summary plot
shap.summary_plot(shap_values_for_plot, X_train, show=False)

plt.savefig(r"C:\Users\91988\OneDrive\Desktop\Data Analyst Portfolio\Python Project\Plots\shap_summary.png",
            dpi=300, bbox_inches='tight')


# In[132]:


# SHAP Bar Plot (global average impact)
shap.summary_plot(shap_values_for_plot, X_train, show=False, plot_type="bar")

plt.savefig(r"C:\Users\91988\OneDrive\Desktop\Data Analyst Portfolio\Python Project\Plots\shap_barplot.png",
            dpi=300, bbox_inches='tight')


# In[133]:


#SHAP Force Plot for a Single Prediction
i = 10  # example row
shap.force_plot(explainer.expected_value, shap_values_for_plot[i],
    X_train.iloc[i], matplotlib=True)


# In[134]:


# SHAP Dependence Plot
shap.dependence_plot("dti", shap_values_for_plot, X_train)


# In[140]:


# Exporting SHAP values to a csv file for visualization purposes in Power Bi

test_pool = Pool(data=X_test,label=y_test,cat_features=cat_indices)

shap_values_test = model_1.get_feature_importance(data=test_pool,type="ShapValues")

# Removing the last column and extracting the actual SHAP values only (n_samples, n_features + 1)
shap_values_for_Bi = shap_values_test[:, :-1]

# Computing mean absolute shap importance
shap_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": np.abs(shap_values_for_Bi).mean(axis=0)})

# Export summary file
shap_importance.to_csv("shap_summary_Bi.csv", index=False)


# In[399]:


# Checking Calibration
y_test_prob = model_1.predict_proba(X_test)[:, 1]

prob_true, prob_pred = calibration_curve(y_test,y_test_prob,n_bins=20,
    strategy="quantile")   # recommended for imbalanced data

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], "--")
plt.xlabel("Predicted probability")
plt.ylabel("True probability in each bin")
plt.title("Calibration Curve")
plt.show()


# In[400]:


### Interpretation of Calibration Curve ###
# The above plot shows:
# Orange dashed line: Perfect calibration - If the model predicts 0.30, then 30% of those cases should truly default.
# Blue curve: model’s actual calibration - How well predicted probabilities match real-world frequencies.
# In the above plot we can see that the curve is well below the diagonal. This means that the model is
# overconfident (overestimates probability of default) For example: When the model predicts 0.70 probability of default
# the actual observed probability is only around 0.30.
# Here the curve rises monotonically (good discrimination) even though probabilities are miscalibrated i.e;
# higher predicted probabilities → higher true default rates. So the model still ranks customers correctly (good AUC).
# This is why our final model has a decent AUC (~0.71) even though calibration is poor. The model is good at 
# ranking risky customers (AUC) but it is not good at giving the true probability of default. It can be used for
# risk ranking, loan approval prioritization and segmentation but raw predicted probabilities shouldn't be used for
# expected loss calculation, pricing and provisioning unless calibrated.


# In[401]:


#--------------------------------#
### Calibration of the model ###
#--------------------------------#


# In[94]:


# Platt Scaling (Logistic Calibration)
# Fitting the Calibrated Model
calibrated_model = CalibratedClassifierCV(model_1, method='sigmoid', cv='prefit')
calibrated_model.fit(X_val, y_val)


# In[403]:


#Plotting the Sigmoid calibration curve
y_test_prob_cal = calibrated_model.predict_proba(X_test)[:,1]
prob_true, prob_pred = calibration_curve(y_test, y_test_prob_cal, n_bins=20)

plt.plot(prob_pred, prob_true, marker='o', label="Calibrated Model")
plt.plot([0,1],[0,1],"--", label="Perfect Calibration")
plt.title("Calibration Curve (After Sigmoid Calibration)")
plt.legend()
plt.show()


# In[405]:


# Comparing before vs. after calibration in one plot

# BEFORE calibration
y_test_prob = model_1.predict_proba(X_test)[:, 1]
prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_test_prob, n_bins=20)

# AFTER calibration
y_test_calib_prob = calibrated_model.predict_proba(X_test)[:, 1]
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_test_calib_prob, n_bins=20)

plt.figure(figsize=(7,5))

plt.plot(prob_pred_raw, prob_true_raw, marker='.', label='Before Calibration')
plt.plot(prob_pred_cal, prob_true_cal, marker='o', label='After Calibration')
plt.plot([0,1], [0,1], "--", label='Perfect calibration')

plt.title("Calibration Curve Comparison")
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.legend()
plt.show()


# In[97]:


# Saving Calibrated Model
joblib.dump(calibrated_model, "calibrated_model.pkl")


# In[91]:


# Loading the calibrated model
calibrated_model = joblib.load("calibrated_model.pkl")


# In[92]:


# Generating probabilities again
y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]


# In[ ]:


#-------------------------------------#
### Computing Calibration Metrics ###
#-------------------------------------#


# In[93]:


# AUC
auc = roc_auc_score(y_test, y_prob_calibrated)


# In[94]:


# Brier Score
brier = brier_score_loss(y_test, y_prob_calibrated)


# In[95]:


# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    # Step 1: get calibration curve data
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Step 2: Recreate the bin edges actually used
    bins = np.linspace(0, 1, n_bins + 1)

    # Step 3: Assign each sample to a bin
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0
    N = len(y_prob)

    # Step 4: Loop through each bin that actually exists in prob_true/prob_pred
    for i in range(len(prob_true)):
        # mask for all samples in this bin
        mask = bin_ids == i
        bin_size = mask.sum()

        if bin_size > 0:
            ece += (bin_size / N) * abs(prob_true[i] - prob_pred[i])

    return ece


# In[96]:


ece = expected_calibration_error(y_test, y_prob_calibrated)
print(ece)


# In[97]:


# Checking defaults per bin to make sure our ece is valid
def defaults_per_bin(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1  # bin indices from 0 to n_bins-1

    results = []

    for i in range(n_bins):
        in_bin = (bin_ids == i)
        defaults = y_true[in_bin].sum()          # defaults in this bin
        total_obs = in_bin.sum()                 # total observations
        avg_pd = y_prob[in_bin].mean() if total_obs > 0 else None

        results.append({
            "bin": i + 1,
            "range": f"{bins[i]:.2f}–{bins[i+1]:.2f}",
            "obs_in_bin": total_obs,
            "defaults_in_bin": int(defaults),
            "avg_pred_pd": avg_pd
        })

    return pd.DataFrame(results)

df_bin_stats = defaults_per_bin(y_test, y_prob_calibrated, n_bins=10)
df_bin_stats


# In[98]:


# KS Statistic
fpr, tpr, _ = roc_curve(y_test, y_prob_calibrated)
ks = max(tpr - fpr)


# In[99]:


# Avg PD & Actual Default Rate
avg_pd = y_prob_calibrated.mean()
actual_default_rate = y_test.mean()


# In[102]:


# Creating Calibrartion data frame
Calibration_metrics_df = pd.DataFrame({
    "Metric": ["AUC", "Brier Score", "ECE", "KS", "Average PD", "Actual Default Rate"],
     "Value": [auc, brier, ece, ks, avg_pd, actual_default_rate]})

Calibration_metrics_df.to_csv("calibration_full_metrics.csv", index=False)


# In[100]:


calibration_df["AUC"] = auc
calibration_df["Brier_Score"] = brier
calibration_df["ECE"] = ece
calibration_df["KS"] = ks
calibration_df["Gini"] = gini_score
calibration_df["Average_PD"] = avg_pd
calibration_df["Actual_Default_Rate"] = actual_default_rate

calibration_df.to_csv("calibration_full_metrics.csv", index=False)


# In[406]:


#--------------------------------#
### Checking Predicted Values ###
#--------------------------------#


# In[407]:


# Getting predicted values from our trained data

# Getting predicted classes using your selected threshold
y_test_pred = (y_test_prob >= best_threshold).astype(int)
y_test_pred[:10]


# In[408]:


# Creating a comparison table of real vs predicted
pred_df = pd.DataFrame({
    "actual": y_test,
    "predicted_label": y_test_pred,
    "predicted_probability": y_test_prob
})
pred_df.head()


# In[409]:


# This model is optimized for recall of defaults, because in credit risk the cost of missing 
# a default is far higher than the cost of flagging a safe borrower. As a result, the model produces
# some false positives—borrowers predicted as risky even though they repay—but it successfully catches
# the majority of true defaults. This is reflected in the predictions: cases around 0.58–0.65 are borderline
# probabilities where false positives are expected and acceptable depending on business strategy. 
# Calibration analysis also indicated that these mid-range probabilities need careful interpretation.


# In[410]:


# Saving predictions to CSV
pred_df.to_csv("model_predictions.csv", index=False)


# In[411]:


# Seeing which rows were missing & how our Model predicted them

# Inspecting rows that originally had missing values:
rows_with_missing = df[df.isnull().any(axis=1)]
rows_with_missing

# Seeing how the model predicted for those rows after preprocessing
# Align rows with missing values to X_test rows
missing_index = rows_with_missing.index.intersection(X_test.index)

pd.DataFrame({
    "actual": y_test.loc[missing_index],
    "pred_probability": model_1.predict_proba(X_test.loc[missing_index])[:,1]
}).head()


# In[412]:


# The threshold we have chosen is 0.49~. If probability is < 0.49 then model predicts (non-default-0). 
# If probability is > 0.49 then model predicts (default-1). The above table shows 2 incorrect and 3 correct results.
# Most incorrect ones are false positives. This again matches the global model behaviour:
# High recall → catches many true defaults
# Low precision → many false alarms for default
# This is exactly what our classification report showed.


# In[413]:


# Computing threshold for calibrated model

# Getting calibrated probabilities
pred_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

# Computing ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, pred_proba_calibrated)

best_threshold = thresholds[(tpr - fpr).argmax()]
print(best_threshold)


# In[414]:


# Generating predictions for the whole dataset and adding it as a column to our dataset 

# Raw probabilities from CatBoost
df["pred_proba_raw"] = model_1.predict_proba(df[features])[:, 1]

# Calibrated probabilities from sigmoid calibrated model
df["pred_proba_calibrated"] = calibrated_model.predict_proba(df[features])[:, 1]
                        
# Final label using best threshold
threshold = 0.105894773
df["predicted_label"] = (df["pred_proba_calibrated"] >= threshold).astype(int)


# In[422]:


# Adding "bucket" to dti_bucket before exporting dataset to csv file so excel doesn't misinterpret as date
df["dti_bucket"] = df["dti_bucket"].apply(lambda x: f"'{x}")


# In[424]:


# Exporting the cleaned data to csv file
df.to_csv("lending_club_with_predictions.csv", index=False)

