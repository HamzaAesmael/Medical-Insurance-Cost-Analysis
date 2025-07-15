import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import requests as re
import os
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

"""==== Task 1 : Import the dataset ===="""

# Download the dataset
def download (url , filename ) : 
    response = re.get(url)
    if response.status_code == 200 :
        with open(filename, "wb") as f:
                f.write(response.content)

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNsetwork-DA0101EN-Coursera/medical_insurance_dataset.csv"
file_name = "Medical_insurance.csv"

# Download the file if it doesn't exist
if not os.path.exists(file_name):
    if download (filepath , "Medical_insurance.csv") :
         print ('File downloaded successfully')
    else:
        print("Failed to download file")
        exit()
    
try :   
    df = pd.read_csv(file_name)
    print("The first 5 rows of the dataframe")
    print(df.head(10))
except FileNotFoundError:
    print(f"Error: The file {file_name} was not found in {os.getcwd()}")
except Exception as e:
    print(f"An error occurred: {e}")
print("\n📊 Basic Statistics:")
print(df.describe())

print("\n🔍 Missing Values:")
print(df.isnull().sum())
print (df.info ())

"""==== Task 2 : Data Wrangling ===="""
df['smoker_numeric']=df['smoker'].apply(lambda x:1 if x == 'yes' else 0)
df['region_numeric']=df['region'].apply(lambda x:1 if x == 'northwest' else 2 if x == 'northeast' else 3 if x == 'southwest' else 4 if x == 'southeast' else np.nan)
df['sex_numeric']=df['sex'].apply(lambda x:1 if x == 'male' else 2)


"""==== Task 3 : Exploratory Data Analysis (EDA) ===="""
#Implement the regression plot for `charges` with respect to `bmi`. 
plt.figure(figsize=(8,6))
sns.regplot ( x ='bmi' , y = 'charges' , data=df ,
    scatter_kws={'alpha':0.4, 'color':'blue'},  
    line_kws={'color':'red', 'linewidth':2}     
            )
plt.title('Medical Insurance Charges vs BMI', fontsize=16)
plt.xlabel('BMI (Body Mass Index)', fontsize=12)
plt.ylabel('Insurance Charges ($)', fontsize=12)
plt.tight_layout()
plt.show()

#Implement the box plot for charges with respect to smoker.
sns.set(style="whitegrid")
plt.figure(figsize=(8,6))
sns.boxplot ( x ='smoker' , y = 'charges' , data=df ,
    palette={"yes": "lightcoral", "no": "lightblue"}  # Colors for smokers/non-smokers
            )
plt.title('Insurance Charges: Smokers vs. Non-Smokers', fontsize=16)
plt.xlabel('smoker', fontsize=12)
plt.ylabel('Insurance Charges ($)', fontsize=12)
plt.xticks([0, 1], ['Smoker', 'Non-Smoker'])  # Clearer labels
plt.tight_layout()
plt.show()
print("Median charges:")
print(df.groupby('smoker')['charges'].median())

# Age vs Charges
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='charges', hue='smoker', data=df, alpha=0.6)
plt.title("Insurance Charges by Age (Smokers vs Non-Smokers)")
plt.show()

# Distribution of Charges
plt.figure(figsize=(8, 6))
sns.histplot(df['charges'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Insurance Charges")
plt.show()

#Print the correlation matrix for the dataset.
corr_matrix = df.corr(numeric_only=True)
print("Correlation Matrix:")
print(corr_matrix)
print(df['children'].value_counts())

"""===Task 4 :  Model Developmen ===="""

print('linear regression model : Charges vs Smoker')
lm = LinearRegression ()
x = df[['smoker_numeric']]
y = df[['charges']]
lm.fit(x,y)
print('R2 score:',lm.score(x, y))
print('Intercept : ',lm.intercept_ )
print('Coefficient :',lm.coef_)

print(' Multiple Regression ')
lm_m=LinearRegression()
x_m = df[["age", "bmi", "children", "smoker_numeric", "region_numeric","sex_numeric"]].astype(float)
x_m = x_m.dropna()
y_m = df[["charges"]].astype(float)
lm_m.fit(x_m , y_m)
print('R2 score:',lm_m.score(x_m , y_m))
print('Intercept : ',lm_m.intercept_ )
print('Coefficient :',lm_m.coef_)

# Predict the charges value uses StandardScaler(), PolynomialFeatures() and LinearRegression()
input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures()),('model',LinearRegression())]
pipe = Pipeline(input)
pipe.fit(x_m  ,y_m)
y_pred = pipe.predict(x_m)
print(f"R² Score: {r2_score(y_m, y_pred):.4f}")


"""==== Task 5 : Model Refinement ===="""
#Split the data into training and testing subsets, assuming that 20% of the data will be reserved for testing.
x_train , x_test , y_train , y_test = train_test_split (x_m,y_m,test_size=0.2, random_state=1)
RigeModel = Ridge (alpha=0.1)
RigeModel.fit(x_train,y_train)
y_hat = RigeModel.predict(x_test)
print('R² Score for test data :', r2_score(y_test,y_hat))

#Apply polynomial transformation to the training parameters
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.transform(x_test)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)
print('R² Score for test subset :' ,r2_score(y_test,yhat))