import pandas as pd
import re

#  importing excel data as dataframe 

recipes = pd.read_excel(r"C:\Users\KIIT\Desktop\newrecipe.xlsx")

print (recipes)

print (recipes.head(10)) #for cross-checking

# Data Preprocessing

print (recipes.info()) 

print (recipes.duplicated().sum()) #the output is zero showing that the data has no duplicated values

print (recipes.isnull().sum()) #the output is zero showing that there are no missing values in the dataset

 #converting to lower case to make machine learning model
recipes['Title'] = recipes['Title'].str.lower()                  
recipes['Ingredients'] = recipes['Ingredients'].str.lower()  
recipes['Instructions'] = recipes['Instructions'].str.lower()  
recipes['Image_Name'] = recipes['Image_Name'].str.lower()  
recipes['Cleaned_Ingredients'] = recipes['Cleaned_Ingredients'].str.lower()  

print (recipes)

#  #removing punctuations
pattern = r'[^\w\s]' #Matches any character except word characters (alphanumeric) and whitespace

recipes['Title'] = recipes['Title'].apply(lambda x: re.sub(pattern, '', x))         
recipes['Ingredients'] = recipes['Ingredients'].apply(lambda x: re.sub(pattern, '', x))
recipes['Instructions'] = recipes['Instructions'].apply(lambda x: re.sub(pattern, '', x))
recipes['Image_Name'] = recipes['Image_Name'].apply(lambda x: re.sub(pattern, '', x))
recipes['Cleaned_Ingredients'] = recipes['Cleaned_Ingredients'].apply(lambda x: re.sub(pattern, '', x))

print (recipes)
print (recipes.info())
print (recipes.describe())

#one-hot encoding
from sklearn.preprocessing import OneHotEncoder

cleaned_ingredients_features = recipes[['Cleaned_Ingredients']]

encoder = OneHotEncoder( handle_unknown='ignore')

encoder.fit(cleaned_ingredients_features)

cleaned_ingredients_encoded = encoder.transform(cleaned_ingredients_features)

print(cleaned_ingredients_encoded)

ingredients_features = recipes[['Ingredients']]  

encoder = OneHotEncoder( handle_unknown='ignore')

encoder.fit(ingredients_features)

ingredients_encoded = encoder.transform(cleaned_ingredients_features)

print(ingredients_encoded)

title_features = recipes[['Title']]

encoder = OneHotEncoder( handle_unknown='ignore')

encoder.fit(title_features)

title_encoded = encoder.transform(title_features)

print(title_encoded)

Instructions_features = recipes[['Instructions']]

encoder = OneHotEncoder( handle_unknown='ignore')

encoder.fit(Instructions_features)

Instructions_encoded = encoder.transform(Instructions_features)

print(Instructions_encoded)

Image_Name_features = recipes[['Image_Name']]

encoder = OneHotEncoder( handle_unknown='ignore')

encoder.fit(Image_Name_features)

Image_Name_encoded = encoder.transform(Image_Name_features)

print(Image_Name_encoded)

