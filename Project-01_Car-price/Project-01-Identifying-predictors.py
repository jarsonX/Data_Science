
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline 

#--------------------------------------------------------------------------------------------------Load dataset
df = pd.read_csv('automobileEDA.csv')

#-------------------------------------------------------------------------------------------------Check dataset
df.dtypes

#Missing values
missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 

#Fix missing values in stroke column
df.interpolate(inplace=True)  #fix missing in stroke
df.drop(['horsepower-binned'], inplace=True, axis=1)

#--------------------------------------------------------------------------------------------------Correlations
#Quick look
df_corr = df.corr()
df_corr.sort_values(by=['price'])

#Only the following numerical variables should be considered in further analysis (due to moderate or strong 
#positive/negative correlation): length, width, curb-weight, engine-size, horsepower, city-mpg, highway-mpg, 
#wheel-base, bore.

#Detailed Pearson correlation analysis
print('Length')
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("Correlation Coefficient:", pearson_coef, " P-value:", p_value)
print(" ")
#Since the p-value is < 0.001, the correlation between length and price is statistically
#significant, and the linear relationship is moderately strong (~0.691).

sns.regplot(x="length", y="price", data=df)
plt.ylim(0,)

print('Width')
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("Correlation Coefficient:", pearson_coef, " P-value:", p_value)
print(" ")
#Since the p-value is < 0.001, the correlation between width and price is statistically 
#significant, and the linear relationship is quite strong (~0.751).

sns.regplot(x="width", y="price", data=df)
plt.ylim(0,)

print('Curb-weight')
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "Correlation Coefficient:", pearson_coef, " P-value:", p_value)
print(" ")
#Since the p-value is < 0.001, the correlation between curb-weight and price is statistically 
#significant, and the linear relationship is quite strong (~0.834).

sns.regplot(x="curb-weight", y="price", data=df)
plt.ylim(0,)

print('Engine-size')
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("Correlation Coefficient:", pearson_coef, " P-value:", p_value) 
print(" ")
#Since the p-value is < 0.001, the correlation between engine-size and price is statistically 
#significant, and the linear relationship is very strong (~0.872).

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

print('Horsepower')
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("Correlation Coefficient:", pearson_coef, " P-value:", p_value)
print(" ") 
#Since the p-value is < 0.001, the correlation between horsepower and price is statistically 
#significant, and the linear relationship is quite strong (~0.809, close to 1).

sns.regplot(x="horsepower", y="price", data=df)
plt.ylim(0,)

print('City-mpg')
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("Correlation Coefficient:", pearson_coef, " P-value:", p_value) 
print(" ")
#Since the p-value is < 0.001, the correlation between city-mpg and price is statistically 
#significant, and the coefficient of about -0.687 shows that the relationship is negative and
#moderately strong.

sns.regplot(x="city-mpg", y="price", data=df)
plt.ylim(0,)

print('Highway-mpg')
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "Correlation Coefficient:", pearson_coef, " P-value:", p_value)
print(" ") 
#Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically 
#significant, and the coefficient of about -0.705 shows that the relationship is negative and 
#moderately strong.

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

print("Wheel-base")
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("Correlation Coefficient:", pearson_coef, " P-value:", p_value)  
print(" ")
#Since the p-value is < 0.001, the correlation between wheel-base and price is statistically 
#significant, although the linear relationship isn't extremely strong (~0.585).

sns.regplot(x="wheel-base", y="price", data=df)
plt.ylim(0,)

print('Bore')
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("Correlation Coefficient:", pearson_coef, " P-value:", p_value) 
#Since the p-value is < 0.001, the correlation between bore and price is statistically 
#significant, but the linear relationship is only moderate (~0.521)

sns.regplot(x="bore", y="price", data=df)
plt.ylim(0,)

#-----------------------------------------------------------------------------------------Categorical variables
#make, aspiration, num-of-doors, body-style, drive-wheels, engine-location, engine-type, num-of-cylinders, 
#fuel-system

#Exclude variables that are not appropriate for predictors

#Make
sns.boxplot(x="make", y="price", data=df)
#exclude due to overlapping

#Num-of-doors
sns.boxplot(x="num-of-doors", y="price", data=df)
#exclude due to overlapping

#Body-style
sns.boxplot(x="body-style", y="price", data=df)
#exclude due to overlapping

#Drive-wheels
sns.boxplot(x="drive-wheels", y="price", data=df)

#Count per drive-wheels
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

#Average price for drive-wheels
df_group_one = df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one.rename(columns={'price': 'avg price'}, inplace=True)
df_group_one

#Average price for drive-wheels and body-style
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'], as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)
grouped_pivot

#average price for body-style, heatmap to better understand distribution
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

#ANOVA for drive-wheels
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
print( "ANOVA results: F =", f_val, "| P =", p_val) 

#A large F-test score shows a strong correlation and a P-value of almost 0 implies almost certain statistical significance. 

#Check correlations of particular groups:

#ANOVA for drive-wheels / fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F =", f_val, "| P =", p_val )

#ANOVA for drive-wheels / 4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F =", f_val, "| P =", p_val)   

#ANOVA for drive-wheels / 4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F =", f_val, "| P =", p_val)  

#Engine-location
sns.boxplot(x="engine-location", y="price", data=df)

#Count per engine-location
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts

#exclude becasue there are only 3 rears

#Num-of-cylinders
sns.boxplot(x="num-of-cylinders", y="price", data=df)

#Count per num-of-cylinders
engine_loc_counts = df['num-of-cylinders'].value_counts().to_frame()
engine_loc_counts.rename(columns={'num-of-cylinders': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'num-of-cylinders'
engine_loc_counts

#exclude becasue most models fall into four cylinders category

#Fuel-system
sns.boxplot(x="fuel-system", y="price", data=df)
#exclude due to overlapping

#----------------------------------------------------------------------------------------------------Conclusion

#The following should be considered as predictors:

#----Numerical----

#   Length
#   Width
#   Curb-weight
#   Engine-size
#   Horsepower
#   City-mpg
#   Highway-mpg
#   Wheel-base
#   Bore

#----Categorical----

#   Drive-wheels
