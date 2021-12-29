import numpy as np

import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('HospitalCosts.csv')



print(df.describe()[['AGE', 'LOS', 'TOTCHG']])

df['MDCOST'] = df['TOTCHG'] / df['LOS']


# To record the patient statistics, we want to find the age category of people who visited the hospital and has the maximum overhead.


unique, count = np.unique(df["AGE"], return_counts=True) #return count of each unique value in two separate lists
unique_lst = list(zip(unique,count)) #zip noth lists into a single list
print(unique_lst)


df["AGE"].value_counts().sort_index(ascending=False) #sorting the unique values in descending order

# create a new column to calculate cost of stay per day for each patient
df.rename({"TOTCHG":"COST"}, axis=1, inplace=True)
df["SCPD"] = df["COST"] / df["LOS"] #SCPD = staying cost per day


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16,12))
sns.boxplot(data=df, x="AGE", y="COST")
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='AGE', y='COST')
plt.show()

# With this plot, I could see that the group between 3 and 6 years old had the utmost overhead. This is without considering the potiential outliers.
sns.barplot(data=df, x='AGE', y='COST')
plt.show()

df.groupby(df["AGE"]).mean()["COST"].sort_index(ascending=False)

# In order to determine the severity of the diagnosis and treatments I needed to find out the expensive treatments, to do this, I needed to find the specific group that had the max hospitalization and overhead.

df['APRDRG'].unique()

df['APRDRG'].value_counts()

plt.figure(figsize=(12,8),dpi=200)
plt.xticks(rotation=90)
sns.countplot(data=df, x='APRDRG');
plt.show()

plt.figure(figsize=(12,8), dpi=200)
sns.scatterplot(data=df, x='LOS', y='COST', hue='APRDRG', palette='viridis');
plt.show()

df.groupby(['APRDRG']).mean()['COST'].sort_values(ascending=False)

df.groupby(['APRDRG']).mean()['LOS'].sort_values(ascending=False)

df['TOTCOST'] = df['APRDRG'] * df['LOS']
df.head()

df.sort_values('TOTCOST', ascending=False)
# What I discovered was: That despite that the 911 treatment group had a high overhead, and that the 640 group had the high hospitalization, I concluded that the 602 group had the highest value of both the overhead and the hospitalization.
# To ensure that there was no malpractice, I determined that I needed to analyze whether or not the race of the patient was related to the hospitalization costs or not.

df.groupby(['RACE']).mean()['COST'].sort_values(ascending=False)

plt.figure(figsize=(12,8),dpi=100)
sns.boxplot(data=df, x='RACE', y='COST')
plt.show()

sns.displot(data=df, x='COST', kde=True)
plt.show()

df['COST'].describe()

df.groupby(['RACE']).describe()['COST']

def detecta_outlier(x, limite = 1.5):
    dq = np.percentile(x, 75) - np.percentile(x, 25)
    limite_inf = np.maximum(np.percentile(x, 25) - limite * dq, np.min(x))
    limite_sup = np.minimum(np.percentile(x, 75) + limite * dq, np.max(x))
    print("Limite inferior:", limite_inf)
    print("Limite superior:", limite_sup)
    return np.where((x < limite_inf) | (x > limite_sup), 1, 0)

outliers = detecta_outlier(df.COST)
# These results led me to conclude that: despite the race 2 mean being below the superior limit, I believe it shoul've been investigated because the value of the mean TOTCHG for this group was so much higher than that of the others.
#In order to properly utilize the costs here, I needed to analyze the severity of these hospital costs by looking at the age and gender for proper allocation of resources.

plt.figure(figsize=(12,8),dpi=100)
sns.boxplot(data=df, x='FEMALE', y='COST')
plt.show()

df.groupby(['FEMALE']).mean()['COST']

plt.figure(figsize=(12,8),dpi=100)
sns.barplot(data=df, x='AGE', y='COST', hue='FEMALE')
plt.show()

plt.figure(figsize=(12,8),dpi=100)
sns.countplot(data=df, x='AGE',hue='FEMALE')
plt.show()

#It's possible to see here, that over the age of 12, you can see that the number of female patients is higher than male patients. There is also a lack of female patients to been seen between the ages of 6 and 9.
# I was also able to conclude that the costs with male patients is larger than that of female patients, that inlcudes almost all the ages, except that of the ages of 5, 11 to 13 and > 17.
# And since the length of stay is the most crucial factor for inpatients, I then knew the task at hand was to find out if the length of stay was something that could be predicted from just knowing the age, gender, and race of the patients.

sns.regplot(data=df, x='AGE', y='LOS')
plt.show()

sns.regplot(data=df, x='FEMALE', y='LOS')
plt.show()

sns.regplot(data=df, x='RACE', y='LOS')
plt.show()

# As you can see, the regression line is almost parallel to x axis, this indicated that there was no correlations between the IVs and the DVs.
#To perform a complete analysis, I needed to find the specific variable that affected the hospital costs.

sns.regplot(data=df, x='LOS', y='COST')
plt.show()

sns.regplot(data=df, x='APRDRG', y='COST')
plt.show()

sns.regplot(data=df, x='AGE', y='COST')
plt.show()

sns.regplot(data=df, x='RACE', y='COST')
plt.show()

# With these visualizations, I was able to conclude that the LOS variable had the highest positive correlation, so I was able to conclude that this was the variable that mainly affected the cost of hospitalization.

