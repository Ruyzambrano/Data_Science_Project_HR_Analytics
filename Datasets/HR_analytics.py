import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import csv

def csv_to_dataframe(csv_file):
    return pd.read_csv(csv_file)


def plot_hist():
    # Plotting histograms
    columns_to_plot = ['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction',
                       'JobSatisfaction', 'WorkLifeBalance']


    for column in columns_to_plot:
        plt.figure(figsize=(10, 6))
        plt.hist(full_dataframe[column], bins=10, edgecolor='black')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column}')
        plt.show()

# Read employee_survey_data and check for null values
employee_survey_pd = csv_to_dataframe('Datasets/employee_survey_data.csv')

# Read general_data and check for null values
general_data_pd = csv_to_dataframe('Datasets/general_data.csv')

# Read manager_survey_data and check for null values
manager_survey_pd = csv_to_dataframe('Datasets/manager_survey_data.csv')

# Merge dataframes
full_dataframe = pd.merge(employee_survey_pd, general_data_pd, on='EmployeeID')
full_dataframe = full_dataframe.merge(manager_survey_pd, on='EmployeeID')

# Read in_time and check for null values
in_time = csv_to_dataframe('Datasets/in_time.csv')

# Read out_time and check for null values
out_time = csv_to_dataframe('Datasets/out_time.csv')

# Replace null values in 'TotalWorkingYears' column with the median value
full_dataframe['TotalWorkingYears'].fillna(full_dataframe['TotalWorkingYears'].median().__round__, inplace=True)

# Same but with 'NumCompaniesWorked'
full_dataframe['NumCompaniesWorked'].fillna(full_dataframe['NumCompaniesWorked'].mean().__round__, inplace=True)

# EnvironmentSatisfaction
full_dataframe['EnvironmentSatisfaction'].fillna(full_dataframe['EnvironmentSatisfaction'].mean().__round__(), inplace=True)

# Job satisfaction
full_dataframe['JobSatisfaction'].fillna(full_dataframe['JobSatisfaction'].mean().__round__(), inplace=True)

# WorkLifeBalance
full_dataframe['WorkLifeBalance'].fillna(full_dataframe['WorkLifeBalance'].mean().__round__(), inplace=True)

education_mapping = {1: 'Below College',
                     2: 'College',
                     3: 'Bachelor',
                     4: 'Master',
                     5: 'Doctor',
                     }
environment_satisfaction_mapping = {1: 'Low',
                                    2: 'Medium',
                                    3: 'High',
                                    4: 'Very High',
                                    }


job_involvement_mapping = {1: 'Low',
                           2: 'Medium',
                           3: 'High',
                           4: 'Very High',
                           }

job_satisfaction_mapping = {1: 'Low',
                            2: 'Medium',
                            3: 'High',
                            4: 'Very High'
                            }

performance_rating_mapping = {1: 'Low',
                              2: 'Good',
                              3: 'Excellent',
                              4: 'Outstanding'
                              }

relationship_satisfaction_mapping = {1: 'Low',
                                     2: 'Medium',
                                     3: 'High',
                                     4: 'Very High'
                                     }

work_life_balance_mapping = {1: 'Bad',
                             2: 'Good',
                             3: 'Better',
                             4: 'Best'
                             }

full_dataframe['Education'] = full_dataframe['Education'].map(education_mapping)
full_dataframe['EnvironmentSatisfaction'] = full_dataframe['EnvironmentSatisfaction'].map(environment_satisfaction_mapping)
full_dataframe['JobInvolvement'] = full_dataframe['JobInvolvement'].map(job_involvement_mapping)
full_dataframe['JobSatisfaction'] = full_dataframe['JobSatisfaction'].map(job_satisfaction_mapping)
full_dataframe['PerformanceRating'] = full_dataframe['PerformanceRating'].map(performance_rating_mapping)
# full_dataframe['RelationshipSatisfaction'] = full_dataframe['RelationshipSatisfaction'].map(relationship_satisfaction_mapping)
full_dataframe['WorkLifeBalance'] = full_dataframe['WorkLifeBalance'].map(work_life_balance_mapping)

in_time.columns = in_time.columns.str.replace("Unnamed: 0", "EmployeeID")
out_time.columns = out_time.columns.str.replace("Unnamed: 0", "EmployeeID")

in_time=in_time.replace(np.nan,0)
out_time=out_time.replace(np.nan,0)

in_time.iloc[:,1:] = in_time.iloc[:,1:].apply(pd.to_datetime, errors='coerce')
out_time.iloc[:,1:] = out_time.iloc[:,1:].apply(pd.to_datetime, errors='coerce')

times = pd.concat([in_time,out_time], axis=0)

times = times.set_index('EmployeeID')

times = times.diff(periods=4410)
times = times.iloc[4410:]

cols2drop = []
for col in times.columns:
    if len(times[col].unique()) != 1:
        pass
    else:
        cols2drop.append(col)

times.drop(cols2drop, axis=1, inplace=True)

times['Actual Time']=times.mean(axis=1)

times['hrs']=times['Actual Time']/np.timedelta64(1, 'h')

times.reset_index(inplace=True)

times.drop(times.columns.difference(['EmployeeID','hrs']), axis=1, inplace=True)

full_dataframe = pd.merge(full_dataframe,times,how='left',on='EmployeeID')

full_dataframe.to_csv(r'full_dataset.csv', index=False)

print('Done!')

full_dataframe.info()