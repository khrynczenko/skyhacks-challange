import pandas as pd

df = pd.read_csv("./labels.csv")

df_train = df.loc[df['task2_class'] != 'validation']
df_valid = df.loc[df['task2_class'] == 'validation']

task1_train = df_train.drop(labels=['standard', 'task2_class', 'tech_cond'], axis=1)
task1_valid = df_valid.drop(labels=['standard', 'task2_class', 'tech_cond'], axis=1)

task2 = df_train[['filename', 'task2_class']]

task3_train = df_train[['filename', 'standard', 'tech_cond']]
task3_valid = df_valid[['filename', 'standard', 'tech_cond']]

task1_train.to_csv("./task1_train.csv", index=False)
task1_valid.to_csv("./task1_valid.csv", index=False)
task2.to_csv("./task2.csv", index=False)
task3_train.to_csv("./task3_train.csv", index=False)
task3_valid.to_csv("./task3_valid.csv", index=False)
