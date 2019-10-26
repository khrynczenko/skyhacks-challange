import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\luktu\Desktop\skyhacks-challange\data\labels.csv")

df["task2_class"] = df["task2_class"].replace("house", 0)

df["task2_class"] = df["task2_class"].replace("dining_room", 1)

df["task2_class"] = df["task2_class"].replace("kitchen", 2)

df["task2_class"] = df["task2_class"].replace("bathroom", 3)

df["task2_class"] = df["task2_class"].replace("living_room", 4)

df["task2_class"] = df["task2_class"].replace("bedroom", 5)

df_task1 = pd.DataFrame(data=np.asarray(np.hstack((np.expand_dims(df.iloc[:, 0].values, -1), df.iloc[:, 4:]))),
                        columns=['filename'] + list(df.iloc[:, 4:].columns))
df_task2 = pd.DataFrame(data=np.asarray(np.hstack((np.expand_dims(df.iloc[:, 0].values, -1),
                                                   np.expand_dims(df.iloc[:, 2].values, -1)))),
                        columns=['filename'] + ['task2_class'])
df_task2.drop(df_task2.loc[df_task2['task2_class'] == 'validation'].index, inplace=True)

k = np.asarray(np.hstack((np.expand_dims(df.iloc[:, 0].values, -1),
                          np.expand_dims(df.iloc[:, 1].values, -1))))
k = np.hstack((k, np.expand_dims(df.iloc[:, 3], -1)))
df_task3 = pd.DataFrame(data=k, columns=['filename'] + ['standard'] + ['tech_cond'])

df_task1.to_csv(r"C:\Users\luktu\Desktop\skyhacks-challange\data\task1.csv", index=False)
df_task2.to_csv(r"C:\Users\luktu\Desktop\skyhacks-challange\data\task2.csv", index=False)
df_task3.to_csv(r"C:\Users\luktu\Desktop\skyhacks-challange\data\task3.csv", index=False)
