import pandas as pd

train = pd.read_csv("./task2_train.csv")
valid = pd.read_csv("./task2_valid.csv")

train["task2_class"] = train["task2_class"].replace("house", 0)
valid["task2_class"] = valid["task2_class"].replace("house", 0)

train["task2_class"] = train["task2_class"].replace("dining_room", 1)
valid["task2_class"] = valid["task2_class"].replace("dining_room", 1)

train["task2_class"] = train["task2_class"].replace("kitchen", 2)
valid["task2_class"] = valid["task2_class"].replace("kitchen", 2)

train["task2_class"] = train["task2_class"].replace("bathroom", 3)
valid["task2_class"] = valid["task2_class"].replace("bathroom", 3)

train["task2_class"] = train["task2_class"].replace("living_room", 4)
valid["task2_class"] = valid["task2_class"].replace("living_room", 4)

train["task2_class"] = train["task2_class"].replace("bedroom", 5)
valid["task2_class"] = valid["task2_class"].replace("bedroom", 6)

train.to_csv("./task2_train_categorized.csv", index=False)
valid.to_csv("./task2_valid_categorized.csv", index=False)
