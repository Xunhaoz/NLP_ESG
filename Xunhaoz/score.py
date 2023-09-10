import pandas as pd

df = pd.read_csv("../check.csv")
all_true = df[df['impact_type'] == df['predict']][df['impact_type'] == 1]
all_false = df[df['impact_type'] == df['predict']][df['impact_type'] == 0]
impact_true = df[df['impact_type'] != df['predict']][df['impact_type'] == 1]
impact_false = df[df['impact_type'] != df['predict']][df['impact_type'] == 0]

print(len(all_true))
print(len(all_false))
print(len(impact_true))
print(len(impact_false))
