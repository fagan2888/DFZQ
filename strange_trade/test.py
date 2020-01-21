import pandas as pd
data = [    [11.53,  11.69,  11.70,  11.51,   871365.0,  1],
            [11.64,  11.63,  11.72,  11.57,   722764.0,  2],
            [11.59,  11.48,  11.59,  11.41,   461808.0,  3],
            [11.39,  11.19,  11.40,  11.15,  1074465.0,  4]]
df = pd.DataFrame(data, index=["2017-10-18", "2017-10-19", "2017-10-20", "2017-10-23"],
                   columns=["open", "close", "high", "low", "volume", "code"])
print(df)
print(df.as_matrix())
print(df.values)
