import numpy as np

go = True

#df = pd.DataFrame(columns=["1","2","3","4"])
df = []

while go:

    name = input("Name ")

    df.append(np.array([name ,input("1"), input("2"), input("3"), input("4")]))

    if not input("Go") == "ja":
        go = False

    print(df)

    