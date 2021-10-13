import numpy as np
import pandas as pd

numbersinlotto = 49
winningnumbers = 7

np.random.seed(5) # used to generate always same rundom numbers for debuging and analysing

def numbersgenerator():
    number = np.random.choice(numbersinlotto,winningnumbers, replace=False)
    df = pd.DataFrame(number).T

    for i in range (800):
        number = np.random.choice(numbersinlotto,winningnumbers, replace=False)
        df = df.append(pd.DataFrame(number).T)

    df.to_csv('rundomnumbers.csv', index=False)


def compare():
    df1=pd.read_csv('germanlotto_train_wo1.csv')
    df2=pd.read_csv('rundomnumbers_wo0.csv')

    result = df1[df1.apply(tuple,1).isin(df2.apply(tuple,1))]
    print(result)

#numbersgenerator()
#compare()

