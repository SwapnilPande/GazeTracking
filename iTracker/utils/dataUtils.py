import pandas as pd

def getDescriptiveStats(data):
    df = pd.DataFrame(data)
    return df.describe()