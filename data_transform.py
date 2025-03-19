import pandas as pd


def Process(renames, path):
    data = pd.read_csv(path)
    for rename in renames:
        data.rename(columns={rename[0]: rename[1]}, inplace=True)
    return data


def Merge(predictors, outputs, csvPath, columns):
    for dataSet in predictors:
        outputs = pd.merge(dataSet, outputs, on=["Entity", "Year"], how="inner")
    outputs = outputs[columns].dropna()
    outputs.to_csv(csvPath, index=False)
    return True


if __name__ == "__main__":
    lifeExpectancy = Process([["Period life expectancy at birth - Sex: total - Age: 0","Life Expectancy"]], "./data/life-expectancy.csv")
    tbCases = Process([["Estimated incidence of all forms of tuberculosis", "TB Cases"]], "./data/tb.csv")
    gdp = Process([["GDP per capita, PPP (constant 2021 international $)", "GDP"]], "./data/gdp.csv")
    Merge([tbCases, gdp], lifeExpectancy, "./data/data.csv",["TB Cases", "GDP", "Life Expectancy"])
