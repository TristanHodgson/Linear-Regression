import pandas as pd

tb_path = "./data/incidence-of-tuberculosis-sdgs.csv"
life_path = "./data/life-expectancy.csv"
final_path = "./data/data.csv"

def Process():
    tb_data = pd.read_csv(tb_path)
    life_expectancy_data = pd.read_csv(life_path)
    tb_data.rename(columns={"Estimated incidence of all forms of tuberculosis": "TB Cases"}, inplace=True)
    life_expectancy_data.rename(columns={"Period life expectancy at birth - Sex: total - Age: 0": "Life Expectancy"}, inplace=True)

    merged_data = pd.merge(tb_data, life_expectancy_data, on=["Entity", "Year"], how="inner")
    merged_data = merged_data[["TB Cases", "Life Expectancy"]].dropna()

    merged_data.to_csv(final_path, index=False)

if __name__ == "__main__":
    Process()