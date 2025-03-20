# Linear-Regression
A playground for messing around with linear regressions

As a very basic first starting point, we are using `data_transform.py` to take some csv files some [Our World in Data](https://ourworldindata.org/) and pre-process them such that we have our first column as the TB cases, our second column as the GDP per Capita, and the third column as the life expectancy in each country in each year. We then use `main.py` to apply a multiple linear regression model to the outputted data and then test that model (obviously using different data to the training data). We then plot the results as a 3D scatter chart.


# Data Sources

* https://ourworldindata.org/grapher/incidence-of-tuberculosis-sdgs
* https://ourworldindata.org/grapher/gdp-per-capita-worldbank
* https://ourworldindata.org/grapher/life-expectancy