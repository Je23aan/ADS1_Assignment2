# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:19:13 2023

@author: jayan
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis


# Function to process data for a specific indicator and subset of countries
def process_data(filename, indicator_name):
    '''Process data from a CSV file for a specific indicator and subset of
    countries.

    Parameters:
    - filename (str): Path to the CSV file.
    - indicator_name (str): Name of the indicator to focus on.
    '''
    # Reading CSV file and skipping unnecessary rows
    dataframe = pd.read_csv(filename, skiprows=4)
    country = ['Australia', 'Japan', 'China',
               'India', 'United Kingdom', 'South Africa']
    # Selecting specific countries
    selection = dataframe[(dataframe['Indicator Name'] == indicator_name) & (
        dataframe['Country Name'].isin(country))]
    # Dropping unnecessary columns
    df = selection.drop(dataframe.columns[1:4], axis=1)
    df = df.drop(dataframe.columns[-1:], axis=1)
    #Data has NaN values dropping the unnecessasy years.
    # Creating a list of years to drop
    years = [str(year) for year in range(1960, 2001)] + [
        str(year)for year in range(2021, 2023)]
    # Dropping columns for specified years
    data = df.drop(columns=years)
    data = data.reset_index(drop=True)
    
    # Transposing the data for better visualization
    data_t = data.transpose()
    data_t.columns = data_t.iloc[0]
    data_t = data_t.iloc[1:]
    data_t.index = pd.to_numeric(data_t.index)
    data_t['Years'] = data_t.index
    return data, data_t

# Function to perform slicing on a DataFrame
def slicing(df):
    '''Perform slicing on a DataFrame, keeping only the 'Country Name' and '2020' columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    '''
    df = df[['Country Name', '2020']]
    return df

# Function to perform multiple outer merges on DataFrames
def merge(a, b, c, d, e):
    '''Perform multiple outer merges on DataFrames using the 'Country Name'
    column.

    Parameters:
    - a, b, c, d, e (pd.DataFrame): Input DataFrames to be merged.
    '''
    mer1 = pd.merge(a, b, on='Country Name', how='outer')
    mer2 = pd.merge(mer1, c, on='Country Name', how='outer')
    mer3 = pd.merge(mer2, d, on='Country Name', how='outer')
    mer4 = pd.merge(mer3, e, on='Country Name', how='outer')
    mer4 = mer4.reset_index(drop=True)
    return mer4

# Function to generate a heatmap for the correlation matrix
def heatmap(df):
    '''Generate a heatmap to visualize the correlation matrix of numeric
    columns in a DataFrame.
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    '''
    plt.figure(figsize=(6, 4))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', square=True,
                linewidths=.5, annot=True, fmt=".2f", center=0)
    plt.title("Correlation matrix of Indicators")
    plt.show()

# Function to generate a line plot for specific countries over the years
def lineplot(df, y_label, title):
    '''Generate a line plot for specific countries over the years.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for countries
    over the years.
    - y_label (str): Label for the y-axis.
    - title (str): Title of the plot.
    '''
    sns.set_style("whitegrid")
    df.plot(x='Years', y=['Australia', 'Japan', 'China', 'India',
                          'United Kingdom', 'South Africa'],
            xlabel='Years', ylabel=y_label, marker='.')
    plt.title(title)
    plt.xlabel('Years')
    plt.ylabel(y_label)
    plt.xticks(range(2000, 2021, 2))
    plt.legend(loc='best', bbox_to_anchor=(1, 0.4))
    plt.show()

# Function to generate a bar plot for a DataFrame
def barplot(df, x_value, y_value, head_title, x_label, y_label, colors,
            figsize=(6, 4)):
    '''Generate a bar plot for a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - x_value (str): The column name for the x-axis values.
    - y_value (str): The column name for the y-axis values.
    - head_title (str): Title of the plot.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - colors (list): List of colors for the bars.
    - figsize (tuple): Size of the figure. Default is (6, 4).
    '''
    sns.set_style('whitegrid')
    
    df_barplot = df[df['Years'].isin([2001, 2005, 2010, 2015, 2020])]
    df_barplot.plot(x=x_value, y=y_value, kind='bar', title=head_title,
                    color=colors,width=0.65, figsize=figsize, xlabel=x_label,
                    ylabel=y_label)
    plt.legend(loc='best', bbox_to_anchor=(1, 0.4))
    plt.savefig('barplot.png')
    plt.show()

# Function to generate a pie chart to represent the distribution of data
def pieplot(df, Years, title, autopct='%1.0f%%', fontsize=11):
    '''Generate a pie chart to represent the distribution of data for 
    specific years and selected countries.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for the specified
    years and countries.
    - Years (int): The year for which the pie chart is generated.
    - title (str): Title of the pie chart.
    - autopct (str): Format string for percentage labels. Default is '%1.0f%%'.
    - fontsize (int): Font size for labels. Default is 11.
    '''
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    label = ['Australia', 'Japan', 'China',
             'India', 'United Kingdom', 'South Africa']
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(4, 5))
    plt.pie(numeric_df[str(Years)], autopct=autopct, labels=label,
            explode=explode,
            startangle=180, wedgeprops={"edgecolor": "black", "linewidth": 2,
                                        "antialiased": True},)
    plt.title(title)
    plt.savefig('pieplot.png')
    plt.show()

# Function to generate a box plot to visualize the distribution of renewable energy consumption
def boxplot(df, countries, shownotches=True):
    '''Generate a box plot to visualize the distribution of renewable energy
    consumption for specific countries.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing renewable energy
    consumption data for countries.
    - countries (list): List of countries to be included in the box plot.
    - shownotches (bool): Whether to show notches in the box plot. Default is
    True.
    '''
    plt.figure(figsize=(6, 4))
    plt.boxplot([df[country] for country in countries])
    plt.title(
        'Renewable energy consumption (% of total energy consumption) in Box')
    plt.xticks(range(1, len(countries) + 1), countries)
    plt.savefig('boxplot.png')
    plt.show()

# Function to generate a dot plot to visualize data for different countries over the years
def dotplot(df, title, y_label):
    '''Generate a dot plot to visualize data for different countries over the
    years.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for countries and
    years.
    - title (str): Title of the dot plot.
    - y_label (str): Label for the y-axis.
    '''
    # Setting the style of the plot
    sns.set_style("whitegrid")
    # Creating a dot plot using Seaborn
    dot = sns.catplot(x='Years', y='Value', hue='Country', data=df.melt(
        id_vars=['Years'], var_name='Country', value_name='Value'),
        kind="point")
    # Rotating x-axis labels for better readability
    dot.set_xticklabels(rotation=90)
    # Adding title and labels to the plot
    plt.title(title)
    plt.ylabel(y_label)
    # Saving the plot as an image
    plt.savefig('dotplot.png')
    # Displaying the plot
    plt.show()

# Function to calculate skewness for each column in a DataFrame
def skew(df):
    '''Calculate skewness for each column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    '''
    # Calculating skewness using pandas
    df = df.skew()
    return df

# Function to calculate kurtosis for each column in a DataFrame
def kurt(df):
    '''Calculate kurtosis for each column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    '''
    # Calculating kurtosis using pandas
    df = df.kurtosis()
    return df

# Function to calculate the mean of a DataFrame using NumPy
def mean(df):
    '''Calculate the mean of a DataFrame using NumPy.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    '''
    # Calculating mean using NumPy
    df = np.mean(df)
    return df

# Main programme
# Specify the path to the CSV file
filename = r"C:\Users\jayan\OneDrive\Desktop\API_19_DS2_en_csv_v2_6183479.csv"
# Process data for different indicators
popu, popu_t = process_data(filename, 'Population growth (annual %)')
co2E, co2E_t = process_data(filename, 'CO2 emissions (metric tons per capita)')
agriff, avgff_t = process_data(
    filename, 'Agriculture, forestry, and fishing, value added (% of GDP)')
renew, renew_t = process_data(
    filename,
    'Renewable energy consumption (% of total final energy consumption)')
avgpre, avgpre_t = process_data(
    filename, 'Average precipitation in depth (mm per year)')

# Perform slicing and merging of DataFrames
popu_sli = slicing(popu).rename(
    columns={'2020': 'Population growth'})
co2E_sli = slicing(co2E).rename(
    columns={'2020': 'CO2 emissions'})
agriff_sli = slicing(agriff).rename(
    columns={'2020': 'Agriculture, forestry, and fishing'})
renew_sli = slicing(renew).rename(columns={
    '2020': 'Renewable energy consumption'})
avgpre_sli = slicing(avgpre).rename(
    columns={'2020': 'Average precipitation in depth'})

popu_co2E_agriff_renew_avgpre = merge(
    popu_sli, co2E_sli, agriff_sli, renew_sli, avgpre_sli)

'''Calling the function describe(), to know the value of Mean, Median, Count,
 Standard deviation, Quartile, Minimum, Maximum'''
 # Print descriptive statistics for the merged DataFrame
print(popu_co2E_agriff_renew_avgpre.describe())

# Generate a heatmap to visualize the correlation matrix
heatmap(popu_co2E_agriff_renew_avgpre)

# Generate a line plot for population growth
lineplot(popu_t, 'annual %', 'Population growth annual % in line plot')

# Generate a bar plot for agriculture value added
barplot(avgff_t, 'Years', ['Australia', 'Japan', 'China', 'India', 
                       'United Kingdom', 'South Africa'],
        'Barplot of Agriculture, forestry and fishing, value added (% of GDP)',
        'Years', 'value', ('yellow', 'grey', 'lightblue', 'green',
                           'purple', 'orange'))

# Generate a pie chart for average precipitation in 2020
pieplot(
    avgpre, 2020, 'Average precipitation in depth (mm per year) in 2020')
# Generate a box plot for renewable energy consumption
boxplot(renew_t, ['Australia', 'Japan', 'China', 'India',
                  'United Kingdom', 'South Africa'], shownotches=True)
# Generate a dot plot for CO2 emissions
dotplot(co2E_t, 'co2 emissions', 'metric tons per capita')

'''Calculate skewness, kurtosis, and mean for a specific country (India) in 
population growth
'''
skewness = skew(popu_t['India'])

kurtosis = kurt(popu_t['India'])

mean= mean(popu_t['India'])

# Print calculated statistics
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")
print(f"mean: {mean}")