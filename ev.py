import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_path = r'C:\Users\DELL\PycharmProjects\ev sales\.venv\IEA Global EV Data 2010-2024.csv'
data = pd.read_csv(file_path)

print(data.head(10))


# handling missing values

missing_values = data.isnull().sum()
print(missing_values)
print("missing values in each column:")
print(missing_values)

# #filter data for EV stock share
# ev_stock_share = data[data['parameter'] == 'EV stock share']
#
# # plot the distribution of EV stock share over the years
# plt.figure(figsize=(12,8))
# sns.lineplot(data=ev_stock_share,x='year',y='value',hue='region',marker='o')
# plt.title('EV stock share distribution over years')
# plt.xlabel('year')
# plt.ylabel('EV stock share(%)')
# plt.legend(title='region')
# plt.grid(True)
# plt.show()
#
# # in matplotib by line chart
# plt.figure(figsize=(10, 6))
# plt.plot(data['year'], data['value'], marker='o', color='red',linewidth= 0.4,label="region")
# plt.title('EV stock share distribution over years')
# plt.xlabel('year')
# plt.ylabel('EV stock share(%)')
# plt.legend(title="region")
# plt.xticks(rotation=45)
# plt.show()
#
# # plot the distribution for the ev stock share
#
# # Filter data for "EV sales"
# ev_sales = data[data['parameter'] == 'ev sales']
#
# plt.figure(figsize=(10, 8))
# sns.countplot(data=ev_sales, x=data['region'], palette='crest', edgecolor='black', linewidth=1.5)
# plt.title('Number of EV Sales Data Entries by Region', fontsize=16, fontweight='bold')
# plt.xlabel('Region', fontsize=14)
# plt.ylabel('Count of Data Entries', fontsize=14)
# plt.xticks(rotation=45 ,ha="right")
# plt.tight_layout()
# plt.show()
#
# # filter data for ev stock
# ev_stock = data [data['parameter'] == 'EV stock']
#
# # aggregrate EV stock by powertrain type
# ev_stock_by_powertrain = ev_stock.groupby('powertrain')['value'].sum().sort_values(ascending=False)
#
# # plot EV stock by powertrain
# plt.figure(figsize=(12,8))
# ev_stock_by_powertrain.plot(kind='bar',color=plt.get_cmap('Paired').colors)
# plt.title('EV stock by powertrain type')
# plt.xlabel('powertrain type')
# plt.ylabel('EV stock (vehicles)')
# plt.xticks(rotation=45,ha='right')
# plt.gca().set_axisbelow(True)
# plt.grid(axis='y',linestyle='--',alpha=1)
# plt.show()
#
# # in matplotib by bar chart
# plt.figure(figsize=(12,8))
# plt.bar(data['powertrain'], data['value'],color='yellow')
# plt.title('EV stock by powertrain type')
# plt.xlabel('powertrain type')
# plt.ylabel('EV stock (vehicles)')
# plt.show()
#
#
# # filter data for ev sales shares
# ev_sales_share = data[data['parameter'] == 'EV sales share']
#
# # Aggregate EV sales share by region
# ev_sales_share_by_region = ev_sales_share.groupby('region')['value'].mean().sort_values(ascending=False)
#
# # in matplotib by bar chart
# plt.figure(figsize=(12,8))
# plt.bar(data["region"],data["value"],color=["blue","aqua","brown","red","white","pink","green","black","yellow","skyblue"],edgecolor='red',linewidth=1.1,label="car_model")
# plt.title("units sold per car model")
# plt.gca().set_axisbelow(True)
# plt.grid(True,axis='y',linestyle='--',alpha=0.7)
# plt.xlabel("region",fontsize=14,fontweight='bold')
# plt.ylabel("Average EV sales share(%)",fontsize=14,fontweight='extra bold')
# plt.xticks(rotation=45,fontsize=14,ha= 'right')
# plt.legend(title='')
# plt.tight_layout()
# plt.show()
#
# # Filter data for EV stock and EV sales by powertrain
# ev_stock = data[data['parameter'] == 'EV stock']
# ev_sales = data[data['parameter'] == 'EV sales']
#
# # Aggregate by powertrain type
# ev_stock_by_powertrain = ev_stock.groupby('powertrain')['value'].sum()
# ev_sales_by_powertrain = ev_sales.groupby('powertrain')['value'].sum()
#
# # Plot EV stock and sales by powertrain
# plt.figure(figsize=(14, 8))
# plt.bar(ev_stock_by_powertrain.index, ev_stock_by_powertrain, color='lightblue', label='EV Stock')
# plt.bar(ev_sales_by_powertrain.index, ev_sales_by_powertrain, color='lightcoral', alpha=0.7, label='EV Sales')
# plt.title('EV Stock and Sales by Powertrain')
# plt.xlabel('Powertrain Type')
# plt.ylabel('Count')
# plt.legend()
# plt.xticks(rotation=45, ha='right')
# plt.grid(axis='y')
# plt.show()
# #
# # # Filter data for EV stock share by powertrain
# ev_stock_share = data[data['parameter'] == 'EV stock share']
# #
# # # Plot trend of EV stock share by powertrain over the years
# plt.figure(figsize=(14,8))
# sns.lineplot(data=ev_stock_share, x='year', y='value', hue='powertrain', marker='o')
# plt.title('Trend of EV stock share by powertrain over the years')
# plt.xlabel('year')
# plt.ylabel('EV stock share (%)')
# plt.legend(title='powertrain')
# plt.grid(True)
# plt.show()
#
# # what is the average EV stock by country?
# # how has the ev sales share changes globally over the year?
# # what are the top 5 countries with highest ev stock in 2024?
# # what is the trend of bev stock over the years in usa?
# # which model has the highest ev sales share globally in 2021?
# # how many unique countries in the dataset?
# # what is the total ev stock in china in 2020?
# # what is the % growth in ev stock fron 2020 to 2024 globally?
# # which country has the most consistent ev stock share?
#
#
# # # ------------------------------------*answer*---------------------------------------------------------
# # #filter data for EV stock
# avg_ev_stock_share = data[data['parameter'] == 'EV stock share'].groupby ('region')['value'].mean()
# print(avg_ev_stock_share)
# #
# # #  how has the ev sales share changes globally over the year?
# globally_ev_sales_share = data[(data['parameter'] == 'EV sales share') & (data['year'])].groupby ('region')['value'].mean()
# print(globally_ev_sales_share)
# #
# # # what are the top 5 countries with highest ev stock in 2024?
# ev_stock = data[data['parameter'] == 'ev stock']
# #
# # # what is the trend of bev stock over the years in usa?
# trend_of_bev_stock=data[data['powertrain']=='BEV'].groupby('region')['value'].sum()
# print(trend_of_bev_stock)
# #
# # # which model has the highest ev sales share globally in 2021
# model=data[(data['parameter']=='EV sales share')  & (data["year"] == 2021)].groupby("mode")["value"].sum().idxmax()
# print(model)
# #
# # # how many unique countries in the dataset?
# unique_countries = data['region'].nunique()
# print(f"Number of unique countries: {unique_countries}")
#
# # # what is the total ev stock in china in 2020?
# total_ev_stock_share = data[data['parameter'] == 'EV stock share'].groupby ('region')['value'].mean()
# print(total_ev_stock_share)
# #
# # # what is the % growth in ev stock fron 2020 to 2024 globally?
# globally_ev_stock_share = data[(data['parameter'] == 'EV stock share') & (data['year'])].groupby ('region')['value'].mean()
# print(globally_ev_stock_share)
# #
# # # # which country has the most consistent ev stock share?
# consistent_ev_stock_share = data[data['parameter'] == 'EV stock share'].groupby ('region')['value'].var().idxmin()
# print(consistent_ev_stock_share)
# #
# #
# #
# #
# # --------------------------------answer----------------------------------------------------
# # # What is the total number of EVs sold globally from 2010 to 2024?
# globally_ev_sold_share = data[(data['parameter'] == 'EV sales share') & (data['year'])].groupby ('region')['value'].sum()
# print(globally_ev_sold_share)
#
# # # Which country had the highest increase in EV sales in the last 5 years?
# increase=data[(data["parameter"]=="EV sales")  & (data['year']>=2019)].groupby("region")["value"].sum().diff().idxmax()
# print(increase)
#
# # # What percentage of global EV sales comes from the top 3 countries in 2024?
# global_ev_sales_share = data[(data['parameter'] == 'EV sales') & (data['year']==2023)].groupby ('region')['value'].sum().nlargest(3)
# print(global_ev_sales_share)
