import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

# import data from the CSV as a pandas dataframe
teams_df = pd.read_csv('Teams.csv')

# documentation: http://www.seanlahman.com/files/database/readme2017.txt
cols = ['yearID', 'lgID', 'teamID', 'franchID', 'divID', 'Rank',
        'G', 'GHome', 'W', 'L', 'DivWin', 'WCWin', 'LgWin', 'WSWin',
        'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP',
        'SF', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPOuts', 'HA', 'HRA',
        'BBA', 'SOA', 'E', 'DP', 'FP', 'name', 'park', 'attendance', 'BPF',
        'PPF', 'teamIDBR', 'teamIDlahman45', 'teamIDretro']


pd.set_option("display.max_columns", 55)
teams_df.columns = cols

# these columns do not hold any relevant data
drop_columns = ['lgID','franchID','divID','Rank','GHome',
             'L','DivWin','WCWin','LgWin','WSWin','SF',
             'name','park','attendance','BPF','PPF',
             'teamIDBR','teamIDlahman45','teamIDretro',
             'franchID', 'CS', 'HBP']

# create a new dataframe without the useless data
df = teams_df.drop(drop_columns, axis=1)
# problem cols are BB, SO, SB
print(df.head(2))
# display the amount of null values in each dataframe
print(df.isnull().sum(axis=0).tolist())
# fill empty data points with the median
df['SO'] = df['SO'].fillna(df['SO'].median())
df['SB'] = df['SB'].fillna(df['SB'].median())

print(df.head(2))

plt.hist(df['W'])
plt.title('Wins per season visualized')
plt.xlabel('wins')
plt.ylabel('final records')
plt.show()

# drop all values before 1990
df = df[df['yearID'] > 1950]

runs_per_year = {}
games_per_year = {}

runs_per_year = {}
games_per_year = {}

for index, row in df.iterrows():
    year = row['yearID']
    runs = row['R']
    games = row['G']
    if year in runs_per_year:
        runs_per_year[year] = runs_per_year[year] + runs
        games_per_year[year] = games_per_year[year] + games
    else:
        runs_per_year[year] = runs
        games_per_year[year] = games

mlb_runs_per_game = {}

for year, games in games_per_year.items():
    runs = runs_per_year[year]
    mlb_runs_per_game[year] = runs/games

print(mlb_runs_per_game) # display the average amount of runs in an mlb game every year

lists = sorted(mlb_runs_per_game.items())
x, y = zip(*lists)

print(mlb_runs_per_game)
print(x)
print(y)

plt.plot(x, y)
plt.title('MLB Yearly Runs Per Game')
plt.xlabel('Year')
plt.ylabel('Runs / Game')
# plt.xlim(1950, 2020)
# plt.ylim(2, 6)
plt.show()

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

df['R_per_game'] = df['R']/df['G']
df['RA_per_game'] = df['RA']/df['G']

ax1.scatter(df['R_per_game'], df['W'], c='blue')
ax1.set_title('Runs / Game and Wins')
ax1.set_ylabel('Wins')
ax1.set_xlabel('Runs / Game')

ax2.scatter(df['RA_per_game'], df['W'], c='red')
ax2.set_title('Runs Allowed / Game and Wins')
ax2.set_xlabel('Runs Allowed / Game')
ax2.set_ylabel('Wins')
plt.show()

# print values and their correlation to winning
corr = df.corr()['W'].sort_values().plot(kind='bar')
plt.title("Correlations of Variables and Wins")
plt.xlabel("Variable")
plt.ylabel("Win Correlation")
plt.show()
