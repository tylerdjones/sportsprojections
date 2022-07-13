#!/usr/bin/env python
# coding: utf-8

# ## Capstone: Projecting Wins in Major League Baseball
# #### By Tyler Jones

# The goal of this analysis is to project a player's contribution to his team winning, based on his age and past performance.
# 
# ---
# 
# Start by importing the holy trinity.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Next, import all available MLB data dating back to 1871, courtesy of Sean Lahman (https://www.seanlahman.com/baseball-archive/statistics/).

# In[3]:


appearances = pd.read_csv('desktop/core/Appearances.csv')
batting = pd.read_csv('desktop/core/Batting.csv')
fielding = pd.read_csv('desktop/core/Fielding.csv')
fieldingOF = pd.read_csv('desktop/core/FieldingOF.csv')
fieldingOFsplit = pd.read_csv('desktop/core/FieldingOFsplit.csv')
homegames = pd.read_csv('desktop/core/HomeGames.csv')
parks = pd.read_csv('desktop/core/Parks.csv')
people = pd.read_csv('desktop/core/people.csv')
pitching = pd.read_csv('desktop/core/Pitching.csv')
salaries = pd.read_csv('desktop/core/Salaries.csv')
teams = pd.read_csv('desktop/core/Teams.csv')


# After examing all of the dataframes (I won't muddy the markdown by looking at all of them, as I'll just need three for this project), I merge `batting` and `pitching` with `people`.

# In[4]:


offense = pd.merge(batting, people, on='playerID', how='outer')


# In[5]:


pitching = pd.merge(pitching, people, on='playerID', how='outer')


# ### What's in a win?
# #### Step 1: Run Differential
# 
# The first step is to model runs against winning percentage. Baseball has a large sample (162 games per season with roughly 8 runs scored per game). The difference between Runs Scored and Runs Allowed is known as "Run Differential," and it will the critical component of this analysis.
# 
# I'll start by using the `teams` dataframe to model run differential against winning percentage.
# 
# I need to create a few key components in order to model run differential per unit of input against overall winning percentage. These are:
# 
# - `W%`: The percent of total games that each team won in a season.
# - `PA`: Plate appearances are the number of times a hitter gets to bat.
# - `BF`: Batters faced are the number of batters a pitcher faces.
# - `R/PA`: Number of runs scored per plate appearance.
# - `R/BF`: Number of runs allowed per batter faced.
# - `RunDiff/Batter`: The number of runs scored per plate appearance minus the number of runs allowed per batter faced.
# 
# I've also reduced the dataframe to only include data since 1947. This year was the first that black players were allowed to play.

# In[6]:


teams['W%'] = teams['W'] /teams['G']
teams['PA'] = teams['AB'] + teams['BB'] + teams['HBP'] + teams['SF']
teams['BF'] = teams['IPouts'] + teams['HA'] + teams['BB']
teams['RA/BF'] = teams['RA']/teams['BF']
teams['R/PA'] = teams['R']/teams['PA']
teams['RunDiff/Batter'] = teams['R/PA'] - teams['RA/BF']
teams['yearID'] = teams['yearID'].astype(int)
teams = teams[teams['yearID'] > 1946]


# Next, I'll set up a train-test split for Linear Regression.

# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

winmodel = LinearRegression()


# In[8]:


q = teams['RunDiff/Batter']
r = teams['W%']

q = q.astype(float)
q = q.values.reshape(-1,1)


# In[9]:


q_train, q_test, r_train, r_test = train_test_split(q, r, test_size = 0.3, random_state=1)


# In[10]:


winmodel = LinearRegression()

winmodel.fit(q_train, r_train)


# In[11]:


winmodel.score(q_train, r_train)


# In[12]:


winmodel.score(q_test,r_test)


# The models scored pretty well. Roughly 12% of winning percentage isn't explained by run differential, which makes sense. A team could win a game 15-0, and lose a game 1-0. In total, their run differential is +14, but their winning percentage is only .500.
# 
# Next, I want to get a visual of the relationship between `W%` and `RunDiff/Batter`.

# In[13]:


winmodel_pred = winmodel.predict(q_test)


# In[14]:


plt.figure()
plt.xlim(-.1,.1)
plt.scatter(q_test, r_test, c='red', label='data') # plot original data
plt.scatter(q_test, winmodel_pred, c='blue', label='predictions') # plot predictions
plt.legend()
plt.title('How Run Differential Correlates with Winning')
plt.xlabel('Run Differential/Batter')
plt.ylabel('Winning%')
plt.show()


# Quite linear.
# 
# Now that it's been identified that runs are the foundation of wins, I'll move to find the foundation of runs.

# ### Runs Scored
# #### Step 2: Expected offensive runs
# 
# The goal of this step of the analysis is to determine the factors that contribute to each hitter's runs created.

# In[15]:


offense.info()


# Looks like there a few null values, but it doesn't appear to be a significant number. I'll just make sure:

# In[16]:


odatamissing = offense.isna().sum()
odatamissing/len(offense)


# Less than 0.2% of the data contains null values, so I feel comfortable simply dropping those columns. Likely, this data is from the 1800's when data collection wasn't perfect, so it will likely be removed from the dataframe organically anyway.

# In[17]:


## drop nulls

offense = offense.dropna()


# In[18]:


offense.info()


# Null values are gone. Now to create a column that gives us the age of each player in each season.

# In[19]:


offense['yearID'] = offense['yearID'].astype(int)
offense['birthYear'] = offense['birthYear'].astype(int)


# In[20]:


offense['age'] = offense['yearID'] - offense['birthYear']


# Now to combine the `nameFirst` and `nameLast` columns to give us each player's name.

# In[21]:


offense['player_name'] = offense['nameFirst'] + ' ' + offense['nameLast']


# In[22]:


offense.info()


# There is a ton of unnecessary data in this dataframe. I'll now drop all of it.

# In[23]:


offense = offense.drop(['G',"playerID","lgID","birthMonth",'birthYear', 'nameFirst','nameLast',"birthDay","birthCountry","birthState","birthCity","deathYear","deathMonth","deathDay","deathCountry","deathState","deathCity","nameGiven","weight","height","bats","throws","finalGame","retroID","bbrefID",'debut','stint'], axis = 1)


# Then, I want to make sure the `player_name` column is at the front of the dataframe.

# In[24]:


name = offense.pop('player_name')
offense.insert(0, "player_name", name)


# In[25]:


offense.info()


# Next, I'll create the following columns to use for analysis.
# 
# - `PA`: plate appearances
# - `H-HR`: all of a player's successful hits, minus their home runs. While home runs are hits, they have a direct impact on run creation (if you hit one, you automatically score a run). Other hits (singles, doubles, and triples) contribute to run production differently, however the difference between hitting a single and a double, or a double and a triple, is often a result of luck*. Perhaps the defense was out of position, or wasn't paying attention as you took an extra base.
# 
# **these types of "luck" events are important to exclude from the analysis where possible, so as to get the most accurate projections independent of factors out of the hitter's control.*

# In[26]:


offense['PA'] = offense['AB'] + offense['BB'] + offense['HBP'] + offense['IBB']
offense['H-HR'] = offense['H'] - offense['HR']


# In[27]:


offense.info()


# Next, narrow the dataframe down to results only since 1947.

# In[28]:


offense = offense[offense['yearID'] > 1946]


# In[29]:


offense.info()


# Next, I need to create a new dataframe called `qualoff`, which will contain qualified hitters with a minimum number of plate appearances. In baseball, players will often get limited opportunities to play for a few games here and there. This data can have lots of variance and it is advantageous to exclude it from the dataset.
# 
# Additionally, because players will sometimes be traded between teams during the course of a season, they will have multiple entires per year. To avoid this, I will group my `qualoff` dataframe by `player_name` and `yearID`.

# In[30]:


PAmean = offense['PA'].mean()
qualoff1 = offense[offense['PA'] >= PAmean]


# In[31]:


qualoff = qualoff1.groupby(['player_name', 'yearID'], as_index = False).mean().round(decimals = 0)
qualoff.head()


# In[32]:


qualoff.info()


# Next, I want to know how batters the typical team uses per season, and the percentage of total batters that reach the `PA` qualification threshold.

# In[33]:


total_teams = len(teams)
total_batters = len(offense)
batters_per_team = total_batters/total_teams
qualified_batters = len(qualoff)
batters_above_replacement = qualified_batters/total_batters
print(batters_per_team)
print(batters_above_replacement)


# I'm not sure I'll need that information, but at the very least it gives the insight that only about 30% of total hitters will reach a qualified number of plate appearances in a season. This could be useful later in narrowing the dataframe.

# ---
# 
# Next, I'll define variables that are more-or-less in the control of every batter, and use it to set up linear regression. These variables are:
# 
# - `H-HR` & `HR`: discussed earlier
# - `SB`: Stolen bases. When a player reaches base and then advances another base without a ball being hit by the batter. By stealing a base, a player gets closer to scoring a run.
# - `CS`: Caught stealing. This is the number of times a player is tagged out while trying to steal a base.
# - `BB`: Walks. If a player sees four balls before any other outcome, they are automatically awarded first base.
# - `SO`: Strikeouts. If a player swings and misses, or fails to swing at, 3 strikes then they are out.

# In[34]:


X = qualoff[['H-HR','HR','SB','CS','BB','SO']]
y = qualoff['R']


# In[35]:


offlin = LinearRegression()


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)


# In[37]:


offlin.fit(X_train,y_train)
score = offlin.score(X_train,y_train)
score


# In[38]:


offlin.predict(X_test)


# In[39]:


offlin.score(X_test, y_test)


# The model scores very well. It seems that the 6 factors above explain >90% of run production.

# The linear regression model produces a player's expected run contribution to a team based on their performance. By fitting this model prediction, I'll get an expected run contribution per plate appearance, which I will define as `RC/PA`.

# In[40]:


y_pred = offlin.predict(X)
qualoff['RC'] = y_pred
qualoff['RC/PA'] = qualoff['RC'] / qualoff['PA']


# In[41]:


o = qualoff['RC/PA']


# In[42]:


o = o.values.reshape(-1,1)


# The final step of determining win contribution is to fit the `RC/PA` data to our model that predicted run differential and wins.
# 
# I then subtract .5 (a 50% winning percentage) from the fit data, as offensive run production is just one-half of run differential, and thus one-half of winning percentage. Even if a hitter does everything right, pitchers can still give up more runs than the hitter has produced, leading to a loss.

# In[43]:


qualoff['WC'] = winmodel.predict(o) - .5


# Let's see which players have had the greatest contribution to winning in our dataset.

# In[44]:


qualoff.sort_values('WC', ascending = False)


# No surprise, the top 5 season by `WC` were produced by players now in the Hall of Fame.

# ---
# 
# So there you have it, we know what players need to do to contribute to wins.
# 
# Now, based on their ages the following season, what can we project will happen to their win contribution?
# 
# In order to determine this, I'll remodel the data using a new dataframe that groups all statistics by player age. I will then scale that data and, based on the rate of year-over-year change, adjust the previous season's output to attain a projected value for the following year.

# In[45]:


qualoffgroup = qualoff.groupby('age', as_index = False).mean()


# In[46]:


qualoffgroup


# Next, to remodel the data using the run prediction regression model for each age group.

# In[47]:


g = qualoffgroup[['H-HR','HR','SB','CS','BB','SO']]
h = qualoffgroup['R']


# In[48]:


gy_pred = offlin.predict(g)
qualoffgroup['all_RC'] = gy_pred
qualoffgroup['all_RC/PA'] = qualoffgroup['all_RC'] / qualoffgroup['PA']


# In[49]:


j = qualoffgroup['all_RC/PA']


# In[50]:


j = j.values.reshape(-1,1)


# In[51]:


qualoffgroup['all_WC'] = winmodel.predict(j) - .5


# In[52]:


qualoffgroup['all_WC']


# The data is all pretty tighly knit with a few outliers. I think that scaling it could help return meaningful differences and assist with accurate projection.
# 
# I've chosen the MinMaxScaler, because StandardScaler uses standard deviations from the mean of the dataset, and I'd prefer a scaling with a lighter touch.

# In[53]:


qog = qualoffgroup['all_WC'].values.reshape(-1,1)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X_data = qog
scaler.fit(X_data)
X_scaled = scaler.fit_transform(X_data)


qogDF = pd.DataFrame(X_scaled, columns = ['Scaled_WC'])


# The minimum age in the dataset is 19, so I'll add a column to our scaled dataframe that starts there and ends at the end of the list.

# In[54]:


qogDF['age'] = np.arange(19, 19+len(qogDF))


# In order to ensure the data doesn't have major outliers that completely skew projections, I will use a rolling average in the normalization columnn.

# In[55]:


qogDF['normalized'] = qogDF['Scaled_WC'].diff().rolling(3, min_periods = 1, center = True).mean()


# In[56]:


qogDF


# On examination, the normalized scaled data seems like it will be useful. Players get significantly better as they enter adulthood, level off, and then begin a slow decline around age 30. The rare few that make it to age 45 and beyond are usually all-time greats who seem to defy age.

# Next, to create a dataframe consisting solely of qualified players from 2021, to use for our 2022 projections.

# In[57]:


qualoff2021 = qualoff[qualoff['yearID'] == 2021]


# In[58]:


qualoff2021.sort_values('WC', ascending = False)


# In[59]:


qualoffgrouped2021 = qualoff2021.groupby(['player_name', 'age'], as_index=False).sum()


# In[60]:


qualoff2022proj = qualoffgrouped2021[['player_name', 'age','WC']]


# Next, to increase the age column by 1, since each player will be a year older in 2022 than they were in 2021.

# In[61]:


qualoff2022proj['age'] = qualoff2022proj['age'] + 1


# Next, to merge the dataframe to include the percentage change in performance at each age.

# In[62]:


qualoff2022proj = qualoff2022proj.merge(qogDF, on = 'age', how = 'left')


# In[63]:


qualoff2022proj.sort_values('WC', ascending = False)


# And finally, to add the expected increase or decrease in player performance in the dataset.

# In[64]:


qualoff2022proj['ProjectedW/PA'] = qualoff2022proj['WC'] * qualoff2022proj['normalized'] + qualoff2022proj['WC']


# In[65]:


qualoff2022proj.sort_values('ProjectedW/PA', ascending=False)


# So, what has been achieved here?

# In[66]:


z = qualoff2022proj['ProjectedW/PA']

plt.hist(z)
plt.axvline(z.mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(z.median(), color='r', linestyle='dashed', linewidth=1)
plt.show()
print(z.mean())
print(z.mean() * 162)
print(z.median())
print(z.median() * 162)


# When observing the distribution of qualified hitters, the mean `ProjectedW/PA` contribution results in a .459 winning percentage, or 74 wins out of 162 games. Intuitively, most hitters are likely worse than average, as it's a competitive sport with low margin for error.

# Hitters are responsible for runs scored by a team. The second component of run differential is runs allowed, which is controlled by pitching and defense. For the purposes of this analysis, I will focus solely on defense. I've explained my reasoning in the written report.

# ### Step 3
# #### Expected pitching runs

# First, to examine the pitching dataframe, which I merged earlier.

# In[67]:


pitching.info()


# There is a much greater than expected quantity of entries with null values. However, because all of the null values are in the columns that use the statistics necessary analysis, I must drop them.

# In[68]:


pitching = pitching.dropna()


# Now, to get the dataframe in order in the same way that I did with `offense`.

# In[69]:


pitching['yearID'] = pitching['yearID'].astype(int)
pitching['birthYear'] = pitching['birthYear'].astype(int)
pitching['age'] = pitching['yearID'] - pitching['birthYear']
pitching['player_name'] = pitching['nameFirst'] + ' ' + pitching['nameLast']


# In[70]:


pitching.info()


# In[71]:


pitching = pitching.drop(['G',"birthMonth",'birthYear','nameFirst','nameLast',"birthDay","birthCountry","birthState","birthCity","deathYear","deathMonth","deathDay","deathCountry","deathState","deathCity","nameGiven","weight","height","bats","throws","finalGame","retroID","bbrefID",'debut','stint'], axis = 1)


# In[72]:


name = pitching.pop('player_name')
pitching.insert(0, "player_name", name)


# In[73]:


pitching.info()


# In[74]:


pitching = pitching.drop(["playerID","lgID"], axis = 1)
pitching = pitching.drop(["W","L",'GS','CG','SHO','SV'], axis = 1)
pitching = pitching.drop(["BFP"], axis = 1)


# In[75]:


pitching.info()


# The dataframe is now free of most of the information unnecessary to the analysis. I will now add the following columns:
# 
# - `BF`: Batters faced. The total number of batters a pitcher faces in a season.
# - `H-HR`: Similarly useful to column of the same name in the `offense` dataframe.

# In[76]:


pitching['BF'] = pitching['IPouts'] + pitching['BB'] + pitching['HBP'] + pitching['IBB'] + pitching['H']
pitching['H-HR'] = pitching['H'] - pitching['HR']


# In[77]:


pitching.info()


# Now, I'll narrow the dataframe to data since 1947.

# In[78]:


pitching = pitching[pitching['yearID'] > 1946]


# In[79]:


pitching.info()


# Now to create a dataframe of qualified pitchers with a requirement of minimum batters faced.

# In[80]:


BFmean = pitching['BF'].mean()
qualpitch1 = pitching[pitching['BF'] >= BFmean]


# In[81]:


print(BFmean)


# In[84]:


qualpitch = qualpitch1.groupby(['player_name', 'yearID'], as_index = False).mean().round(decimals = 0)


# In[85]:


total_teams = len(teams)
total_pitchers = len(pitching)
pitchers_per_team = total_pitchers/total_teams
qualified_pitchers = len(qualpitch)
pitchers_above_replacement = qualified_pitchers/total_pitchers
print(pitchers_per_team)
print(pitchers_above_replacement)


# Again, not sure if this data will prove useful, but it is interesting to know that only 36% of pitchers face enough batters to qualify. (Note: I suspect this is how many baseball projection models define a "replacement-level" player: in other words, a player who is only in the 36th percentile of overall performers. These players are usually minor-league replacements.

# In[86]:


qualpitch


# I'll now set up linear regression. The difference in this linear regression, compared to the one performed for the `offense` dataset, is that I will use `ER` as the independent variable. `ER`, or Earned Runs, attempts to measure how many runs were scored by a team that were the fault of the pitcher, and not the fault of a defensive player making an error.

# In[87]:


a = qualpitch[['H','HR','SO','BB','HBP']]
b = qualpitch['ER']


# In[88]:


pitlin = LinearRegression()


# In[89]:


a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.3, random_state=1)


# In[90]:


pitlin.fit(a_train, b_train)
score = pitlin.score(a_train, b_train)
score


# In[91]:


pitlin.predict(a_test)


# In[92]:


pitlin.score(a_test, b_test)


# These factors explain >90% of runs allowed by pitchers.
# 
# Now, as with `offense`, I will create a column that measures pitcher run contribution per batter faced.

# In[93]:


b_pred = pitlin.predict(a)
qualpitch['RC'] = b_pred
qualpitch['RC/BF'] = qualpitch['RC'] / qualpitch['BF']


# In[94]:


p = qualpitch['RC/BF']


# In[95]:


p = p.values.reshape(-1,1)


# I'll fit this to the `winmodel`, but in reverse, in order to get a result that indicates a contribution to total wins.

# In[96]:


qualpitch['WC'] = winmodel.predict(p*-1) + .5


# In[97]:


qualpitch.sort_values('WC', ascending = False)


# In[98]:


qualpitch2021copy = qualpitch[qualpitch['yearID'] == 2021]

qualpitch2021copy['age'].max()


# I'll group the dataset by age in order to re-run regression and get a normalization factor to adjust performance by age for the 2022 player performance predictions.

# In[99]:


qualpitchgroup = qualpitch.groupby('age', as_index = False).mean()


# In[100]:


qualpitchgroup


# In[101]:


e = qualpitchgroup[['H','HR','SO','BB','HBP']]
f = qualpitchgroup['ER']


# In[102]:


ge_pred = pitlin.predict(e)
qualpitchgroup['all_RC'] = ge_pred
qualpitchgroup['all_RC/BF'] = qualpitchgroup['all_RC'] / qualpitchgroup['BF']


# In[103]:


k = qualpitchgroup['all_RC/BF']


# In[104]:


k = k.values.reshape(-1,1)


# In[105]:


qualpitchgroup['all_WC'] = winmodel.predict(k*-1) + .5


# In[106]:


qpg = qualpitchgroup[['age', 'all_WC']]


# In[107]:


qpg = qpg['all_WC'].values.reshape(-1,1)


# In[108]:


print(qualpitch['WC'].max())
print(qualpitch['WC'].min())


# Again, I will use the MinMaxScaler in order to prevent extreme variance from wildly misleading the projection model.

# In[109]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
a_data = qpg
scaler.fit(a_data)
a_scaled = scaler.fit_transform(a_data)


# In[110]:


qpgDF = pd.DataFrame(a_scaled, columns = ['Scaled_WC'])


# In[111]:


qpgDF['age'] = np.arange(18, 18+len(qpgDF))


# In[112]:


qpgDF['normalized'] = qpgDF['Scaled_WC'].diff().rolling(3, min_periods = 1, center = True).mean()


# In[113]:


qpgDF


# In[114]:


n = qpgDF['Scaled_WC']

plt.hist(n)
plt.axvline(n.mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(n.median(), color='k', linestyle='dashed', linewidth=1)
plt.show()


# The results of the scaling and normalization are very similar to that of the offense dataset. While this won't cause major shifts in the final projected model, I'd actually consider that to be advantageous. It is intuitive to lightly adjust individual performance against trends, rather than shifting it wildly.
# 
# Almost done with pitchers. Now to create a dataframe of 2021 results.

# In[115]:


qualpitch2021 = qualpitch[qualpitch['yearID'] == 2021]


# In[116]:


qualpitch2021.sort_values('WC', ascending = False)


# In[117]:


qualpitchgrouped2021 = qualpitch2021.groupby(['player_name', 'yearID', 'age'], as_index=False).sum()


# In[118]:


qualpitch2022proj = qualpitchgrouped2021[['player_name', 'age','WC']]


# In[119]:


qualpitch2022proj['age'] = qualpitch2022proj['age'] + 1


# In[120]:


qualpitch2022proj = qualpitch2022proj.merge(qpgDF, on = 'age', how = 'left')


# In[121]:


qualpitch2022proj.sort_values('WC', ascending = False)


# In[122]:


qualpitch2022proj['ProjectedW/BF'] = qualpitch2022proj['WC'] * qualpitch2022proj['normalized'] + qualpitch2022proj['WC']


# In[123]:


qualpitch2022proj.sort_values('ProjectedW/BF', ascending=False)


# What are we looking at here?

# In[124]:


w = qualpitch2022proj['ProjectedW/BF']

plt.hist(w)
plt.axvline(w.mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(w.median(), color='r', linestyle='dashed', linewidth=1)
plt.show()
print(w.mean())
print(w.mean() * 162)
print(w.median())
print(w.median() * 162)


# It appears that the average qualified pitcher has a big advantage over the average qualified hitter. At first this might be shocking, but I suspect it is because a qualified pitcher is in the 36th percentile of performers, whereas the bar for qualified hitters is only 28%.
# 
# ---
# 
# 
# Hitters: done.
# Pitchers: done.
# 
# Now, how does this modeled performance apply to a hypothetical team?

# ### Step 4
# #### The Test: How many games would a hypothetical team win?

# There are 26 players per team roster. As such, I will pull 13 hitters and 13 pitchers from our qualified datasets in order to produce results for a sample team, sorted in descending order of run contribution.

# In[125]:


testDFo = qualoff2022proj[['player_name','ProjectedW/PA']].sample(13).sort_values('ProjectedW/PA', ascending = False)
testDFp = qualpitch2022proj[['player_name','ProjectedW/BF']].sample(13).sort_values("ProjectedW/BF", ascending = False)


# Here is what the roster will look like:

# In[126]:


print(qualoff2021['PA'].max())
print(qualoff2021['PA'].min())


# The dataset has all of the player win contributions per plate appearance, or per batter faced. Next, I will use a random number generator to assign a number of plate appearnaces, or batters faced, to each player.
# 
# I will create an array in descending order, and then apply that array to the dataframe in descending order by Projected Wins. In doing this, we'll see what a team's performance might look like if total opportunities to bat or pitch are distributed to the top performers on a team.

# In[127]:


d = qualoff2021['PA'].max()
l = qualoff2021['PA'].min()
print(f'The maximum number of plate appearances by a player in 2021 was: {d}')
print(f'The minimum number of plate appearances by a player in 2021 was: {l}')


# I'll create an array of random integers between 721 and 143 for this team.

# In[128]:


r = np.random.randint(192,721,testDFo.shape[0])


# In[129]:


ri = -np.sort(-r)


# In[130]:


ri = ri.reshape(-1,1)


# In[131]:


testDFo['samplePA&BF'] = ri


# In[132]:


testDFo


# In[133]:


d = qualpitch2021['BF'].max()
l = qualpitch2021['BF'].min()
print(f'The maximum number of batters faced by a player in 2021 was: {d}')
print(f'The minimum number of batters faced by a player in 2021 was: {l}')


# Same process, the minimum and maximum batters faced will be the range of random integers.

# In[134]:


s = np.random.randint(313,864,testDFp.shape[0])


# In[135]:


si = -np.sort(-s)


# In[136]:


si = si.reshape(-1,1)


# In[137]:


testDFp['samplePA&BF'] = si


# In[138]:


testDFp


# Now, to combine it all into one dataframe.

# In[139]:


testDFall = pd.concat([testDFo, testDFp])


# In[140]:


testDFall


# Just need to replace the NaN values with 0 in order to complete the final calculation.

# In[141]:


testDFall = testDFall.fillna(0)


# The final column I'll create here is `sampleWC`, which is the theoretical runs created by a batter or pitcher in this environment, given their assigned plate appearances or batters faced.

# In[142]:


testDFall['sampleWC'] = ((testDFall['ProjectedW/PA']+testDFall['ProjectedW/BF'])*testDFall['samplePA&BF'])


# In[143]:


testDFall.head()


# In[144]:


testDFall['WC1'] = testDFall['ProjectedW/PA'] * (testDFall['samplePA&BF']/testDFall['samplePA&BF'].sum())*162
testDFall['WC2'] = testDFall['ProjectedW/BF'] * (testDFall['samplePA&BF']/testDFall['samplePA&BF'].sum())*162
testDFall['WC'] = testDFall['WC1'] + testDFall['WC2']


# In[145]:


testDFall.sort_values('WC', ascending = False)


# In[146]:


testDFall['WC'].sum()


# So, what are we looking at?
# 
# In order to end up with a final win projection, I'll add the total run contribution of the team and divide the sum by the total number of plate appearances and batters faced. After multiplying this figure by 162 (the number of games in a season), we'll finally land on a projection of team wins.

# In[147]:


print(testDFall['ProjectedW/PA'].sum()/13*162)
print(testDFall['ProjectedW/BF'].sum()/13*162)
print((testDFall['ProjectedW/PA'].sum()/13*162+testDFall['ProjectedW/BF'].sum()/13*162)/2)


# In[148]:


t = (testDFall['ProjectedW/PA'].sum()/13*162+testDFall['ProjectedW/BF'].sum()/13*162)/2
u = testDFall['sampleWC'].sum()
v = testDFall['samplePA&BF'].sum()


# In[149]:


print(f'This team would be expected to win {(u / v * 162)} games with this allocation of PA and BF.')
print(f'This represents {(u / v * 162)- t} more wins than if PA and BF were allocated evenly.')


# The model is complete.
