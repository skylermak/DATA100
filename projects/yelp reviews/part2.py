#!/usr/bin/env python
# coding: utf-8

# In[120]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj1b.ok')


# # Project 1 Part B
# 
# ## Due Date: Monday, Feb 24th, 11:59PM
# 
# ## Collaboration Policy
# 
# Data science is a collaborative activity. While you may talk with others about
# the homework, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** below.
# 
# 

# **Collaborators**: *list  collaborators here*

# ## Scoring Breakdown
# 
# |Question|Points|
# |---|---|
# |1a|1|
# |1b|2|
# |1ci|3|
# |1cii|1|
# |2a|2|
# |2b|1|
# |2ci|4|
# |2cii|2|
# |2d|2|
# |2e|1|
# |2f|1|
# |2g|3|
# |3a|3|
# |3b|4|
# |3c|1|
# |3d|2|
# |4 |5|
# |**Total**|38|
# 

# First we import the relevant libraries for this project.

# In[121]:


import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
plt.style.use('fivethirtyeight')


# In the following cell, we will load the cleaned data from Part A of Project 1. Note that we will be using the relevant data files based on the staff solution.

# In[122]:


ins = pickle.load(open('./data/ins.p', 'rb'))
vio = pickle.load(open('./data/vio.p', 'rb'))
ins2vio = pickle.load(open('./data/ins2vio.p', 'rb'))
bus = pickle.load(open('./data/bus.p', 'rb'))


# **Note**: For all parts of this project requiring you to produce a visualization, we won't be enforcing any specific size on the plots you make, as long as they are clear (i.e. no overlaps) and follow the specifications. 

# <br/><br/><br/><br/>
# 
# ---
# 
# <br/><br/>
# 
# ## 1: Explore Inspection Scores
# 
# In this first part we explore some of the basic inspection score values visually.

# <br/><br/><br/>
# 
# ---
# 
# 
# ### Question 1a
# Let's look at the distribution of inspection scores. As we saw before when we called head on this data frame, inspection scores appear to be integer values. The discreteness of this variable means that we can use a barplot to visualize the distribution of the inspection score. Make a bar plot of the counts of the number of inspections receiving each score.
# 
# It should look like the image below. It does not need to look exactly the same (e.g., no grid), but make sure that all labels and axes are correct.
# 
# ![](data/1a.png)
# 
# You might find this matplotlib.pyplot tutorial useful. Key syntax that you'll need:
# 
# ```
# plt.bar
# plt.xlabel
# plt.ylabel
# plt.title
# ```
# 
# *Note*: If you want to use another plotting library for your plots (e.g. plotly, sns) you are welcome to use that library instead so long as it works on DataHub. If you use seaborn sns.countplot(), you may need to manually set what to display on xticks.
# 
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[123]:


ins = ins[ins['score'] > 0]
x = ins['score'].value_counts()
plt.bar(x.keys(), x, edgecolor = 'black')
plt.xlabel("Score")
plt.ylabel("Count")
plt.title("Distribution of Inspection Scores")


# <br/><br/><br/>
# 
# ---
# 
# ### Question 1b
# Describe the qualities of the distribution of the inspections scores based on your bar plot. Consider the mode(s), symmetry, tails, gaps, and anomalous values. Are there any unusual features of this distribution? What do your observations imply about the scores?
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# This distribution is a unimodal, skewed left graph. Inspection score of 100 are the most common at about 2000 total scores. An interesting feature of this grpah is that there are more even scores than odd scores (no scores of 99, 97, 95, etc). This may be due to the criteria that the inspections are graded upon, having less odd point deductions.

# <br/><br/><br/>
# 
# ---
# 
# ### Question 1c
# Let's figure out which restaurants had the worst scores ever (single lowest score). Let's start by creating a new dataframe called `ins_named`. It should be exactly the same as ins, except that it should have the name and address of every business, as determined by the bus dataframe. If a `business_id` in ins does not exist in bus, the name and address should be given as `NaN`.
# 
# **Hint**: Use the merge method to join the ins dataframe with the appropriate portion of the bus dataframe. See the official documentation on how to use merge.
# 
# **Note**: For quick reference, a pandas left join keeps the keys from the left frame, so if ins is the left frame, all the keys from ins are kept and if a set of these keys don't have matches in the other frame, the columns from the other frame for these "unmatched" key rows contains NaNs.
# 
# <!--
# BEGIN QUESTION
# name: q1ci
# points: 3
# -->

# In[124]:


ins_named = ins.merge(bus[['bid', 'name', 'address']], how='left', left_on='bid', right_on='bid')
ins_named.head()


# In[125]:


ok.grade("q1ci");


# In[126]:


worst_restaurant = ins_named.sort_values('score', ascending=True)
worst_restaurant


# **Use the cell above to identify the restaurant** with the lowest inspection scores ever. Be sure to include the name of the restaurant as part of your answer in the cell below. You can also head to yelp.com and look up the reviews page for this restaurant. Feel free to add anything interesting you want to share.
# 
# <!--
# BEGIN QUESTION
# name: q1cii
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# Lollipot. With a Yelp rating of about 3.5, I thought this place seemed pretty normal for a mediocre restaurant in SF. This review was pretty funny to me, "Is this place a joke? The hot pot broth has 0 taste at all whether it's spicy oily broth or not. Food quality is so mediocre and the decoration is just so old and out-dated." 
# 
# It seems as thought this restaurant has the lowest inspection score due to how insanitary an AYCE place can be, coupled with the poor service.
# 
# Another thing to note is that the restaurant is premanently close, so maybe the inspectors are just bagging on this establishment.

# <br/><br/><br/><br/>
# 
# ---
# 
# <br/><br/>
# 
# ## 2: Restaurant Ratings Over Time
# 
# Let's consider various scenarios involving restaurants with multiple ratings over time.
# 

# <br/><br/><br/>
# 
# ---
# 
# ### Question 2a
# 
# Let's see which restaurant has had the most extreme improvement in its rating, aka scores. Let the "swing" of a restaurant be defined as the difference between its highest-ever and lowest-ever rating. **Only consider restaurants with at least 3 ratings, aka rated for at least 3 times (3 scores)!** Using whatever technique you want to use, assign `max_swing` to the name of restaurant that has the maximum swing.
# 
# *Note*: The "swing" is of a specific business. There might be some restaurants with multiple locations; each location has its own "swing".

# The city would like to know if the state of food safety has been getting better, worse, or about average. This is a pretty vague and broad question, which you should expect as part of your future job as a data scientist! However for the ease of grading for this assignment, we are going to guide you through it and offer some specific directions to consider.
# 
# <!--
# BEGIN QUESTION
# name: q2a
# points: 2
# -->

# In[127]:


ins_names_three = ins_named.groupby('bid').filter(lambda x : len(x) > 2)
max_swing = ins_names_three['score'].groupby(ins_names_three['name']).agg(lambda x : max(x)- min(x)).sort_values(ascending = False).index[0]
max_swing


# In[128]:


ok.grade("q2a");


# <br/><br/><br/>
# 
# ---
# 
# ### Question 2b
# 
# To get a sense of the number of times each restaurant has been inspected, create a multi-indexed dataframe called `inspections_by_id_and_year` where each row corresponds to data about a given business in a single year, and there is a single data column named count that represents the number of inspections for that business in that year. The first index in the MultiIndex should be on `bid`, and the second should be on year.
# 
# An example row in this dataframe might look tell you that `bid` is 573, `year` is 2017, and `count` is 4.
# 
# Hint: Use `groupby` to group based on both the `bid` and the `year`.
# 
# Hint: Use `rename` to change the name of the column to count.
# 
# <!--
# BEGIN QUESTION
# name: q2b
# points: 1
# -->

# In[129]:


inspections_by_id_and_year = ins.groupby(['bid', "year"]).size().to_frame()
inspections_by_id_and_year = inspections_by_id_and_year.rename(columns={0: "count"})
inspections_by_id_and_year.head()


# In[130]:


ok.grade("q2b");


# You should see that some businesses are inspected many times in a single year. Let's get a sense of the distribution of the counts of the number of inspections by calling `value_counts`. There are quite a lot of businesses with 2 inspections in the same year, so it seems like it might be interesting to see what we can learn from such businesses.

# In[131]:


inspections_by_id_and_year['count'].value_counts()


# <br/><br/><br/>
# 
# ---
# 
# 
# ### Question 2c
# 
# What's the relationship between the first and second scores for the businesses with 2 inspections in a year? Do they typically improve? For simplicity, let's focus on only 2018 for this problem, using `ins2018` data frame that will be created for you below.
# 
# First, make a dataframe called `scores_pairs_by_business` indexed by `business_id` (containing only businesses with exactly 2 inspections in 2018). This dataframe contains the field score_pair consisting of the score pairs ordered chronologically [first_score, second_score].
# 
# Plot these scores. That is, make a scatter plot to display these pairs of scores. Include on the plot a reference line with slope 1.
# 
# You may find the functions `sort_values`, `groupby`, `filter` and `agg` helpful, though not all necessary.
# 
# The first few rows of the resulting table should look something like:
# 
# |     | **score_pair** |
# |-----|------------|
# | **bid** |            |
# | 48  | [94, 87]   |
# | 66  | [98, 98]   |
# | 146  | [81, 90]   |
# | 184  | [90, 96]  |
# | 273  | [83, 84]   |
# 
# In the cell below, create `scores_pairs_by_business` as described above.
# 
# Note: Each score pair must be a list type; numpy arrays will not pass the autograder.
# 
# Hint: Use the filter method from lecture 5 to create a new dataframe that only contains restaurants that received exactly 2 inspections.
# 
# Hint: Our code that creates the needed DataFrame is a single line of code that uses `sort_values`, `groupby`, `filter`, `groupby`, `agg`, and `rename` in that order. Your answer does not need to use these exact methods.
# 
# <!--
# BEGIN QUESTION
# name: q2ci
# points: 4
# -->

# In[132]:


ins2018 = ins[ins['year'] == 2018]
# Create the dataframe here
scores_pairs_by_business = ins2018.sort_values('date').groupby('bid').filter(lambda x: len(x)==2).groupby('bid').agg({'score': lambda group: group.tolist()}).rename(columns={'score':'score_pair'})

scores_pairs_by_business


# In[133]:


ok.grade("q2ci");


# Now, create your scatter plot in the cell below. It does not need to look exactly the same (e.g., no grid) as the sample below, but make sure that all labels, axes and data itself are correct.
# 
# ![](data/2c.png)
# 
# Key pieces of syntax you'll need:
# 
# `plt.scatter` plots a set of points. Use `facecolors='none'` and `edgecolors=b` to make circle markers with blue borders. 
# 
# `plt.plot` for the reference line.
# 
# `plt.xlabel`, `plt.ylabel`, `plt.axis`, and `plt.title`.
# 
# Note: If you want to use another plotting library for your plots (e.g. `plotly`, `sns`) you are welcome to use that library instead so long as it works on DataHub.
# 
# Hint: You may find it convenient to use the `zip()` function to unzip scores in the list.
# 
# <!--
# BEGIN QUESTION
# name: q2cii
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[134]:


x, y = zip(*scores_pairs_by_business['score_pair'])
plt.scatter(x, y, s=20, facecolors='none', edgecolors='b')
plt.plot([55, 100], [55, 100], 'r-')
plt.xlabel('First Score')
plt.ylabel('Second Score')
plt.axis([55, 100, 55, 100])
plt.title("First Inspection Score vs. Second Inspection Score")


# <br/><br/><br/>
# 
# 
# ---
# 
# 
# ### Question 2d
# 
# Another way to compare the scores from the two inspections is to examine the difference in scores. Subtract the first score from the second in `scores_pairs_by_business`. Make a histogram of these differences in the scores. We might expect these differences to be positive, indicating an improvement from the first to the second inspection.
# 
# The histogram should look like this:
# 
# ![](data/2d.png)
# 
# Hint: Use `second_score` and `first_score` created in the scatter plot code above.
# 
# Hint: Convert the scores into numpy arrays to make them easier to deal with.
# 
# Hint: Use `plt.hist()` Try changing the number of bins when you call `plt.hist()`.
# 
# <!--
# BEGIN QUESTION
# name: q2d
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[135]:


diff = np.array(y) - np.array(x)
plt.hist(diff, bins = 30, edgecolor="black")
plt.xlabel('Score Difference(Second Score - First Score)')
plt.ylabel("Count")
plt.title("Distribution of Score Differences")


# <br/><br/><br/>
# 
# 
# ---
# 
# 
# ### Question 2e
# 
# If restaurants' scores tend to improve from the first to the second inspection, what do you expect to see in the scatter plot that you made in question 2c? What do you oberve from the plot? Are your observations consistent with your expectations? 
# 
# Hint: What does the slope represent?
# 
# <!--
# BEGIN QUESTION
# name: q2e
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# The slope represents no change in score. In other words, the slope of the red line is the point at which the first inspection score was the same as the second inspection score. Since this is the case, since restaurants' scores tend to improve from the first to the second inspection, we can expect there to be more points above the red line. This is consistent with the graph, especially when we examine the points when the score of the first inspection was low.

# <br/><br/><br/>
# 
# ---
# 
# ### Question 2f
# 
# If a restaurant's score improves from the first to the second inspection, how would this be reflected in the histogram of the difference in the scores that you made in question 2d? What do you oberve from the plot? Are your observations consistent with your expectations? Explain your observations in the language of Statistics: for instance, the center, the spread, the deviation etc.
# 
# <!--
# BEGIN QUESTION
# name: q2f
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# If the scores improves from the first to the second inspection, the respected values for those restaurants would lie to the right of the 0 center. Since this is a unimodal distribution, and the mode of the differences lies at the center 0, we can conclude that most of the restaurants in this data set do not observe a change in inspection scores from the first to second inspection. The spread of the distribution is fairly even, having a max and min deviating about 20 units from the center.  

# <br/><br/><br/>
# 
# ---
# 
# ### Question 2g 
# To wrap up our analysis of the restaurant ratings over time, one final metric we will be looking at is the distribution of restaurant scores over time. Create a side-by-side boxplot that shows the distribution of these scores for each different risk category from 2017 to 2019. Use a figure size of at least 12 by 8.
# 
# The boxplot should look similar to the sample below:
# 
# ![](data/2g.png)
# 
# **Hint**: Use `sns.boxplot()`. Try taking a look at the first several parameters.
# 
# **Hint**: Use `plt.figure()` to adjust the figure size of your plot.
# 
# <!--
# BEGIN QUESTION
# name: q2g
# points: 3
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[136]:


# Do not modify this line
sns.set()
merged = ins.merge(ins2vio, on='iid').merge(vio, on='vid')
ax = sns.boxplot(x='year', y='score', data=merged, hue='risk_category', hue_order=['Low Risk', 'Moderate Risk', 'High Risk'], order=[2017, 2018, 2019])
plt.figure(figsize=(12, 8))


# <br/><br/><br/><br/>
# 
# ---
# 
# <br/><br/>
# 
# 
# ## Question 3 Interpreting Visualizations ##
# 

# <br/><br/><br/>
# 
# ---
# 
# ### Question 3a ###
# 
# Given a set of data points `(x[i], y[i], c[i])`, a hexbin plot is a visualization of what the aggregated quantity of `c[i]` values are for each coordinate `(x[i], y[i])`.
# 
# For example, given the following toy dataset:
# 
# | **x** | **y**  | **c** |
# |-------|--------|-------|
# | 1 | 0  | 3 |
# | 1 | 0  | 4 |
# | 1 | 0  | 5 |
# | 2 | 1  | 1 |
# | 2 | 1  | 2 |
# | 3 | -1 | 3 |
# 
# Assume the aggregate function we are using here is `np.size`, for each coordinate (x, y), we will be counting how many c values there are for that coordinate. Specifically,
# 
# - For the coordinate (x = 1, y = 0), we will have an aggregated value of 3 for c because there are three entires corresponding to (x = 1, y = 0).
# - For the coordinate (x = 2, y = 1), we will have an aggregated value of 2 for c.
# - For the coordinate (x = 3, y = -1) we will have an aggregated value of 1 for c.
# 
# These aggregated c values will be used to determine the intensity of the color that we assign to each hexigonal bin. It is also important to see that when the bins have the same size, counting the number of occurrences of c is equivalent to determining the density of c for each coordinate.
# 
# In the context of restaurant ratings, we can choose our `x[i]`, `y[i]`, `c[i]` values to be the longitude, latitude, and inspection score for each restaurant in San Francisco respectively. Since `x[i]` and `y[i]` also encode the geolocation of each restaurant, we can produce a geospatial hexbin plot that maps the density of scores to different locations within the city.
# 
# In order to produce the geospatial plot, we need to make sure we have all the data we need to create the plot. First, create a DataFrame `rated_geo` that includes the `longitude`, `latitude`, and `score` for each restaurant.
# 
# Hint: Note that not all the current data we have are actually valid. Some scores might be negative, and some longitude and latitudes are also invalid coordinates on Earth. Make sure to filter out those values in your resulting DataFrame.
# 
# Hint: Note that we are only concerned with the restaurant in the San Francisco region, so make sure that when you are filtering out the `latitude` and `longitude` columns, the range you provide in the flitering statement **makes sense** with the latitude and longitude of an actual location from San Francisco. **Don't worry too much about the how strict the bound needs to be**; as long as you cover all of San Francisco, you should be able to reproduce the same results we have for this question.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# points: 3
# -->

# In[137]:


#df = ins.merge(bus, how='left', on=['bid'])
df = ins.merge(bus, how='left', on=['bid']).merge(ins2vio).merge(vio) 
df = df[df['longitude'] > -122.6445]
df = df[df['longitude'] < -121.5871]
df = df[df['latitude'] < 38.2033]
final = df[df['latitude'] > 37.1897]
rated_geo = final[['longitude', 'latitude', 'score']]
rated_geo
#rated_geo.shape[0]


# In[138]:


ok.grade("q3a");


# <br/><br/><br/>
# 
# ---
# 
# ### Question 3b
# 
# Now that we have our DataFrame ready, we can start creating our geospatial hexbin plot.
# 
# Using the `rated_geo` DataFrame from 3a, produce a geospatial hexbin plot that shows the inspection count for all restaurant locations in San Francisco. 
# 
# Your plot should look similar to the one below:
# 
# ![](data/3a.png)
# 
# Hint: Use `pd.DataFrame.plot.hexbin()` or `plt.hexbin()` to create the hexbin plot.
# 
# Hint: For the 2 functions we mentioned above, try looking at the parameter `reduce_C_function`, which determines the aggregate function for the hexbin plot.
# 
# Hint: Use `fig.colorbar()` to create the color bar to the right of the hexbin plot.
# 
# Hint: Try using a `gridsize` of 200 when creating your hexbin plot; it makes the plot cleaner.
# 
# <!--
# BEGIN QUESTION
# name: q3b
# points: 4
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[139]:


# DO NOT MODIFY THIS BLOCK
min_lon = rated_geo['longitude'].min()
max_lon = rated_geo['longitude'].max()
min_lat = rated_geo['latitude'].min()
max_lat = rated_geo['latitude'].max()
max_score = rated_geo['score'].max()
min_score = rated_geo['score'].min()
bound = ((min_lon, max_lon, min_lat, max_lat))
min_lon, max_lon, min_lat, max_lat
map_bound = ((-122.5200, -122.3500, 37.6209, 37.8249))
# DO NOT MODIFY THIS BLOCK

# Read in the base map and setting up subplot
# DO NOT MODIFY THESE LINES
basemap = plt.imread('./data/sf.png')
fig, ax = plt.subplots(figsize = (11,11))
ax.set_xlim(map_bound[0],map_bound[1])
ax.set_ylim(map_bound[2],map_bound[3])
# DO NOT MODIFY THESE LINES


# Create the hexbin plot
p = ax.hexbin(rated_geo['longitude'], rated_geo['latitude'], gridsize=200, reduce_C_function=np.size, mincnt=1)
plt.xlabel('Longitude')
plt.ylabel("Latitude")
plt.title("Geospatial Density of Scores of Rated Restaurants")

fig.colorbar(p).set_label('Inspection Count')


# Setting aspect ratio and plotting the hexbins on top of the base map layer
# DO NOT MODIFY THIS LINE
ax.imshow(basemap, zorder=0, extent = map_bound, aspect= 'equal');
# DO NOT MODIFY THIS LINE


# <br/><br/><br/>
# 
# ---
# 
# ### Question 3c
# 
# Now that we've created our geospatial hexbin plot for the density of inspection scores for restaurants in San Francisco, let's also create another hexbin plot that visualizes the **average inspection scores** for restaurants in San Francisco.
# 
# Hint: If you set up everything correctly in 3b, you should only need to change 1 parameter here to produce the plot.
# 
# <!--
# BEGIN QUESTION
# name: q3c
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[140]:


# Read in the base map and setting up subplot
# DO NOT MODIFY THESE LINES
basemap = plt.imread('./data/sf.png')
fig, ax = plt.subplots(figsize = (11,11))
ax.set_xlim(map_bound[0],map_bound[1])
ax.set_ylim(map_bound[2],map_bound[3])
# DO NOT MODIFY THESE LINES

# Create the hexbin plot
p2 = ax.hexbin(rated_geo['longitude'], rated_geo['latitude'], rated_geo['score'], gridsize=200, reduce_C_function=np.mean, mincnt=1)
plt.xlabel('Longitude')
plt.ylabel("Latitude")
plt.title("Geospatial Average Scores of Rated Restaurants")

fig.colorbar(p2).set_label('Inspection Average')


# Setting aspect ratio and plotting the hexbins on top of the base map layer
# DO NOT MODIFY THIS LINE
ax.imshow(basemap, zorder=0, extent = map_bound, aspect= 'equal');
# DO NOT MODIFY THIS LINE


# <br/><br/><br/>
# 
# ---
# 
# ### Question 3d
# 
# Given the 2 hexbin plots you have just created above, did you notice any connection between the first plot where we aggregate over the **inspection count** and the second plot where we aggregate over the **inspection mean**? In several sentences, comment your observations in the cell below. 
# 
# Here're some of the questions that might be interesting to address in your response:
# 
# - Roughly speaking, did you notice any of the actual locations (districts/places of interest) where inspection tends to be more frequent? What about the locations where the average inspection score tends to be low?
# - Is there any connection between the locations where there are more inspections and the locations where the average inspection score is low?
# - What have might led to the connections that you've identified?
# 
# <!--
# BEGIN QUESTION
# name: q3d
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# There are a lot more inspections on the major streets of San Francisco. Streets like Mission, Van Ness, and Geary visibly have a lighter red hue to it, revealing a higher density of inspections per location. This is most likely the case because there is a much higher concentration of restaurants on these streets compared to the rest of SF, which as a result would tend towards more evaluations at these locations. 
# 
# It seems as though the distribution of average scores are pretty uniformly spread out throughout the city, with a few exceptions. There is a much high concentration of lower average scores in Chinatown and Sunset area, which honestly pains me to see, since I am of Chinese descent. Both of these areas have a much higher chinese demographic and thus have a higher concentration of Chinese restaurants. Speaking from experience, these restaurants are not necessarily the cleanest, which helps explain the low inspection scores. Furthermore, lower average scores seem to be more frequent on the outskirts of SF, which tend to be poorer neighborhoods. Due to the lack of wealth/support, these restaurants may have a harder time affording up to standard equipment and upkeep of their place, which again, would lead to lower inspection scores.
# 
# For the most part, there is a correlation between the number of inspections and the average inspection score. As we have discovered earlier, scores tend to increase from the first inspection to the next. In an area where there is a higher density of inspections, scores tend to be higher, compared to the outskirts, where there are a lot less inspections. Although this is one conclusion, the more obvious one may just be solely based on the distribution of wealth, where the businesses in the center of SF can actually afford to improve their conditions. 

# ## Summary of Inspections Data
# 
# We have done a lot in this project! Below are some examples of what we have learned about the inspections data through some cool visualizations!
# 
# - We found that the records are at the inspection level and that we have inspections for multiple years.
# - We also found that many restaurants have more than one inspection a year.
# - By joining the business and inspection data, we identified the name of the restaurant with the worst rating and optionally the names of the restaurants with the best rating.
# - We identified the restaurant that had the largest swing in rating over time.
# - We also examined the change of scores over time! Many restaurants are not actually doing better.
# - We created cool hexbin plots to relate the ratings with the location of restaurants! Now we know where to go if we want good food!

# <br/><br/><br/><br/>
# 
# ---
# 
# <br/><br/>
# 
# ## Question 4 Create some more cool visualizations!
# 
# <br/>

# It is your turn now! Play with the data, and try to produce some visualizations to answer one question that you find interesting regarding the data. You might want to use `merge`/`groupby`/`pivot` to process the data before creating visualizations.
# 
# Please show your work in the cells below (feel free to use extra cells if you want), and describe in words what you found in the same cell. This question will be graded leniently, but good solutions may be used to create future homework problems. 
# 
# 

# ### Grading ###
# 
# Since the assignment is more open ended, we will have a more relaxed rubric, classifying your answers into the following three categories:
# 
# - **Great** (4-5 points): The chart is well designed, and the data computation is correct. The text written articulates a reasonable metric and correctly describes the relevant insight and answer to the question you are interested in.
# - **Passing** (3-4 points): A chart is produced but with some flaws such as bad encoding. The text written is incomplete but makes some sense.
# - **Unsatisfactory** (<= 2 points): No chart is created, or a chart with completely wrong results.
# 
# We will lean towards being generous with the grading. We might also either discuss in discussion or post on Piazza some examplar analysis you have done (with your permission)!
# 
# You should have the following in your answers:
# * a few visualizations; Please limit your visualizations to 5 plots.
# * a few sentences (not too long please!)
# 
# Please note that you will only receive support in OH and Piazza for Matplotlib and seaborn questions. However, you may use some other Python libraries to help you create you visualizations. If you do so, make sure it is compatible with the PDF export (e.g., Plotly does not create PDFs properly, which we need for Gradescope).
# 
# <!--
# BEGIN QUESTION
# name: q4
# points: 5
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[141]:


# YOUR DATA PROCESSING AND PLOTTING HERE
merge_ins_bus = ins.merge(bus, how='left', on=['bid'])

# Since I am an avid foodie, I want to learn a little more about each neighborhood in SF. 
# First, I want to examine which zip code has the most inspections, and thus I can start to guage which 
# areas have the most traffic, thus would necessitate more inspections. (more restaurants = more inspections)
count = bus.groupby(['postal5']).size().sort_values()
plt.barh(count.keys(), count, edgecolor = 'black', height = .5)
plt.xlabel("Number of Inspections per Zip Code")
plt.ylabel("Zip Code")
plt.title("Distribution of Inspections Throughout SF")
plt.show()
# from this, I can tell94110 and 94103 are nearly tied with the most inspections, with over 550 a piece

# Next I want to know which zip code has the best average inspection rating. 
count = merge_ins_bus.groupby(['postal5'])['score'].mean().sort_values()
plt.barh(count.keys(), count, edgecolor = 'black', height = .5)
plt.xlabel("Number of Inspections per Zip Code")
plt.ylabel("Zip Code")
plt.title("Average Inspection Score Per Zip Code")
plt.show()
# This however does not reveal much. If we examine the zip code with the highest inspection rating, when comparing it to
# the total number of inspections, we see that there are virtually none. This most likely means that of the restaurant that
# was inspected, it was the only one in the zip code, which gives us an inaccurate reading of which zip code has the best
# average rating. Zip code 94129 is the Presidio, which would explain why there are so few inspections.

# Since less data points for a zip code tends to mean that there are simply less restaurants/less traffic, I wanted to see
# if it would be beneficial to examine only the zip codes with over 300 inspections. This would ensure to me that there
# is actually a reason to explore the area, since there would be lots of variety to choose from, whether it is good or bad
ax = sns.boxplot(x='postal5', 
                 y='score', data=final, 
                 hue='risk_category', 
                 hue_order=['Low Risk', 'Moderate Risk', 'High Risk'], 
                 order=['94102', '94110', '94103', '94107', '94133', '94109'])
plt.xlabel("Most Inspected Zip Codes")
plt.ylabel("Scores")
plt.title("Risk Categories for the 6 Most Inspected Zip Codes")
plt.figure(figsize=(12, 8))
# Judging from these box plots, I would most likely want to explore the 94107 zip code first. This is because the 
# interquartile range of both low and medium risk is the same, and is centered around a score of 90. This would give me
# me a high chance of finding a quality restaurant, especially since there are few outliers I would have to account for. 
# Furthermore, the entire range of low and medium risk (excluding the few outliers) is above a score of 70, which furthers
# my confidence in finding a quality restaurant. 94107 is Dogpatch which is a relatively nice neighborhood.

# The last thing I want to do, that is not quite related to the theme of this portion, is to see the number of inspections
# per year. It would be interesting to see whether there is an increase or a consistent amount of inspections regardless
# of year, and see if there is an explanation.
ins_year = ins.groupby('year').agg('count').reset_index()
plt.plot(ins_year['year'], ins_year['iid'])
plt.xticks(np.arange(2015, 2019, 1))
plt.xlabel('Year')
plt.ylabel('Number of Inspections')
plt.title('Number of Inspections Over Time')
plt.show()
# Looking at the line plot, there is a sharp increase in inspections from 2016 to 2017. This might be because there is
# just more data being recorded, or maybe it is a cause of the recent surge of instagram foodies. Or maybe even the rise
# of social media, and the ease of reporting.


# In[142]:


# THIS CELL AND ANY CELLS ADDED BELOW WILL NOT BE GRADED


# ## Congratulations! You have finished Part B of Project 1! ##

# # Submit
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.
# **Please save before submitting!**
# 
# <!-- EXPECT 12 EXPORTED QUESTIONS -->

# In[ ]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj1b.ipynb', 'proj1b.pdf')
ok.submit()


# In[ ]:




