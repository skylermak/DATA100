#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj2a.ok')


# # Project 2 Part A: Spam/Ham Classification
# ## EDA, Feature Engineering, Classifier
# ### The assignment is due on Monday, April 20th at 11:59pm PST.
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## This Assignment
# In project 2, you will use what you've learned in class to create a classifier that can distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails. In addition to providing some skeleton code to fill in, we will evaluate your work based on your model's accuracy and your written responses in this notebook.
# 
# After this project, you should feel comfortable with the following:
# 
# - Feature engineering with text data
# - Using sklearn libraries to process data and fit models
# - Validating the performance of your model and minimizing overfitting
# - Generating and analyzing precision-recall curves
# 
# In project 2A, you will undersatand the data through EDAs and do some basic feature engineerings. At the end, you will train your first logistic regression model to classify Spam/Ham emails. 
# 
# ## Warning
# We've tried our best to filter the data for anything blatantly offensive as best as we can, but unfortunately there may still be some examples you may find in poor taste. If you encounter these examples and believe it is inappropriate for students, please let a TA know and we will try to remove it for future semesters. Thanks for your understanding!

# ## Score Breakdown
# Question | Points
# --- | ---
# 1a | 1
# 1b | 1
# 1c | 2
# 2 | 3
# 3a | 2
# 3b | 2
# 4 | 2
# 5 | 2
# Total | 15

# In project 2a, we will try to undersatand the data and do some basic feature engineerings for classification.

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# ### Loading in the Data
# 
# In email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8348 labeled examples, and the test set contains 1000 unlabeled examples.
# 
# Run the following cells to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. You will predict labels for these emails and submit your predictions to Kaggle for evaluation.

# In[3]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()


# ### Question 1a
# First, let's check if our data contains any missing values. Fill in the cell below to print the number of NaN values in each column. If there are NaN values, replace them with appropriate filler values (i.e., NaN values in the `subject` or `email` columns should be replaced with empty strings). Print the number of NaN values in each column after this modification to verify that there are no NaN values left.
# 
# Note that while there are no NaN values in the `spam` column, we should be careful when replacing NaN labels. Doing so without consideration may introduce significant bias into our model when fitting.
# 
# *The provided test checks that there are no missing values in your dataset.*
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# -->

# In[4]:


original_training_data['subject'].fillna('',inplace=True)
original_training_data['email'].fillna('',inplace=True)
test['email'].fillna('',inplace=True)
test['subject'].fillna('',inplace=True)

print(original_training_data.isnull().sum())


# In[5]:


ok.grade("q1a");


# ### Question 1b
# 
# In the cell below, print the text of the first ham and the first spam email in the original training set.
# 
# *The provided tests just ensure that you have assigned `first_ham` and `first_spam` to rows in the data, but only the hidden tests check that you selected the correct observations.*
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 1
# -->

# In[6]:


first_ham = original_training_data[original_training_data['spam']==0]['email'].iloc[0]
first_spam = original_training_data[original_training_data['spam']==1]['email'].iloc[0]
print(first_ham)
print(first_spam)


# In[7]:


ok.grade("q1b");


# ### Question 1c
# 
# Discuss one thing you notice that is different between the two emails that might relate to the identification of spam.
# 
# <!--
# BEGIN QUESTION
# name: q1c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# It looks like the ham email has a nice little salutation/closer with a "thanks, misha" giving the impression that it was sent by an actual person, while in the spam, a closer is not present. Also, we can take a look at the choice of verbs the respected spam/ham emails use. The verb that jumps out to me the most is the word "guarantee". This word especially sounds like it would be from spam emails rather than ham emails, because it comes off as trying to market some product to someone else.

# ## Training Validation Split
# The training data we downloaded is all the data we have available for both training models and testing the models that we train.  We therefore need to split the training data into separate training and testing datsets. Note that we set the seed (random_state) to 42. This will produce a pseudo-random sequence of random numbers that is the same for every student. **Do not modify this in the following questions, as our tests depend on this random seed.**

# In[8]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(original_training_data, test_size=0.1, random_state=42)


# # Basic Feature Engineering
# 
# We would like to take the text of an email and predict whether the email is ham or spam. This is a *classification* problem, so we can use logistic regression to train a classifier. Recall that to train an logistic regression model we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$.  Unfortunately, our data are text, not numbers. To address this, we can create numeric features derived from the email text and use those features for logistic regression.
# 
# Each row of $X$ is an email. Each column of $X$ contains one feature for all the emails. We'll guide you through creating a simple feature, and you'll create more interesting ones when you are trying to increase your accuracy.

# ### Question 2
# 
# Create a function called `words_in_texts` that takes in a list of `words` and a pandas Series of email `texts`. It should output a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. For example:
# 
# ```
# >>> words_in_texts(['hello', 'bye', 'world'], 
#                    pd.Series(['hello', 'hello worldhello']))
# 
# array([[1, 0, 0],
#        [1, 0, 1]])
# ```
# 
# *The provided tests make sure that your function works correctly, so that you can use it for future questions.*
# 
# <!--
# BEGIN QUESTION
# name: q2
# points: 3
# -->

# In[9]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_array = [[word in text for word in words] for text in texts]
    return indicator_array


# In[10]:


ok.grade("q2");


# # Basic EDA
# 
# We need to identify some features that allow us to distinguish spam emails from ham emails. One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails. If the feature is itself a binary indicator, such as whether a certain word occurs in the text, this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.
# 

# The following plot (which was created using `sns.barplot`) compares the proportion of emails in each class containing a particular set of words. 
# 
# ![training conditional proportions](./images/training_conditional_proportions.png "Class Conditional Proportions")
# 
# Hint:
# - You can use DataFrame's `.melt` method to "unpivot" a DataFrame. See the following code cell for an example.

# In[11]:


from IPython.display import display, Markdown
df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(Markdown("> Our Original DataFrame has some words column and a type column. You can think of each row as a sentence, and the value of 1 or 0 indicates the number of occurances of the word in this sentence."))
display(df);
display(Markdown("> `melt` will turn columns into variale, notice how `word_1` and `word_2` become `variable`, their values are stored in the value column"))
display(df.melt("type"))


# ### Question 3a
# 
# Create a bar chart like the one above comparing the proportion of spam and ham emails containing certain words. Choose a set of words that are different from the ones above, but also have different proportions for the two classes. Make sure to only consider emails from `train`.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# points: 2
# manual: true
# image: true
# -->
# <!-- EXPORT TO PDF -->

# In[15]:


train=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts

set_of_words = ['thanks', 'buy', '#1', 'free', 'access']
ham = train[train['spam'] == 0]['email']
spam = train[train['spam'] == 1]['email']
ham_words = words_in_texts(set_of_words, ham)
spam_words = words_in_texts(set_of_words, spam)
prop_ham = np.sum(ham_words, axis=0)/len(ham)
prop_spam = np.sum(spam_words, axis=0)/len(spam)

plt.bar(x=set_of_words, align='edge', height=prop_ham, label='Ham', width=-0.35)
plt.bar(x=set_of_words, align='edge', height=prop_spam, label='Spam', width=0.35)
plt.legend()
plt.xlabel('Words')
plt.ylabel('Proportion of Emails')
plt.title('Frequency of Words in Spam/Ham Emails')


# When the feature is binary, it makes sense to compare its proportions across classes (as in the previous question). Otherwise, if the feature can take on numeric values, we can compare the distributions of these values for different classes. 
# 
# ![training conditional densities](./images/training_conditional_densities2.png "Class Conditional Densities")
# 

# ### Question 3b
# 
# Create a *class conditional density plot* like the one above (using `sns.distplot`), comparing the distribution of the length of spam emails to the distribution of the length of ham emails in the training set. Set the x-axis limit from 0 to 50000.
# 
# <!--
# BEGIN QUESTION
# name: q3b
# points: 2
# manual: true
# image: true
# -->
# <!-- EXPORT TO PDF -->

# In[13]:


len_of_spam = [len(i) for i in train[train['spam'] == 1]['email']]
len_of_ham = [len(i) for i in train[train['spam'] == 0]['email']]

plt.xlim(0, 50000)
sns.distplot(len_of_ham, label='Ham', hist=False)
sns.distplot(len_of_spam, label='Spam', hist=False)
plt.legend()
plt.ylabel('Distribution')
plt.xlabel('Length of email body')


# # Basic Classification
# 
# Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email. This means we can use it directly to train a classifier!

# ### Question 4
# 
# We've given you 5 words that might be useful as features to distinguish spam/ham emails. Use these words as well as the `train` DataFrame to create two NumPy arrays: `X_train` and `Y_train`.
# 
# `X_train` should be a matrix of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
# 
# `Y_train` should be a vector of the correct labels for each email in the training set.
# 
# *The provided tests check that the dimensions of your feature matrix (X) are correct, and that your features and labels are binary (i.e. consists of 0 and 1, no other values). It does not check that your function is correct; that was verified in a previous question.*
# <!--
# BEGIN QUESTION
# name: q4
# points: 2
# -->

# In[14]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = np.array(words_in_texts(some_words, train['email'])).astype(int)
Y_train = train['spam']

X_train[:5], Y_train[:5]


# In[15]:


ok.grade("q4");


# ### Question 5
# 
# Now that we have matrices, we can use to scikit-learn! Using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier, train a logistic regression model using `X_train` and `Y_train`. Then, output the accuracy of the model (on the training data) in the cell below. You should get an accuracy around 0.75.
# 
# *The provided test checks that you initialized your logistic regression model correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q5
# points: 2
# -->

# In[16]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)

training_accuracy = model.score(X_train, Y_train)
print("Training Accuracy: ", training_accuracy)


# In[17]:


ok.grade("q5");


# You have trained your first logistic regression model and it can correctly classify around 76% of the training data! Can we do better than this? The answer is yes! In project 2B, you will learn to evaluate your classifier. Moreover, you will have the chance to extract your own features and build your own classifier!

# ## Submission
# Congratulations! You are finished with this assignment. Please don't forget to submit by 11:59pm PST on Monday, 04/20!

# # Submit
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.
# **Please save before submitting!**
# 
# <!-- EXPECT 3 EXPORTED QUESTIONS -->

# In[18]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj2a.ipynb', 'proj2a.pdf')
ok.submit()

