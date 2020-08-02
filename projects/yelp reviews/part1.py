#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj1a.ok')


# # Project 1A: Food Safety
# ## Cleaning and Exploring Data with Pandas
# ## Due Date: Monday 02/17, 11:59 PM
# ## Collaboration Policy
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## Before You Start
# 
# For each question in the assignment, please write down your answer in the answer cell(s) right below the question. 
# 
# We understand that it is helpful to have extra cells breaking down the process towards reaching your final answer. If you happen to create new cells below your answer to run codes, **NEVER** add cells between a question cell and the answer cell below it. It will cause errors when we run the autograder, and it will sometimes cause a failure to generate the PDF file.
# 
# **Important note: The local autograder tests will not be comprehensive. You can pass the automated tests in your notebook but still fail tests in the autograder.** Please be sure to check your results carefully.
# 
# Finally, unless we state otherwise, try to avoid using python for loops or list comprehensions.  The majority of this assignment can be done using builtin commands in Pandas and numpy.  
# 

# In[2]:


import numpy as np
import pandas as pd 

import bz2 # Used to read compressed data
import os # Used to interact with the file system


# ## Obtaining the Data
# 
# ### File Systems and I/O
# 
# 

# In general, we will focus on using python commands to investigate files.  However, it can sometimes be easier to use shell commands in your local operating system.  The following cells demonstrate how to do this.
# 
# 
# The command below starts with `!`. This tells our Jupyter notebook to pass this command to the operating system. In this case, the command is the `ls` POSIX command which lists all files in the current directory. Note what `!ls data` outputs.  
# 
# *Note this `!` only works in ipython shells (Jupyter Notebooks). And the `ls` (list) command only works in posix environments and may not work on default Windows systems.*

# In[3]:


get_ipython().system('ls ')


# In[4]:


get_ipython().system('ls data')


# We are going to use the [`pathlib`](https://docs.python.org/3/library/pathlib.html#basic-use) module to represent our file system paths and perform operations which allow us to learn more about the contents of our data. Note what `pathlib.Path.cwd()` outputs in relation to the output of `!ls` above.

# <br/><br/><br/>
# 
# ---
# 
# <br/><br/><br/>
# 
# # 1: Examining the Files
# Let's first focus on understanding the structure of the data; this involves answering questions such as:
# 
# * How much data do we have?
# * Is the data in a standard format or encoding?
# * Is the data organized in records?
# * What are the fields in each record?
# 
# Let's start by looking at the contents of `data/`. This is not just a single file but rather a directory of multiple compressed files. We could inspect our data by uncompressing each file but in this project we're going to do almost everything in Python for maximum portability.
# 

# You will want to use a few useful python functions.  To move through the local filesystem you can use the `Path` module in `pathlinb`.  For example, to list the current directory you can [Path.cwd](https://docs.python.org/3/library/pathlib.html#pathlib.Path.cwd).

# In[5]:


from pathlib import Path

Path.cwd()


# The function returns a `pathlib.Path` object representing the location of the file.  It can also be used to list contents of directories and many other things. 
# 
# You will also need to work with `bzip2` files and you will want to be able to read their contents using the [bz2](https://docs.python.org/3/library/bz2.html) python library.

# In[6]:


with bz2.open("data/bus.csv.bz2", "r") as f:
    print("The first line:", "\n\t", f.readline())


# <br/><br/><br/>
# 
# ---
# 
# ## Question 1a:
# 
# Implement the `list_files`, `get_file_size`, and `get_linecount_bz2` functions to return the list of files in the directory, the sizes (in bytes) of a file, and the number of lines in the file.  Note the last `get_linecount_bz2` should not produce any intermediate files in the filesystem and should avoid storing the entire file in memory (don't do `len(file.readlines())`).
# 
# **Hints:** *You might find the following documentation useful:*
# 1. [Python pathlib](https://docs.python.org/3.7/library/pathlib.html#pathlib.Path.iterdir)
# 1. [bz2](https://docs.python.org/3/library/bz2.html#bz2.open)
# 
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 3
# -->

# In[7]:


def list_files(directory):
    """
    Return a list of pathlib.Path objects for the files in the directory.
    
    directory: a string describing the directory to list 
        for example 'data/'
    """
    path_obj = Path(directory)
    return [i for i in path_obj.iterdir() if not i.is_dir()]
    
    
def get_file_size(file_name):
    """
    Return file size for a given filename.
    """ 
    dir = Path(file_name)
    return dir.stat().st_size

def get_linecount_bz2(file_name):
    """
    Returns the number of lines in bz2 file.  
    """ 
    size = 0
    with bz2.BZ2File(file_name, 'r') as f:
        while f.readline():
            size += 1
    return size

list_files('data')


# In[8]:


ok.grade("q1a");


# Now, let's see the file size and the number of lines for each data file.  If you implemented the above code correctly the following cell should produce the following (the columns may be in a different order):
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#        <th>linecount</th>
#       <th>name</th>
#       <th>size</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>2</th>
#        <td>66</td>
#       <td>data/vio.csv.bz2</td>
#       <td>1337</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>26664</td>
#       <td>data/ins.csv.bz2</td>
#       <td>110843</td>
#     </tr>
#     <tr>
#       <th>0</th>
#       <td>6254</td>
#       <td>data/bus.csv.bz2</td>
#       <td>113522</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>40211</td>
#       <td>data/ins2vio.csv.bz2</td>
#       <td>146937</td>
#     </tr>
#   </tbody>
# </table>

# In[9]:


info = []
for f in list_files("data/"):
    name = str(f)
    if name[-3:] == "bz2": 
        size = get_file_size(f)
        linecount = get_linecount_bz2(f)
        info.append({"name": name, "size": size, "linecount": linecount})

file_info = pd.DataFrame(info).sort_values("size")
file_info


# <br/><br/><br/>
# 
# ---
# 
# ## Question 1b: Programatically Looking Inside the Files
# 
# Implement the following function `head_bz2` to return a list of the first `nlines` lines of each file. 
# Using your `head_bz2` function implement the following `print_head_bz2` function that uses `print()` to print the filename followed by the first `nlines` of each file and their line numbers in the following format.
# 
# ```
# data/bus.csv.bz2
# 0 :	 b'"business id column","name","address","city","state","postal_code","latitude","longitude","phone_number"\n'
# 1 :	 b'"1000","HEUNG YUEN RESTAURANT","3279 22nd St","San Francisco","CA","94110","37.755282","-122.420493","-9999"\n'
# 2 :	 b'"100010","ILLY CAFFE SF_PIER 39","PIER 39  K-106-B","San Francisco","CA","94133","-9999","-9999","+14154827284"\n'
# 3 :	 b'"100017","AMICI\'S EAST COAST PIZZERIA","475 06th St","San Francisco","CA","94103","-9999","-9999","+14155279839"\n'
# 4 :	 b'"100026","LOCAL CATERING","1566 CARROLL AVE","San Francisco","CA","94124","-9999","-9999","+14155860315"\n'
# ```
# 
# Do not read the entire file contents!
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 2
# -->

# In[10]:


def head_bz2(file, nlines=5):
    """
    Return a list of the first nlines lines of filename
    """
    with bz2.BZ2File(file, 'r') as f:
        first_nlines = [f.readline() for i in range(nlines)]
    return first_nlines

def print_head_bz2(file, nlines=5):
    """
    Print a list of the first nlines lines of filename
    """
    print(file)
    for i in np.arange(nlines):
        x = head_bz2(file)[i]
        print(str(i) + ':     ' + str(x))
    
print_head_bz2('data/bus.csv.bz2')


# In[11]:


ok.grade("q1b");


# The following should display the filename and head of the file for all the files in data:

# In[12]:


for file in list_files("data/"):
    if str(file)[-3:] == "bz2":  
        print_head_bz2(file)
        print()


# <br/><br/><br/>
# 
# ---
# 
# ## Question 1c: Thinking about the files
# 
# Answer the following questions by filling in the correct boolean values in the following variables:
# 
# 1. The bus.csv.bz2 file appears to be tab delimited.
# 1. The values all appear to be quoted.
# 
# 
# <!--
# BEGIN QUESTION
# name: q1c
# points: 1
# -->

# In[13]:


# True or False: The bus.csv.bz2 file appears to be tab delimited.
q1c_1 = False

# True or False: The values all appear to be quoted.
q1c_2 = True


# In[14]:


ok.grade("q1c");


# <br/><br/><br/>
# 
# ---
# 
# ## Question 1d: Reading in the Files
# 
# Based on the above information, let's attempt to load `bus.csv`, `ins2vio.csv`, `vio.csv` and `ins.csv` into Pandas dataframes with the following names: `bus`, `ins2vio`, `vio`, `ins` respectively.
# 
# 
# <!--
# BEGIN QUESTION
# name: q1d
# points: 1
# -->

# In[15]:


# path to directory containing data
bus = pd.read_csv(bz2.BZ2File('data/bus.csv.bz2'))
ins = pd.read_csv(bz2.BZ2File('data/ins.csv.bz2'))
ins2vio = pd.read_csv(bz2.BZ2File('data/ins2vio.csv.bz2'))
vio = pd.read_csv(bz2.BZ2File('data/vio.csv.bz2'))


# In[16]:


ok.grade("q1d");


# Now that you've read in the files, you can try some `pd.DataFrame` methods ([docs](https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.html)).
# You can use the `DataFrame.head` method to show the top few lines of the `bus`, `ins`, `ins2vio` and `vio` dataframes. 

# In[17]:


bus.head()


# In[18]:


ins.head()


# In[19]:


ins2vio.head()


# In[20]:


vio.head()


# <br/><br/><br/>
# 
# ---
# 
# ## Question 1e: Identifying Issues with the Data

# Use the `head` command on your four files again. This time, describe at least one potential problem with the data you see. Consider issues with missing values and bad data.
# 
# Please write your answer in the markdown cell below. You may create new cells below your answer to run code, but **please never add cells between a question cell and the answer cell below it.**
# 
# <!--
# BEGIN QUESTION
# name: q1e
# manual: True
# points: 1
# -->
# <!-- EXPORT TO PDF -->

# One immediate flaw in our data is that the location of the business names are invalid. For business 100010, 100017, 100026, 100030, their lat and long is -9999, which can be interpretted to mean that their location is not logged. This can pose a problem later on in our analysis, such as if we wanted to visualize the data on a map. Because their location is invalid, they would not be plotted on the map, resulting in a incomplete display of information.
# Furthermore, in the violation (vio) table, many businesses have a score of -1, which is an unnatural value to be assigned, especially in the context of a score. This may mean that the business no longer exists, or just had never recieved a score, which leave some unclarity for the data scientist analyzing this data.

# In[21]:


bus.head()
ins.head()
ins2vio.head()
vio.head()


# <br/><br/><br/>
# 
# ---
# 
# <br/><br/><br/>
# 
# # 2: Examining the Business Data File
# 
# From its name alone, we expect the `bus.csv` file to contain information about the restaurants. Let's investigate the granularity of this dataset.

# In[22]:


bus.head()


# <br/><br/><br/>
# 
# ---
# 
# ## Question 2a
# 
# The `bus` dataframe contains a column called `business id column` which probably corresponds to a unique business id.  However, let's first rename that column to `bid`.  Modify the `bus` dataframe by renaming that column to `bid`.
# 
# **Note**: In practice we might want to do this renaming when the table is loaded but for grading purposes we will do it here.
# 
# <!--
# BEGIN QUESTION
# name: q2a
# points: 1
# -->

# In[23]:


bus = bus.rename(columns = {"business id column": "bid"})


# In[24]:


ok.grade("q2a");


# <br/><br/><br/>
# 
# ---
# 
# ## Question 2b
# 
# Examining the entries in `bus`, is the `bid` unique for each record (i.e. each row of data)? Your code should compute the answer, i.e. don't just hard code `True` or `False`.
# 
# Hint: use `value_counts()` or `unique()` to determine if the `bid` series has any duplicates.
# 
# <!--
# BEGIN QUESTION
# name: q2b
# points: 1
# -->

# In[25]:


is_bid_unique = pd.Series(bus['bid']).is_unique
is_bid_unique


# In[26]:


ok.grade("q2b");


# <br/><br/><br/>
# 
# ---
# 
# 
# ## Question 2c
# 
# In the two cells below create two **series** 
# 
# 1. where the index is the `name` of the business and the value is the number of records with that `name`
# 2. where the index is the `address` of the business and the value is the number of records with that `address`
# 
# Order both series in descending order by count. You may need to use `groupby()`, `size()`, `sort_values()`, or `value_counts()`. 
# 
# **Step 1**
# 
# <!--
# BEGIN QUESTION
# name: q2ci
# points: 1
# -->

# In[27]:


name_counts = bus['name'].value_counts()
name_counts.head(20)


# In[28]:


ok.grade("q2ci");


# **Step 2**
# 
# <!--
# BEGIN QUESTION
# name: q2cii
# points: 1
# -->

# In[29]:


address_counts = name_counts = bus['address'].value_counts()
address_counts.head(10)


# In[30]:


ok.grade("q2cii");


# <br/><br/><br/>
# 
# ---
# 
# 
# ## Question 2d
# 
# Based on the above calculations answer each of the following questions by filling the value in the variable.
# 
# 1. What does each record represent?  
# 1. What is the minimal primary key?
# 
# <!--
# BEGIN QUESTION
# name: q2d
# points: 2
# -->

# In[31]:


# What does each record represent?  Valid answers are:
#    "One location of a restaurant."
#    "A chain of restaurants."
#    "A city block."
q2d_part1 = "One location of a restaurant."

# What is the minimal primary key? Valid answers are:
#    "bid"
#    "bid, name"
#    "bid, name, address"
q2d_part2 = "bid"


# In[32]:


ok.grade("q2d");


# <br/><br/><br/>
# 
# ---
# 
# <br/><br/><br/>
# 
# # 3: Cleaning the Business Data Postal Codes
# 
# The business data contains postal code information that we can use to aggregate the ratings over regions of the city.  Let's examine and clean the postal code field.  The postal code (sometimes also called a ZIP code) partitions the city into regions:
# 
# <img src="https://www.usmapguide.com/wp-content/uploads/2019/03/printable-san-francisco-zip-code-map.jpg" alt="ZIP Code Map" style="width: 600px">

# <br/><br/><br/>
# 
# ---
# 
# 
# ## Question 3a
# 
# How many restaurants are in each ZIP code? 
# 
# In the cell below, create a **series** where the index is the postal code and the value is the number of records with that postal code in descending order of count. You may need to use `groupby()`, `size()`, or `value_counts()`. Do you notice any odd/invalid zip codes?
# 
# <!--
# BEGIN QUESTION
# name: q3a
# points: 1
# -->

# In[33]:


zip_counts = bus['postal_code'].value_counts()
print(zip_counts.to_string())


# In[34]:


ok.grade("q3a");


# <br/><br/><br/>
# 
# --- 
# 
# ## Question 3b
# 
# Answer the following questions about the `postal_code` column in the `bus` dataframe.
# 
# 1. The ZIP code column is which of the following type of data:
#     1. Quantitative Continuous
#     1. Quantitative Discrete
#     1. Qualitative Ordinal
#     1. Qualitative Nominal    
# 1. What Python data type is used to represent a ZIP code?
# 
# *Note*: ZIP codes and postal codes are the same thing.
# 
# Please write your answers in the variables below:
# 
# <!--
# BEGIN QUESTION
# name: q3b
# points: 2
# -->

# In[35]:


# The ZIP code column is which of the following type of data:
#   "Quantitative Continuous" 
#   "Quantitative Discrete"
#   "Qualitative Ordinal"
#   "Qualitative Nominal"
q3b_part1 = "Qualitative Nominal"

# What Python data type is used to represent a ZIP code? 
#    "str"
#    "int"
#    "bool"
#    "float"
q3b_part2 = "str"


# In[36]:


ok.grade("q3b");


# <br/><br/><br/>
# 
# --- 
# 
# ## Question 3c
# 
# In question 3a we noticed a large number of potentially invalid ZIP codes (e.g., "CA").  These are likely due to data entry errors.  To get a better understanding of the potential errors in the zip codes we will:
# 
# 1. Import a list of valid San Francisco ZIP codes by using `pd.read_json` to load the file `data/sf_zipcodes.json` and extract a **series** of type `str` containing the valid ZIP codes.  *Hint: set `dtype` when invoking `read_json`.*
# 1. Construct a DataFrame containing only the businesses which DO NOT have valid ZIP codes.  You will probably want to use the `Series.isin` function. 
# 

# **Step 1**
# 
# <!--
# BEGIN QUESTION
# name: q3ci
# points: 1
# -->

# In[37]:


valid_zips = pd.read_json('data/sf_zipcodes.json', dtype=str)['zip_codes']
valid_zips.head()


# In[38]:


ok.grade("q3ci");


# **Step 2**
# 
# <!--
# BEGIN QUESTION
# name: q3cii
# points: 1
# -->

# In[39]:


invalid_zip_bus = bus[~bus['postal_code'].isin(valid_zips)]
invalid_zip_bus[invalid_zip_bus['postal_code'] == 'CA']
invalid_zip_bus.head(20)


# In[40]:


ok.grade("q3cii");


# <br/><br/><br/>
# 
# --- 
# 
# ## Question 3d
# 
# In the previous question, many of the businesses had a common invalid postal code that was likely used to code a MISSING postal code.  Do they all share a potentially "interesting address"?
# 
# In the following cell, construct a **series** that counts the number of businesses at each `address` that have this single likely MISSING postal code value.  Order the series in descending order by count. 
# 
# After examining the output.  Answer the following question by filling in the appropriate variable. If we were to drop businesses with MISSING postal code values would a particular class of business be affected?  If you are unsure try to search the web for the most common addresses.
# 
# 
# <!--
# BEGIN QUESTION
# name: q3d
# points: 3
# -->

# In[41]:


missing = invalid_zip_bus['postal_code'] == "-9999"
missing_zip_address_count = invalid_zip_bus[missing]['address'].value_counts()
missing_zip_address_count.head(30)


# In[42]:


ok.grade("q3d");


# <br/><br/><br/>
# 
# --- 
# 
# ## Question 3e
# 
# **True or False**:  *If we were to drop businesses with MISSING postal code values a particular class of business will be affected.*
# 
# <!--
# BEGIN QUESTION
# name: q3e
# points: 2
# -->

# In[43]:


# True or False: 
#  If we were to drop businesses with MISSING postal code values 
#   a particular class of business be affected.
q3d_true_or_false = True


# In[44]:


ok.grade("q3e");


# <br/><br/><br/>
# 
# --- 
# 
# ## Question 3f
# 
# Examine the `invalid_zip_bus` dataframe we computed above and look at the businesses that DO NOT have the special MISSING ZIP code value.  Some of the invalid postal codes are just the full 9 digit code rather than the first 5 digits.  Create a new column named `postal5` in the original `bus` dataframe which contains only the first 5 digits of the `postal_code` column.   Finally, for any of the `postal5` ZIP code entries that were not a valid San Fransisco ZIP Code (according to `valid_zips`) set the entry to `None`.  
# 
# 
# <!--
# BEGIN QUESTION
# name: q3f
# points: 2
# -->

# In[45]:


bus['postal5'] = None
bus['postal5'] = bus['postal_code'].str[:5]
valid_zips_list = valid_zips.tolist()
bus['postal5'] = bus['postal5'].apply(lambda i: i if i in valid_zips_list else None)

# Checking the corrected postal5 column
bus.loc[invalid_zip_bus.index, ['bid', 'name', 'postal_code', 'postal5']]


# In[46]:


ok.grade("q3f");


# <br/><br/><br/>
# 
# ---
# 
# <br/><br/><br/>
# 
# # 4: Investigate the Inspection Data
# 
# Let's now turn to the inspection DataFrame. Earlier, we found that `ins` has 4 columns named 
# `iid`, `score`, `date` and `type`.  In this section, we determine the granularity of `ins` and investigate the kinds of information provided for the inspections. 

# Let's start by looking again at the first 5 rows of `ins` to see what we're working with.

# In[47]:


ins.head(5)


# <br/><br/><br/>
# 
# ---
# 
# ## Question 4a
# 
# The column `iid` probably corresponds to an inspection id.  Is it a primary key?  Write an expression (line of code) that evaluates to 'True' or 'False' based on whether all the values are unique.
# 
# 
# <!--
# BEGIN QUESTION
# name: q4a
# points: 1
# -->

# In[48]:


is_ins_iid_a_primary_key = pd.Series(ins['iid']).is_unique


# In[49]:


ok.grade("q4a");


# <br/><br/><br/>
# 
# ---
# 
# ## Question 4b
# 
# The column `iid` appears to be the composition of two numbers and the first number looks like a business id.  
# 
# **Part 1.**: Create a new column called `bid` in the `ins` dataframe containing just the business id.  You will want to use `ins['iid'].str` operations to do this.  Also be sure to convert the type of this column to `int`
# 
# **Part 2.**: Then compute how many values in this new column are also valid business ids (appear in the `bus['bid']` column). This is verifying a foreign key relationship. Consider using the `pd.Series.isin` function.
# 
# **Part 3.**: Answer True or False, `ins['bid']` is a foreign key reference to `bus['bid']`.
# 
# 
# **No python `for` loops or list comprehensions required!**

# **Part 1**
# 
# <!--
# BEGIN QUESTION
# name: q4bi
# points: 1
# -->

# In[50]:


ins['bid'] = pd.to_numeric(ins['iid'].str.split('_', 1).str[0])
ins


# In[51]:


ok.grade("q4bi");


# **Part 2**
# 
# <!--
# BEGIN QUESTION
# name: q4bii
# points: 1
# -->

# In[53]:


invalid_bid_count = len(ins['bid'])-((ins['bid'].isin(bus['bid'])).sum())
invalid_bid_count


# In[54]:


ok.grade("q4bii");


# **Part 3**
# 
# <!--
# BEGIN QUESTION
# name: q4biii
# points: 1
# -->

# In[56]:


# True or False: The column ins['bid'] is a foreign key 
#   referencing the bus['bid'] primary key.

q4b_is_foreign_key = True


# In[57]:


ok.grade("q4biii");


# <br/><br/><br/>
# 
# ---
# 
# ## Question 4c
# 
# What if we are interested in a time component of the inspection data?  We need to examine the date column of each inspection. 
# 
# **Part 1:** What is the type of the individual `ins['date']` entries. You may want to grab the very first entry and use the `type` function in python. 
# 
# **Part 2:** Use `pd.to_datetime` to create a new `ins['timestamp']` column containing of `pd.Timestamp` objects.  These will allow us to do more date manipulation.
# 
# **Part 3:** What are the earliest and latest dates in our inspection data?  *Hint: you can use `min` and `max` on dates of the correct type.*
# 
# **Part 4:** We probably want to examine the inspections by year. Create an additional `ins['year']` column containing just the year of the inspection.  Consider using `pd.Series.dt.year` to do this.
# 
# **No python `for` loops or list comprehensions required!**

# **Part 1**
# 
# <!--
# BEGIN QUESTION
# name: q4ci
# points: 1
# -->

# In[58]:


ins_date_type = type(ins['date'][0])
ins_date_type


# In[59]:


ok.grade("q4ci");


# **Part 2**
# 
# <!--
# BEGIN QUESTION
# name: q4cii
# points: 1
# -->

# In[60]:


ins['timestamp'] = pd.to_datetime(ins['date'])


# In[61]:


ok.grade("q4cii");


# **Part 3**
# 
# <!--
# BEGIN QUESTION
# name: q4ciii
# points: 1
# -->

# In[62]:


earliest_date = min(ins['timestamp'])
latest_date = max(ins['timestamp'])

print("Earliest Date:", earliest_date)
print("Latest Date:", latest_date)


# In[63]:


ok.grade("q4ciii");


# **Part 4**
# 
# <!--
# BEGIN QUESTION
# name: q4civ
# points: 1
# -->

# In[64]:


ins['year'] = (ins['timestamp']).dt.year
ins


# In[65]:


ok.grade("q4civ");


# In[66]:


ins.head()


# <br/><br/><br/>
# 
# ---
# 
# ## Question 4d
# 
# What is the relationship between the type of inspection over the 2016 to 2019 timeframe? 
# 
# **Part 1**
# 
# Construct the following table by
# 1. Using the `pivot_table` containing the number (`size`) of inspections for the given `type` and `year`.
# 1. Adding an extra `Total` column to the result using `sum`
# 1. Sort the results in descending order by the `Total`.
# 
# <table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th>year</th>      <th>2016</th>      <th>2017</th>      <th>2018</th>      <th>2019</th>      <th>Total</th>    </tr>    <tr>      <th>type</th>      <th></th>      <th></th>      <th></th>      <th></th>      <th></th>    </tr>  </thead>  <tbody>    <tr>      <th>Routine - Unscheduled</th>      <td>966</td>      <td>4057</td>      <td>4373</td>      <td>4681</td>      <td>14077</td>    </tr>    <tr>      <th>Reinspection/Followup</th>      <td>445</td>      <td>1767</td>      <td>1935</td>      <td>2292</td>      <td>6439</td>    </tr>    <tr>      <th>New Ownership</th>      <td>99</td>      <td>506</td>      <td>528</td>      <td>459</td>      <td>1592</td>    </tr>    <tr>      <th>Complaint</th>      <td>91</td>      <td>418</td>      <td>512</td>      <td>437</td>      <td>1458</td>    </tr>    <tr>      <th>New Construction</th>      <td>102</td>      <td>485</td>      <td>218</td>      <td>189</td>      <td>994</td>    </tr>    <tr>      <th>Non-inspection site visit</th>      <td>51</td>      <td>276</td>      <td>253</td>      <td>231</td>      <td>811</td>    </tr>    <tr>      <th>New Ownership - Followup</th>      <td>0</td>      <td>45</td>      <td>219</td>      <td>235</td>      <td>499</td>    </tr>    <tr>      <th>Structural Inspection</th>      <td>1</td>      <td>153</td>      <td>50</td>      <td>190</td>      <td>394</td>    </tr>    <tr>      <th>Complaint Reinspection/Followup</th>      <td>19</td>      <td>68</td>      <td>70</td>      <td>70</td>      <td>227</td>    </tr>    <tr>      <th>Foodborne Illness Investigation</th>      <td>1</td>      <td>29</td>      <td>50</td>      <td>35</td>      <td>115</td>    </tr>    <tr>      <th>Routine - Scheduled</th>      <td>0</td>      <td>9</td>      <td>8</td>      <td>29</td>      <td>46</td>    </tr>    <tr>      <th>Administrative or Document Review</th>      <td>2</td>      <td>1</td>      <td>1</td>      <td>0</td>      <td>4</td>    </tr>    <tr>      <th>Multi-agency Investigation</th>      <td>0</td>      <td>0</td>      <td>1</td>      <td>2</td>      <td>3</td>    </tr>    <tr>      <th>Special Event</th>      <td>0</td>      <td>3</td>      <td>0</td>      <td>0</td>      <td>3</td>    </tr>    <tr>      <th>Community Health Assessment</th>      <td>1</td>      <td>0</td>      <td>0</td>      <td>0</td>      <td>1</td>    </tr>  </tbody></table>
# 
# 
# **Part 2**
# 
# Based on the above analysis, which year appears to have had a lot of businesses in new buildings?
# 
# **No python `for` loops or list comprehensions required!**

# **Part 1**
# 
# <!--
# BEGIN QUESTION
# name: q4di
# points: 2
# -->

# In[69]:


ins_pivot = pd.pivot_table(ins, index=['type'], columns=['year'], aggfunc=np.size, fill_value=0)['bid']
ins_pivot['Total'] = ins_pivot.sum(axis=1)

ins_pivot_sorted = ins_pivot.sort_values(by='Total', ascending=False)

ins_pivot_sorted


# In[70]:


ok.grade("q4di");


# **Part 2**
# 
# <!--
# BEGIN QUESTION
# name: q4dii
# points: 2
# -->

# In[71]:


year_of_new_construction = 2017


# In[72]:


ok.grade("q4dii");


# <br/><br/><br/>
# 
# ---
# 
# ## Question 4e
# 
# Let's examine the inspection scores `ins['score']`
# 
# 

# In[73]:


ins['score'].value_counts().head()


# There are a large number of inspections with the `'score'` of `-1`.   These are probably missing values.  Let's see what type of inspections have scores and which do not.  Create the following dataframe using steps similar to the previous question.
# 
# You should observe that inspection scores appear only to be assigned to `Routine - Unscheduled` inspections.
# 
# 
# <table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th>Missing Score</th>      <th>False</th>      <th>True</th>      <th>Total</th>    </tr>    <tr>      <th>type</th>      <th></th>      <th></th>      <th></th>    </tr>  </thead>  <tbody>    <tr>      <th>Routine - Unscheduled</th>      <td>14031</td>      <td>46</td>      <td>14077</td>    </tr>    <tr>      <th>Reinspection/Followup</th>      <td>0</td>      <td>6439</td>      <td>6439</td>    </tr>    <tr>      <th>New Ownership</th>      <td>0</td>      <td>1592</td>      <td>1592</td>    </tr>    <tr>      <th>Complaint</th>      <td>0</td>      <td>1458</td>      <td>1458</td>    </tr>    <tr>      <th>New Construction</th>      <td>0</td>      <td>994</td>      <td>994</td>    </tr>    <tr>      <th>Non-inspection site visit</th>      <td>0</td>      <td>811</td>      <td>811</td>    </tr>    <tr>      <th>New Ownership - Followup</th>      <td>0</td>      <td>499</td>      <td>499</td>    </tr>    <tr>      <th>Structural Inspection</th>      <td>0</td>      <td>394</td>      <td>394</td>    </tr>    <tr>      <th>Complaint Reinspection/Followup</th>      <td>0</td>      <td>227</td>      <td>227</td>    </tr>    <tr>      <th>Foodborne Illness Investigation</th>      <td>0</td>      <td>115</td>      <td>115</td>    </tr>    <tr>      <th>Routine - Scheduled</th>      <td>0</td>      <td>46</td>      <td>46</td>    </tr>    <tr>      <th>Administrative or Document Review</th>      <td>0</td>      <td>4</td>      <td>4</td>    </tr>    <tr>      <th>Multi-agency Investigation</th>      <td>0</td>      <td>3</td>      <td>3</td>    </tr>    <tr>      <th>Special Event</th>      <td>0</td>      <td>3</td>      <td>3</td>    </tr>    <tr>      <th>Community Health Assessment</th>      <td>0</td>      <td>1</td>      <td>1</td>    </tr>  </tbody></table>
# 
# 

# <!--
# BEGIN QUESTION
# name: q4e
# points: 2
# -->

# In[74]:


ins['Missing Score'] = (ins['score'] == -1)
ins_missing_score_pivot = pd.pivot_table(ins, index=['type'], columns=['Missing Score'], aggfunc=np.size, fill_value=0)['bid']
ins_missing_score_pivot['Total'] = ins_missing_score_pivot.sum(axis=1)

ins_missing_score_pivot = ins_missing_score_pivot.sort_values(by='Total', ascending=False)
ins_missing_score_pivot[[True, False, 'Total']]

ins_missing_score_pivot


# In[75]:


ok.grade("q4e");


# Notice that inspection scores appear only to be assigned to `Routine - Unscheduled` inspections. It is reasonable that for inspection types such as `New Ownership` and `Complaint` to have no associated inspection scores, but we might be curious why there are no inspection scores for the `Reinspection/Followup` inspection type.

# <br/><br/><br/>
# 
# ---
# 
# <br/><br/><br/>
# 
# # 5: Joining Data Across Tables
# 
# In this question we will start to connect data across mulitple tables.  We will be using the `merge` function. 

# <br/><br/><br/>
# 
# --- 
# 
# ## Question 5a
# 
# Let's figure out which restaurants had the lowest scores. Let's start by creating a new dataframe called `ins_named`. It should be exactly the same as `ins`, except that it should have the name and address of every business, as determined by the `bus` dataframe. 
# 
# *Hint*: Use the merge method to join the `ins` dataframe with the appropriate portion of the `bus` dataframe. See the official [documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html) on how to use `merge`.
# 
# *Note*: For quick reference, a pandas 'left' join keeps the keys from the left frame, so if ins is the left frame, all the keys from ins are kept and if a set of these keys don't have matches in the other frame, the columns from the other frame for these "unmatched" key rows contains NaNs.
# 
# <!--
# BEGIN QUESTION
# name: q5a
# points: 1
# -->

# In[77]:


ins_named = ins.merge(bus[['bid', 'name', 'address']], how='left', on='bid')
ins_named.head()


# In[78]:


ok.grade("q5a");


# <br/><br/><br/>
# 
# --- 
# 
# ## Question 5b
# 
# Let's look at the 20 businesses with the lowest **median** score.  Order your results by the median score followed by the business id to break ties. The resulting table should look like:
# 
# 
# *Hint: You may find the `as_index` argument important*
# 
# <table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>bid</th>      <th>name</th>      <th>median score</th>    </tr>  </thead>  <tbody>    <tr>      <th>3876</th>      <td>84590</td>      <td>Chaat Corner</td>      <td>54.0</td>    </tr>    <tr>      <th>4564</th>      <td>90622</td>      <td>Taqueria Lolita</td>      <td>57.0</td>    </tr>    <tr>      <th>4990</th>      <td>94351</td>      <td>VBowls LLC</td>      <td>58.0</td>    </tr>    <tr>      <th>2719</th>      <td>69282</td>      <td>New Jumbo Seafood Restaurant</td>      <td>60.5</td>    </tr>    <tr>      <th>222</th>      <td>1154</td>      <td>SUNFLOWER RESTAURANT</td>      <td>63.5</td>    </tr>    <tr>      <th>1991</th>      <td>39776</td>      <td>Duc Loi Supermarket</td>      <td>64.0</td>    </tr>    <tr>      <th>2734</th>      <td>69397</td>      <td>Minna SF Group LLC</td>      <td>64.0</td>    </tr>    <tr>      <th>3291</th>      <td>78328</td>      <td>Golden Wok</td>      <td>64.0</td>    </tr>    <tr>      <th>4870</th>      <td>93150</td>      <td>Chez Beesen</td>      <td>64.0</td>    </tr>    <tr>      <th>4911</th>      <td>93502</td>      <td>Smoky Man</td>      <td>64.0</td>    </tr>    <tr>      <th>5510</th>      <td>98995</td>      <td>Vallarta's Taco Bar</td>      <td>64.0</td>    </tr>    <tr>      <th>1457</th>      <td>10877</td>      <td>CHINA FIRST INC.</td>      <td>64.5</td>    </tr>    <tr>      <th>2890</th>      <td>71310</td>      <td>Golden King Vietnamese Restaurant</td>      <td>64.5</td>    </tr>    <tr>      <th>4352</th>      <td>89070</td>      <td>Lafayette Coffee Shop</td>      <td>64.5</td>    </tr>    <tr>      <th>505</th>      <td>2542</td>      <td>PETER D'S RESTAURANT</td>      <td>65.0</td>    </tr>    <tr>      <th>2874</th>      <td>71008</td>      <td>House of Pancakes</td>      <td>65.0</td>    </tr>    <tr>      <th>818</th>      <td>3862</td>      <td>IMPERIAL GARDEN SEAFOOD RESTAURANT</td>      <td>66.0</td>    </tr>    <tr>      <th>2141</th>      <td>61427</td>      <td>Nick's Foods</td>      <td>66.0</td>    </tr>    <tr>      <th>2954</th>      <td>72176</td>      <td>Wolfes Lunch</td>      <td>66.0</td>    </tr>    <tr>      <th>4367</th>      <td>89141</td>      <td>Cha Cha Cha on Mission</td>      <td>66.5</td>    </tr>  </tbody></table>
# 
# 
# <!--
# BEGIN QUESTION
# name: q5b
# points: 3
# -->

# In[82]:


#x['median score'] = ins_named[ins_named['score'] > 0].groupby('bid', as_index=False).median()['score']
#x.sort_values(by='score', ascending=True)
#ins_named.merge(x).sort_values(by='median score', ascending=True)
#x.sort_values(by='score', ascending=True)
#ins_named

#ins_named['median score'] = ins_named[ins_named['score'] > 0].groupby('bid', as_index=False).median()['score']

#twenty_lowest_scoring = ins_named[['bid', 'name', 'median score']].sort_values(by='median score', ascending=True)
#twenty_lowest_scoring.head(20)
#ins_named.merge(score_and_bid[['bid']]).sort_values(by='score', ascending=True)[['bid', 'name', ',edian score']].head(20)

#ins_named[ins_named['score'] > 0].groupby('bid', as_index=False).median()
x = ins_named[ins_named['score'] > 0].groupby(['bid'], as_index=False).median()
x['score'].astype(np.float16)
ins_named.drop_duplicates(subset=['bid', 'name', 'address'])
ins_named['score'].astype(np.float16)
twenty_lowest_scoring = x.merge(ins_named, on='bid').drop_duplicates(subset=['bid'])
twenty_lowest_scoring = twenty_lowest_scoring.sort_values(by=['score_x', 'bid'], ascending=True)[['bid', 'name', 'score_x']]
twenty_lowest_scoring = twenty_lowest_scoring.rename(columns={'score_x':'median score'}).head(20)
twenty_lowest_scoring


# In[83]:


ok.grade("q5b");


# <br/><br/><br/>
# 
# --- 
# 
# ## Question 5c
# 
# Let's now examine the descriptions of violations for inspections with `score > 0` and `score < 65`.  Construct a **Series** indexed by the `description` of the violation from the `vio` table with the value being the number of times that violation occured for inspections with the above score range.  Sort the results in descending order of the count.
# 
# The first few entries should look like:
# 
# ```
# Unclean or unsanitary food contact surfaces                                  43
# High risk food holding temperature                                           42
# Unclean or degraded floors walls or ceilings                                 40
# Unapproved or unmaintained equipment or utensils                             39
# ```
# You will need to use `merge` twice.
# 
# <!--
# BEGIN QUESTION
# name: q5c
# points: 2
# -->

# In[93]:


merged_tables = ins.merge(ins2vio).merge(vio)
merged_tables = merged_tables[(merged_tables['score'] > 0) & (merged_tables['score'] < 65)].groupby(['description'])
merged_tables = merged_tables.count().sort_values(by=['score'], ascending=False).iloc[:,0].rename_axis(None)

low_score_violations = merged_tables

low_score_violations.head(20)


# In[94]:


ok.grade("q5c");


# <br/><br/><br/>
# 
# ---
# 
# <br/><br/><br/>
# 
# # 6: Compute Something Interesting
# 
# Play with the data and try to compute something interesting about the data. Please try to use at least one of groupby, pivot, or merge (or all of the above).  
# 
# Please show your work in the cell below and describe in words what you found in the same cell. This question will be graded leniently but good solutions may be used to create future homework problems.
# 
# **Please have both your code and your explanation in the same one cell below. Any work in any other cell will not be graded.**
# 
# 
# <!--
# BEGIN QUESTION
# name: q6
# points: 3
# manual: True
# -->
# 
# <!-- EXPORT TO PDF -->

# In[95]:


#YOUR CODE HERE
y = pd.DataFrame()
y['month'] = ((ins['timestamp']).dt.month)
y['day'] = (ins['timestamp']).dt.day
r = y.pivot_table(index=['month', 'day'], aggfunc='size').sort_values(ascending=False)
print("the busiest day of the year is " + str(r.index[0][0]) + "/"+ str(r.index[0][1]))


#YOUR EXPLANATION HERE (in a comment) I want to find the busiest day of the year. 
#This would require taking all the the dates, 
#parsing out the day/month away from the year and grouping by the MM/DD
#I pivoted all the months/days together to then find the count of how many inspections and events happened on a 
#specific date to then get the total count of what happens on a particular day on the calendar year.
#After that, I take the max of the Series to come to the conclusion that July 19th is the busiest day of the year.


# In[96]:


#THIS CELL AND ANY CELLS ADDED BELOW WILL NOT BE GRADED


# ## Congratulations! You have finished Part 1 of Project 1! 
# 
# In our analysis of the business data, we found that there are some errors with the ZIP codes. As a result, we made the records with ZIP codes outside of San Francisco have a `postal5` value of `None` and shortened 9-digit zip codes to 5-digit ones. In practice, we could take the time to look up the restaurant address online and fix some of the zip code issues.
# 
# In our analysis of the inspection data, we investigated the relationship between the year and type of inspection, and we figured out that only `Routine - Unscheduled` inspections have inspection scores.
# Finally, we joined the business and inspection data to identify restaurants with the worst ratings and the lowest median scores. 

# In[ ]:





# # Submit
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.
# **Please save before submitting!**
# 
# <!-- EXPECT 2 EXPORTED QUESTIONS -->

# In[97]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj1a.ipynb', 'proj1a.pdf')
ok.submit()

