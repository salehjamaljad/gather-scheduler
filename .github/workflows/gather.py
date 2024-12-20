#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from datetime import datetime
import time


# In[2]:


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
spreadsheet_id = '1kgc6LUF6z2WKsNZ6wLTZUoTEAxsdnk2zlx3l4V-bF5Q'
credentials = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
client = gspread.authorize(credentials)
spreadsheet = client.open_by_key(spreadsheet_id)
sheet_name = 'Talabat links'
sheet = spreadsheet.get_worksheet(1)
data = sheet.get_all_records()
df = pd.DataFrame(data)
df.head()


# In[3]:


branches = list(df.branch[:3])
sheet_links = list(df['sheet link'][:3])


# In[4]:


categories = ['fruit', 'veg', 'leaves', 'dates']  # Categories corresponding to worksheets 0, 1, 2, 3

# Assuming branches and sheet_links are lists
for branch, sheet_link in zip(branches, sheet_links):  
    workbook = client.open_by_url(sheet_link)  # Open the workbook
    for idx, category in enumerate(categories):  # Iterate through worksheets and categories
        worksheet = workbook.get_worksheet(idx)  # Get the worksheet by index
        df_name = f'df_{category}_{branch.lower()}'  # Construct the dynamic DataFrame name
        globals()[df_name] = pd.DataFrame(worksheet.get_all_records())  # Create the DataFrame


# In[5]:


for branch in branches:
    for category in categories:
        df_name = f'df_{category}_{branch.lower()}'  # Construct the dynamic DataFrame name
        if df_name in globals():  # Check if the DataFrame exists
            globals()[df_name].rename(columns={'stock': branch.lower()}, inplace=True)


# In[6]:


merged_dfs = {}  # Dictionary to store the merged DataFrames for Alexandria

# Loop through each category
for category in categories:
    # Collect all DataFrames for the current category
    dfs = [globals()[f'df_{category}_{branch.lower()}'] for branch in branches]
    
    # Initialize the merged DataFrame with the first DataFrame
    merged_df = dfs[0][['title', 'price', branches[0].lower()]]  # Assuming the branch name column is present
    
    # Iteratively merge the rest of the DataFrames
    for df, branch in zip(dfs[1:], branches[1:]):
        branch_column = branch.lower()  # The column to merge
        merged_df = pd.merge(
            merged_df,
            df[['title', 'price', branch_column]],  # Merge by title and the branch column
            on='title', how='outer'
        )
    
    # Clean up the 'price' column by combining values
    price_columns = [col for col in merged_df.columns if 'price' in col]
    merged_df['price'] = merged_df[price_columns].bfill(axis=1).iloc[:, 0]
    
    # Select relevant columns: 'title', 'price', and branch columns
    merged_df = merged_df[['title', 'price'] + [col for col in merged_df.columns if col not in ['price', 'title']]]
    
    # Store the merged DataFrame in the dictionary
    merged_dfs[f'merged_df_{category}_alexandria'] = merged_df


# In[7]:


for df in merged_dfs:
    merged_dfs[df].drop(columns=['price_x', 'price_y'], inplace=True)


# In[8]:


spreadsheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1kAi_R4-CQM4oCNUu2YMXGvLUrehbnx6iz0RywZOuxpE/edit?gid=579208347')

# Select the relevant sheets
sheet_fruit = spreadsheet.worksheet('فواكه')
sheet_veg = spreadsheet.worksheet('خضروات')
sheet_leaves = spreadsheet.worksheet('أعشاب وورقيات')
sheet_dates = spreadsheet.worksheet('تمر وفواكه مجففة')

# Function to clean invalid float values (NaN, inf, -inf) and replace NaN with 0
def clean_invalid_values(df):
    def clean_value(x):
        if isinstance(x, (float, int)):  # Only process numeric values
            if np.isnan(x):
                return 0  # Replace NaN with 0
            elif np.isinf(x):
                return None  # Replace inf/-inf with None
        return x  # Return other values unchanged

    # Apply the cleaning function element-wise
    df = df.applymap(clean_value)
    return df

# Function to add 'Last Updated' column
def add_last_updated_column(df):
    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add the 'Last Updated' column with the same value for all rows
    df['Last Updated'] = current_time
    
    return df

# Clean your DataFrames
merged_df_fruit_alexandria = clean_invalid_values(merged_dfs['merged_df_fruit_alexandria'])
merged_df_veg_alexandria = clean_invalid_values(merged_dfs['merged_df_veg_alexandria'])
merged_df_leaves_alexandria = clean_invalid_values(merged_dfs['merged_df_leaves_alexandria'])
merged_df_dates_alexandria = clean_invalid_values(merged_dfs['merged_df_dates_alexandria'])

# Add 'Last Updated' column to each DataFrame
merged_df_fruit_alexandria = add_last_updated_column(merged_df_fruit_alexandria)
merged_df_veg_alexandria = add_last_updated_column(merged_df_veg_alexandria)
merged_df_leaves_alexandria = add_last_updated_column(merged_df_leaves_alexandria)
merged_df_dates_alexandria = add_last_updated_column(merged_df_dates_alexandria)

# Convert DataFrames to lists of lists
fruit_data = merged_df_fruit_alexandria.values.tolist()
veg_data = merged_df_veg_alexandria.values.tolist()
leaves_data = merged_df_leaves_alexandria.values.tolist()
dates_data = merged_df_dates_alexandria.values.tolist()

# Prepare header rows (including 'Last Updated' column)
fruit_header = merged_df_fruit_alexandria.columns.tolist()
veg_header = merged_df_veg_alexandria.columns.tolist()
leaves_header = merged_df_leaves_alexandria.columns.tolist()
dates_header = merged_df_dates_alexandria.columns.tolist()

# Update each sheet with the new DataFrame
sheet_fruit.clear()  # Clears the existing content
sheet_fruit.append_row(fruit_header)  # Append header
sheet_fruit.append_rows(fruit_data)  # Append data

sheet_veg.clear()  # Clears the existing content
sheet_veg.append_row(veg_header)  # Append header
sheet_veg.append_rows(veg_data)  # Append data

sheet_leaves.clear()  # Clears the existing content
sheet_leaves.append_row(leaves_header)  # Append header
sheet_leaves.append_rows(leaves_data)  # Append data

sheet_dates.clear()  # Clears the existing content
sheet_dates.append_row(dates_header)  # Append header
sheet_dates.append_rows(dates_data)  # Append data


# In[9]:


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
spreadsheet_id = '1kgc6LUF6z2WKsNZ6wLTZUoTEAxsdnk2zlx3l4V-bF5Q'
credentials = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
client = gspread.authorize(credentials)
spreadsheet = client.open_by_key(spreadsheet_id)
sheet_name = 'Talabat links'
sheet = spreadsheet.get_worksheet(1)
data = sheet.get_all_records()
df = pd.DataFrame(data)
df.head()


# In[10]:


branches = list(df.branch[3:])
sheet_links = list(df['sheet link'][3:])


# In[11]:


len(branches)


# In[12]:


'''
categories = ['fruit', 'veg', 'leaves', 'dates']  # Categories corresponding to worksheets 0, 1, 2, 3

# Assuming branches and sheet_links are lists
for branch, sheet_link in zip(branches[:5], sheet_links[:5]):  
    workbook = client.open_by_url(sheet_link)  # Open the workbook
    for idx, category in enumerate(categories):  # Iterate through worksheets and categories
        worksheet = workbook.get_worksheet(idx)  # Get the worksheet by index
        df_name = f'df_{category}_{branch.lower()}'  # Construct the dynamic DataFrame name
        globals()[df_name] = pd.DataFrame(worksheet.get_all_records())  # Create the DataFrame

time.sleep(60)
for branch, sheet_link in zip(branches[5:10], sheet_links[5:10]):  
    workbook = client.open_by_url(sheet_link)  # Open the workbook
    for idx, category in enumerate(categories):  # Iterate through worksheets and categories
        worksheet = workbook.get_worksheet(idx)  # Get the worksheet by index
        df_name = f'df_{category}_{branch.lower()}'  # Construct the dynamic DataFrame name
        globals()[df_name] = pd.DataFrame(worksheet.get_all_records())  # Create the DataFrame

'''


# In[13]:


# Batch size and total branches
batch_size = 5
total_branches = 38
categories = ['fruit', 'veg', 'leaves', 'dates']  # Categories corresponding to worksheets 0, 1, 2, 3
# Loop through the batches
for start_idx in range(0, total_branches, batch_size):
    # Calculate the end index of the current batch
    end_idx = min(start_idx + batch_size, total_branches)
    
    # Process the current batch
    for branch, sheet_link in zip(branches[start_idx:end_idx], sheet_links[start_idx:end_idx]):  
        workbook = client.open_by_url(sheet_link)  # Open the workbook
        for idx, category in enumerate(categories):  # Iterate through worksheets and categories
            worksheet = workbook.get_worksheet(idx)  # Get the worksheet by index
            df_name = f'df_{category}_{branch.lower()}'  # Construct the dynamic DataFrame name
            globals()[df_name] = pd.DataFrame(worksheet.get_all_records())  # Create the DataFrame
    
    # Pause for 1 minute after processing a batch
    time.sleep(60)


# In[ ]:





# In[14]:


for branch in branches:
    for category in categories:
        df_name = f'df_{category}_{branch.lower()}'  # Construct the dynamic DataFrame name
        if df_name in globals():  # Check if the DataFrame exists
            globals()[df_name].rename(columns={'stock': branch.lower()}, inplace=True)


# In[15]:


merged_dfs = {}  # Dictionary to store the merged DataFrames for cairo

# Loop through each category
for category in categories:
    # Collect all DataFrames for the current category
    dfs = [globals()[f'df_{category}_{branch.lower()}'] for branch in branches]
    
    # Initialize the merged DataFrame with the first DataFrame
    merged_df = dfs[0][['title', 'price', branches[0].lower()]]  # Assuming the branch name column is present
    
    # Iteratively merge the rest of the DataFrames
    for df, branch in zip(dfs[1:], branches[1:]):
        branch_column = branch.lower()  # The column to merge
        merged_df = pd.merge(
            merged_df,
            df[['title', 'price', branch_column]],  # Merge by title and the branch column
            on='title', how='outer'
        )
    
    # Clean up the 'price' column by combining values
    price_columns = [col for col in merged_df.columns if 'price' in col]
    merged_df['price'] = merged_df[price_columns].bfill(axis=1).iloc[:, 0]
    
    # Select relevant columns: 'title', 'price', and branch columns
    merged_df = merged_df[['title', 'price'] + [col for col in merged_df.columns if col not in ['price', 'title']]]
    
    # Store the merged DataFrame in the dictionary
    merged_dfs[f'merged_df_{category}_cairo'] = merged_df


# In[16]:


for df in merged_dfs:
    merged_dfs[df].drop(columns=['price_x', 'price_y'], inplace=True)


# In[17]:


spreadsheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1UdwMs7MRYOcR5JG5lrjeOF7B0LBjUfThSGmoERrisXY/edit?gid=978012217#gid=978012217')

# Select the relevant sheets
sheet_fruit = spreadsheet.worksheet('فواكه')
sheet_veg = spreadsheet.worksheet('خضروات')
sheet_leaves = spreadsheet.worksheet('أعشاب وورقيات')
sheet_dates = spreadsheet.worksheet('تمر وفواكه مجففة')

# Function to clean invalid float values (NaN, inf, -inf) and replace NaN with 0
def clean_invalid_values(df):
    def clean_value(x):
        if isinstance(x, (float, int)):  # Only process numeric values
            if np.isnan(x):
                return 0  # Replace NaN with 0
            elif np.isinf(x):
                return None  # Replace inf/-inf with None
        return x  # Return other values unchanged

    # Apply the cleaning function element-wise
    df = df.applymap(clean_value)
    return df

# Function to add 'Last Updated' column
def add_last_updated_column(df):
    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add the 'Last Updated' column with the same value for all rows
    df['Last Updated'] = current_time
    
    return df

# Clean your DataFrames
merged_df_fruit_cairo = clean_invalid_values(merged_dfs['merged_df_fruit_cairo'])
merged_df_veg_cairo = clean_invalid_values(merged_dfs['merged_df_veg_cairo'])
merged_df_leaves_cairo = clean_invalid_values(merged_dfs['merged_df_leaves_cairo'])
merged_df_dates_cairo = clean_invalid_values(merged_dfs['merged_df_dates_cairo'])

# Add 'Last Updated' column to each DataFrame
merged_df_fruit_cairo = add_last_updated_column(merged_df_fruit_cairo)
merged_df_veg_cairo = add_last_updated_column(merged_df_veg_cairo)
merged_df_leaves_cairo = add_last_updated_column(merged_df_leaves_cairo)
merged_df_dates_cairo = add_last_updated_column(merged_df_dates_cairo)

# Convert DataFrames to lists of lists
fruit_data = merged_df_fruit_cairo.values.tolist()
veg_data = merged_df_veg_cairo.values.tolist()
leaves_data = merged_df_leaves_cairo.values.tolist()
dates_data = merged_df_dates_cairo.values.tolist()

# Prepare header rows (including 'Last Updated' column)
fruit_header = merged_df_fruit_cairo.columns.tolist()
veg_header = merged_df_veg_cairo.columns.tolist()
leaves_header = merged_df_leaves_cairo.columns.tolist()
dates_header = merged_df_dates_cairo.columns.tolist()

# Update each sheet with the new DataFrame
sheet_fruit.clear()  # Clears the existing content
sheet_fruit.append_row(fruit_header)  # Append header
sheet_fruit.append_rows(fruit_data)  # Append data

sheet_veg.clear()  # Clears the existing content
sheet_veg.append_row(veg_header)  # Append header
sheet_veg.append_rows(veg_data)  # Append data

sheet_leaves.clear()  # Clears the existing content
sheet_leaves.append_row(leaves_header)  # Append header
sheet_leaves.append_rows(leaves_data)  # Append data

sheet_dates.clear()  # Clears the existing content
sheet_dates.append_row(dates_header)  # Append header
sheet_dates.append_rows(dates_data)  # Append data

