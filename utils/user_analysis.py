import os
import pandas as pd
from difflib import Differ
import matplotlib.pyplot as plt 
from pprint import pprint


def tabulate_categorical_basic(series: pd.Series, min_n:int) -> pd.Series: 
    '''
    Takes: 
        series: pd.Series 
        min_n: minimum number of observations
    Returns: 
        pd.Series: value counts of unique observations. for smaller min_n grouped as other  

    '''
    vcounts = series.value_counts()
    included_cats_filter = vcounts>=min_n
    final_counts = vcounts[included_cats_filter]
    final_counts[f'Other ({sum(~included_cats_filter)} categories)'] = sum(vcounts[~included_cats_filter])

    return final_counts


def user_analysis(df, plotting_kwargs=None, counting_kwargs=None):
    '''
    Some basic stats about users. 
    ''' 
    print(f"Total number of edits: {len(df['userid'])}")
    print(f"Number of unique existing users: {df['userid'].nunique()}")
    print(f"Number of edits by deleted users: {sum(df['userid'].isna())} \n (impossible to detect whether these are the same or different ppl)")
    cutoff_v_counts = tabulate_categorical_basic(df['userid'], **counting_kwargs)
    cutoff_v_counts[:-1].hist(**plotting_kwargs)
    plt.title(f'Number of edits per user.\n(min edits: {counting_kwargs["min_n"]})')
    plt.xlabel('Number of edits')
    plt.ylabel('Number of Users')
    plt.show()


def get_edits(differences_generator):
    deletion = []
    insertion = []
    for d in differences_generator:
        
        if d.startswith('-'): 
            deletion.append(d[2:]) # clean beginning
        elif d.startswith('+'): 
            insertion.append(d[2:])
    return deletion, insertion

def find_edits(text1, text2):
    if text2 is None: 
        return None
    else: 
        diff = Differ()
        text1, text2 = text1.splitlines(), text2.splitlines()
        diff_gen = diff.compare(text1, text2)

        return diff_gen


def detect_overediting(df:pd.DataFrame, timeframe:int, exclude_own_edits=True):
    '''
    Show users that edit same paragraph within timeframe
    takes:
        revisions df 
        timeframe in hours
    Returns: 
        df qith quick overediting (bool, below combined), 
            quick edits (bool, withtin timeframe) and 
            overditing (bool, same paragraph as previous ins) 
    ''' 
    timedelta = (df['timestamp'].diff().dt.total_seconds()/3600).shift(-1).abs() # get total hours since last edit
    df['quick_edits'] = timedelta < timeframe # extract text they have edited, check if that is still in next version
    # in other version the insertion from previous one should not be a deletion in current row 
    df['shifted_ins'] = df['insertions'].shift(-1)
    overedited = df[:-1].apply(lambda x: detect_overlap(x['deletions'], x['shifted_ins']), axis=1)
    overedited[len(df)-1] = False # first one cannot overedit by default 
    df['overedited'] = overedited
    if exclude_own_edits: 
        own_edit = df['userid'] == df['userid'].shift(-1)
        df['quick_overediting'] = df['quick_edits'] & df['overedited'] & ~own_edit
    else: 
        df['quick_overediting'] = df['quick_edits'] & df['overedited']


    return df

def detect_overlap(list1, list2):
    '''
    list1: deletions
    list2: ins
    '''
    for el in list1: 
        if el in list2: 
            return True
    return False
    
