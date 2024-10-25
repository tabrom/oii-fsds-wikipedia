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
            deletion.append(d)
        elif d.startswith('+'): 
            insertion.append(d)
    return deletion, insertion


def find_edits(text1, text2):
    if text2 is None: 
        return None
    else: 
        diff = Differ()
        text1, text2 = text1.splitlines(), text2.splitlines()
        diff_gen = diff.compare(text1, text2)

        return diff_gen

