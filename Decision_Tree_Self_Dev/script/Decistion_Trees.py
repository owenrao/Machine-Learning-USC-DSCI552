#!/usr/bin/env python
# coding: utf-8

# # Report on Decision Tree Learning Algorithm using Python

# by Ruijie Rao on 2022/02/01

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

data_path = "dt_data.txt"

# ## 2. Decision Tree Generation

def p(x):
    result = x/x.sum()
    return result

def entropy(x):
    result = -1*np.sum(x.logp*x.p)
    return result

def cal_node_entropy(col,df):
    df_branch = df.groupby([col]).count()[["count"]]
    df_leaf = df.groupby([col,"Enjoy"]).count()[["count"]]
    df_leaf["p"] = df_leaf.groupby(level=[0]).apply(p)[["count"]]
    df_leaf["logp"] = df_leaf["p"].apply(np.log2)
    df_branch["entropy"] = df_leaf.groupby(level=[0]).apply(entropy)
    df_branch["count*entropy"] = df_branch["count"]*df_branch["entropy"]
    branch_mean_entropy = sum(df_branch["count*entropy"])/sum(df_branch["count"])
    return branch_mean_entropy



def termination_check(df,history):
    if len(df["Enjoy"].unique()) == 1: ## Criteria 1: Pure
        return True
    elif len(history) == 6:
        return True #"Complexity Maximum Reached"
    elif df.empty:
        return True
    else:
        return False

def gen_decision_tree(current_node_df, history):
    response = termination_check(current_node_df,history)
    if response == True: ## Need Modify
        return {"Termination": [str(i) for i in current_node_df["Enjoy"].unique()]}
    current_node = {
        "Parent_Node": None,
        "Entropy": 1
    }
    for column_name in eva_col: 
        if column_name in history: continue
        node_mean_entropy = cal_node_entropy(column_name,current_node_df)
        if node_mean_entropy < current_node["Entropy"]:
            current_node = {
                "Parent_Node": column_name,
                "Entropy": node_mean_entropy,
            }
    current_column_name = current_node["Parent_Node"]
    if current_column_name==None: return {}
    #current_node["Node_df"] = current_node_df
    current_node["History"] = history+[current_column_name]
    current_node_df_gb = current_node_df.groupby(current_column_name)
    current_node["Child_Node"] = [
                    {
                        "Node_name": child_node_name,
                        "Node_info": gen_decision_tree(child_node_df, current_node["History"])
                    }
                    for child_node_name,child_node_df in current_node_df_gb
                ]
    return current_node

import json
def display_tree(tree):
    print(json.dumps(tree, indent=4))



raw_df = pd.read_csv(data_path).set_index("Id")

df = raw_df
for col_name in ['Enjoy']:
    df[col_name] = raw_df[col_name].map({"Yes":True,"No":False})

for col in df.drop("Enjoy",axis=1).columns:
    column_data = df[col]
    categories = column_data.unique()
    cat_type = CategoricalDtype(categories)
    df[col] = column_data.astype(cat_type)

df["count"] = [1 for i in range(len(df))]
eva_col = set(df.drop(["Enjoy","count"],axis=1).columns)



current_node = {
    "Parent_Node": None,
    "Entropy": 1,
    #"Node_df": df,
    "History":[],
    "Child_Node": []
}
for column_name in eva_col: 
        node_mean_entropy = cal_node_entropy(column_name,df)
        if node_mean_entropy < current_node["Entropy"]:
            current_node["Parent_Node"] = column_name
            current_node["Entropy"] = node_mean_entropy
current_column_name = current_node["Parent_Node"]
current_node["History"].append(current_node["Parent_Node"])
current_history = current_node["History"]
current_node_df_gb = df.groupby(current_column_name)
current_node["Child_Node"] = [
                        {
                            "Node_name": child_node_name,
                            "Node_info": gen_decision_tree(child_node_df, current_history)
                        }
                        for child_node_name,child_node_df in current_node_df_gb
                    ]
tree = current_node


# ## 2.2 Display the Tree
display_tree(tree)

## save tree as json file
with open("tree.json","w") as f:
    f.write(json.dumps(tree, indent=4))
