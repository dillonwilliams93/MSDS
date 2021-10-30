import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import math
import operator
from collections import defaultdict

# Part 1
def information_gain_target(dataset_file):
    #        Input: dataset_file - A string variable which references the path to the dataset file.
    #        Output: ig_loan - A floating point variable which holds the information gain associated with the target variable.
    #
    #        NOTE:
    #        1. Return the information gain associated with the target variable in the dataset.
    #        2. The Loan attribute is the target variable
    #        3. The pandas dataframe has the following attributes: Age, Income, Student, Credit Rating, Loan
    #        4. Perform your calculations for information gain and assign it to the variable ig_loan

    df = pd.read_csv(dataset_file)
    ig_loan = 0

    # your code here
    count_total = df.shape[0]
    count_yes = 0
    count_no = 0
    for index, row in df.iterrows():
        if df.loc[index]['Loan'] == 'yes':
            count_yes = count_yes + 1
        else:
            count_no = count_no + 1
    ig_loan = -(count_yes / count_total) * math.log2(count_yes / count_total) - (count_no / count_total) * math.log2(
        count_no / count_total)
    return ig_loan

ig_loan = information_gain_target('test_dataset.csv')
print(f"ig_loan = {ig_loan}")

def information_gain(p_count_yes, p_count_no):
    #   A helper function that returns the information gain when given counts of number of yes and no values.
    #   Please complete this function before you proceed to the information_gain_attributes function below.

    # your code here
    count_total = p_count_yes + p_count_no
    ig = -(p_count_yes / count_total) * math.log2(p_count_yes / count_total) - (p_count_no / count_total) * math.log2(
        p_count_no / count_total)
    return ig

def information_gain_attributes(dataset_file, ig_loan, attributes, attribute_values):
    #        Input:
    #            1. dataset_file - A string variable which references the path to the dataset file.
    #            2. ig_loan - A floating point variable representing the information gain of the target variable "Loan".
    #            3. attributes - A python list which has all the attributes of the dataset
    #            4. attribute_values - A python dictionary representing the values each attribute can hold.
    #
    #        Output: results - A python dictionary representing the information gain associated with each variable.
    #            1. ig_attributes - A sub dictionary representing the information gain for each attribute.
    #            2. best_attribute - Returns the attribute which has the highest information gain.
    #
    #        NOTE:
    #        1. The Loan attribute is the target variable
    #        2. The pandas dataframe has the following attributes: Age, Income, Student, Credit Rating, Loan

    results = {
        "ig_attributes": {
            "Age": 0,
            "Income": 0,
            "Student": 0,
            "Credit Rating": 0
        },
        "best_attribute": ""
    }

    df = pd.read_csv(dataset_file)
    d_range = len(df)


    for attribute in attributes:
        ig_attribute = 0
        value_counts = dict()
        vcount = df[attribute].value_counts()
        for att_value in attribute_values[attribute]:
            # your code here
            ig_attribute_yes = 0
            ig_attribute_no = 0
            for index,row in df.iterrows():
                if (df.iloc[index][f"{attribute}"] == f"{att_value}") and (df.iloc[index]["Loan"] == "yes"):
                    ig_attribute_yes = ig_attribute_yes+1
                elif (df.iloc[index][f"{attribute}"] == f"{att_value}") and (df.iloc[index]["Loan"] == "no"):
                    ig_attribute_no = ig_attribute_no+1
            ig_attribute = ig_attribute+(vcount[f"{att_value}"]/sum(vcount))*information_gain(ig_attribute_yes, ig_attribute_no)
        results["ig_attributes"][attribute] = ig_loan - ig_attribute

    results["best_attribute"] = max(results["ig_attributes"].items(), key=operator.itemgetter(1))[0]
    return results

attribute_values = {
    "Age": ["<=30", "31-40", ">40"],
    "Income": ["low", "medium", "high"],
    "Student": ["yes", "no"],
    "Credit Rating": ["fair", "excellent"]
}

attributes = ["Age", "Income", "Student", "Credit Rating"]

result = information_gain_attributes('test_dataset.csv', ig_loan, attributes, attribute_values)
print(f"Result = {result}")

# Part 2
def naive_bayes(dataset_file, attributes, attribute_values):
    #   Input:
    #       1. dataset_file - A string variable which references the path to the dataset file.
    #       2. attributes - A python list which has all the attributes of the dataset
    #       3. attribute_values - A python dictionary representing the values each attribute can hold.
    #
    #   Output: A proabbilities dictionary which contains the counts of when the Loan target variable is yes or no
    #       depending on the input attribute.
    #
    #   Hint: Starter code has been provided to you to calculate the counts. Your code is very similar to the previous problem.

    probabilities = {
        "Age": {"<=30": {"yes": 0, "no": 0}, "31-40": {"yes": 0, "no": 0}, ">40": {"yes": 0, "no": 0}},
        "Income": {"low": {"yes": 0, "no": 0}, "medium": {"yes": 0, "no": 0}, "high": {"yes": 0, "no": 0}},
        "Student": {"yes": {"yes": 0, "no": 0}, "no": {"yes": 0, "no": 0}},
        "Credit Rating": {"fair": {"yes": 0, "no": 0}, "excellent": {"yes": 0, "no": 0}},
        "Loan": {"yes": 0, "no": 0}
    }

    df = pd.read_csv(dataset_file)
    d_range = len(df)

    vcount = df["Loan"].value_counts()
    vcount_loan_yes = vcount["yes"]
    vcount_loan_no = vcount["no"]

    probabilities["Loan"]["yes"] = vcount_loan_yes / d_range
    probabilities["Loan"]["no"] = vcount_loan_no / d_range

    for attribute in attributes:
        value_counts = dict()
        vcount = df[attribute].value_counts()
        for att_value in attribute_values[attribute]:
    # your code here
            ig_attribute_yes = 0
            ig_attribute_no = 0
            for index, row in df.iterrows():
                if (df.iloc[index][f"{attribute}"] == f"{att_value}") and (df.iloc[index]["Loan"] == "yes"):
                    ig_attribute_yes = ig_attribute_yes + 1
                elif (df.iloc[index][f"{attribute}"] == f"{att_value}") and (df.iloc[index]["Loan"] == "no"):
                    ig_attribute_no = ig_attribute_no + 1
            probabilities[f"{attribute}"][f"{att_value}"]["yes"] = ig_attribute_yes / vcount_loan_yes
            probabilities[f"{attribute}"][f"{att_value}"]["no"] = ig_attribute_no / vcount_loan_no
    return probabilities

print(f"Probabilities = {naive_bayes('test_dataset.csv',attributes,attribute_values)}")