import pandas as pd
import numpy as np
import math
import xml.etree.ElementTree as ET
import sys
# df = pd.read_csv("car.csv",header=None)
# df.columns = ['att'+str(i) for i in df.columns]
# #df = df.rename(columns={'vhigh': 'att1', 'vhigh.1': 'att2', '2': 'att3', '2.1': 'att4','small': 'att5', 'low': 'att6','unacc': 'att7'},inplace=True)
#
# data = df
# number_of_classes = len(data[data.columns[-1]].unique())


def entropy(data, number_of_classes):
    label_column = data[data.columns[-1]]
    unique_classes, count = np.unique(label_column, return_counts=True)
    entropy = np.sum([-count[i] / sum(count) * math.log(count[i] / sum(count), number_of_classes) for i in range(len(count))])
    ##print(entropy)
    return entropy


def InformationGain(data, split_att_name, number_of_classes):  ##name of the feature for which IG should be calculated
    total_entropy = entropy(data, number_of_classes)
    vals, count = np.unique(data[split_att_name], return_counts=True)

    weight_entropy = np.sum(
        [(count[i] / np.sum(count)) * entropy(data[data[split_att_name] == vals[i]], number_of_classes) for i in
         range(len(vals))])
    Infomation_gain = total_entropy - weight_entropy

    return Infomation_gain

def Id3(orignaldata, number_of_classes,root):
        total_entropy = entropy(orignaldata, number_of_classes)
        attr_list = orignaldata.columns[:-1]
        max_ig_val = 0
        max_ig_attr = None

        for attr in attr_list:
            ig = InformationGain(orignaldata, attr, number_of_classes)
            if ig > max_ig_val:
                max_ig_val = ig
                max_ig_attr = attr

        attr_unique_values = orignaldata[max_ig_attr].unique()
        for attr_value in attr_unique_values:
            tmp_df = orignaldata[orignaldata[max_ig_attr] == attr_value]
            child_entropy = entropy(tmp_df, number_of_classes)
            if child_entropy == 0:
                ET.SubElement(root, "node", entropy=str(child_entropy), value=attr_value, feature=max_ig_attr).text = tmp_df['att6'].unique()[0]
                continue
            else:
                child = ET.SubElement(root, "node", entropy=str(child_entropy), value=attr_value, feature=max_ig_attr)
                Id3(tmp_df, number_of_classes,child)

       # print(max_ig_attr)
        # total_entr= entropy(orignaldata,classes)

if __name__ == '__main__':

    arguments_passed = sys.argv
    properties_dict = {}
    for ind in range(len(arguments_passed)):
        if ind == 0:
            continue
        if '--' in arguments_passed[ind]:
            key = arguments_passed[ind].split('--')[1]
            val = arguments_passed[ind + 1]
            properties_dict[key] = val


    df = pd.read_csv(properties_dict['data'], header=None)
    df.columns = ['att' + str(i) for i in df.columns]
    # df = df.rename(columns={'vhigh': 'att1', 'vhigh.1': 'att2', '2': 'att3', '2.1': 'att4','small': 'att5', 'low': 'att6','unacc': 'att7'},inplace=True)
    data = df
    number_of_classes = len(data[data.columns[-1]].unique())

    ## nc= len(df['number_of_classes'].unique())
    total_entropy = entropy(data, number_of_classes)
    root = ET.Element("tree", entropy= str(total_entropy))

    Id3(data, number_of_classes,root)
    tree = ET.ElementTree(root)
    tree.write(properties_dict['output'])

    # tree = ET.Element(root)
    # tree.write(properties_dict['output'])
