import pandas as pd

def _check_disconnect_row(model_df,drow):
    checkset = []
    rowset = []
    rowindex = drow['ID']
    for x in drow['Inputs']:
        rowset.append(x[0])
    for x in drow['Outputs']:
        rowset.append(x[0])
    ind = 0
    for index, row in model_df.iterrows():
        if ind < len(model_df):
            if ind != rowindex:
                for x in row['Inputs']:
                    checkset.append(x[0])
                for x in row['Outputs']:
                    checkset.append(x[0])
        ind = ind + 1
    out = any(check in rowset for check in checkset) 
    if out: 
        disconnected = False  
    else : 
        disconnected = True

    return disconnected
 
def _connect_parts(model_df):
    df = model_df
    if len(model_df) > 1:
        for index, row in df.iterrows():
            if _check_disconnect_row(model_df,row):
                if index == 0:
                    if df.at[index+1,'Inputs'][0][0] != "None":
                        df.at[index,'Outputs'] = df.at[index+1,'Inputs']
                    else:
                        df.at[index+1,'Inputs'] = df.at[index,'Outputs']
                else:
                    if df.at[index-1,'Outputs'][0][0] != "None":
                        df.at[index,'Inputs'] = df.at[index-1,'Outputs']
                    else:
                        df.at[index-1,'Outputs'] = df.at[index,'Inputs']
    return df


# Two rows 1 and 2; 
# if output 1 is defined and input 2 is not, take output1; if opposite, take input 2 
# if act1 is processing and act2 is generating, take output1; if opposite, take input2
# if both have same attributes (proc/proc, def/def), take output1


# 1) check if disconnected; disconnected when none of the inputs is found in any other outputs OR if no outputs is found in any other input
# 2) Establish connection by connecting to the precious row by taking either a) taking the output(s) as input(s) or b) adding the input(s) as output(s) to the row before