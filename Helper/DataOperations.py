def flatten_list(list):
    flat_list = []
    for sublist in list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def group_concat(df, gr_cols, col_concat):
    df_out = (
        df
        .groupby(gr_cols)[col_concat]
        .apply(lambda x: ', '.join(x))
        .to_frame()
        .reset_index()
    )

    return df_out

def list_to_string(slist):
    if type(slist) == list:
        listToStr = ', '.join([str(elem) for elem in slist])  
        return listToStr
    else:
        return str(slist)

def transform_groupby(x):
    res = []
    unique = []
    for object in x:
        if object not in unique:
            unique.append(object)
    if len(unique) > 1:
        for a in x:
            unique.append(a)
            b = a
            app = [b[0],'XOR']
            if app not in res:
                res.append(app)
    else:
        res = unique
    res = list_to_string(res)
    return res