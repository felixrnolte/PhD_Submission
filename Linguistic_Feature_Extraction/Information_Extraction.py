from Helper.Subject_Verb_Object_Extract import _get_svos_sent
from Helper.LanguageOperations import _get_prep_sent, _get_objects, _get_adv_role, _get_advmod_sent
import pandas as pd

def _get_ling_information(phrase_df):
    info_df = pd.DataFrame(columns =['Core', 'Adverbial','Conditional','Relative'])
    condindi = 1
    for index, row in phrase_df.iterrows():
        core_features = _get_core_features(row)
        advcl_features = _get_advcl_features(row)
        condindi, cond_features = _get_cond_features(index,condindi,row,info_df)
        relcl_features = _get_relcl_features(row) 
        info_df = info_df.append({'Core': core_features, 'Adverbial' : advcl_features, 'Conditional': cond_features, 'Relative': relcl_features}, ignore_index=True)
    return info_df


def _get_core_features(df_row):
    info = []
    if df_row['Core']:
        for x in df_row['Core']:
            rem_pps, pps = _get_prep_sent(x)
            for y in rem_pps:
                x = x.replace(y, "")
            svos = _get_svos_sent(x)
            advmod = _get_advmod_sent(x)
            info = [svos,pps,advmod]
    return info

def _get_advcl_features(df_row):
    info = []
    if df_row['Adverbial']:
        for x in df_row['Adverbial']:
            svos = _get_svos_sent(x)
            pps = _get_prep_sent(x)
            role = _get_adv_role(x)
            info = [svos,pps,role]
    return info

def _get_cond_features(index,condindi,df_row,feature_df):
    info = []
    if df_row['Conditional']:
        condrefindex = index - int(condindi)
        for x in df_row['Conditional']:
            refsvos = _get_svos_sent(df_row['Core'][0])[0]
            ref = refsvos[1] + " " + refsvos[2]
            obj = _get_objects(x)
            info = ["before",ref,obj,condrefindex]
            #condrefindex = 1
        condindi = condindi + 1
    else:
        condindi = 1
    return condindi, info

def _get_relcl_features(df_row):
    info = []
    if df_row['Relative']:
        for x in df_row['Relative']:
            role = "before"
            ref_v = _get_svos_sent(df_row['Core'][0])
            if not ref_v:
                break
            svo = _get_svos_sent(x)
            ref_o = _get_svos_sent(df_row['Core'][0])[0][2]
            pps = _get_prep_sent(x)
            info = [role, ref_v, svo[0][1], ref_o, pps]
    return info
