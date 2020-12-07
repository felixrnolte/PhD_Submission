import pandas as pd
import mlconjug
default_conjugator = mlconjug.Conjugator(language='en')
import spacy
from ast import literal_eval
import en_core_web_sm
nlp = en_core_web_sm.load()
from Helper.LanguageOperations import _get_antonym
from Helper.DataOperations import transform_groupby

def model_element_mapping(feature_df):
    mapping_df = pd.DataFrame(columns =['ID', 'Activity','Inputs','Outputs','Role'])
    id = 0
    condindi = 0
    for index, row in feature_df.iterrows():
        core = _map_core(row)
        advcl = _map_advcl(row)
        cond = _map_cond(row)
        relcl = _map_relcl(row)             
        
        if core:
            activity = core[0]
            inputs = core[1]
            outputs = core[2]
            role = core[3]

            if cond:
                for con in cond:
                    if con[0] == "before":
                        inputs = [con[1][1]]
                        ref_outputs = con[2]
                        if (len(mapping_df) - con[3]) >= 0:
                            ref = con[3]
                            if condindi > 0:
                                temp_out = mapping_df.at[ref,"Outputs"]
                                mapping_df.at[ref,"Outputs"] = [temp_out[0],ref_outputs[1]]
                            else:
                                mapping_df.at[ref,"Outputs"] = ref_outputs[1:]
                            condindi = condindi + 1
                        else:
                            condindi = 0
            else:
                condindi = 0

            if relcl:
                for rel in relcl:
                    if rel[0] == activity[0]:
                        inputs = [rel[1]]
            mapping_elements = [id,activity,inputs,outputs,role]
        
        act_suc = []
        if advcl:
            adv_activity = advcl[1]
            adv_inputs = advcl[2]
            adv_outputs = advcl[3]
            adv_role = advcl[4]
            if advcl[0] == "precessor":
                mapping_df = mapping_df.append({'ID': id, 'Activity' : adv_activity, 'Inputs': adv_inputs, 'Outputs': adv_outputs, 'Role': adv_role}, ignore_index=True)
                inputs = adv_outputs
            if advcl[0] == "successor" and core:
                act_suc = [id+1,adv_activity,outputs,adv_outputs,adv_role]

        if act_suc:
            mapping_df = mapping_df.append({'ID': id, 'Activity' : mapping_elements[1], 'Inputs': mapping_elements[2], 'Outputs': mapping_elements[3], 'Role': mapping_elements[4]}, ignore_index=True)
            mapping_df = mapping_df.append({'ID': act_suc[0], 'Activity' : act_suc[1], 'Inputs': act_suc[2], 'Outputs': act_suc[3], 'Role': act_suc[4]}, ignore_index=True)
            id = id + 1
        else:
            mapping_df = mapping_df.append({'ID': id + 1, 'Activity' : mapping_elements[1], 'Inputs': mapping_elements[2], 'Outputs': mapping_elements[3], 'Role': mapping_elements[4]}, ignore_index=True)
            id = id + 1
    
    return mapping_df


## pps have to be included here
def _map_core(row):
    CREAT_VERB = {'create','assemble','make','generate','design','produce','construct','build','prepare'}
    tmp = row[0]
    if tmp and tmp[0]:
        svo = tmp[0][0]
        pps = tmp[1]
        
        # Process SVO
        main_activity = svo[1]
        role = svo[0]
        main_object = svo[2]
        
        if main_object != "None":
            activity_label = main_activity + " " + main_object
        else:
            activity_label = main_activity
        
        vtype = "processing"
        if main_activity in CREAT_VERB:
            vtype = "creating" 
        
        activity = [activity_label,vtype]
        
        main_objects = []
        
        if len(main_object.split()) > 1:
            cnctr = 'AND'
            for tok in nlp(main_object):
                if tok.text.lower() == "or":
                    cnctr = "XOR"
                    break
            for chunk in nlp(main_object).noun_chunks:
                if chunk.root.head.text.lower() == "of":
                    main_objects[-1] = [main_objects[-1][0] + " of " + chunk.text, cnctr]
                else:
                    main_objects.append([chunk.text,cnctr])
        else:
            for chunk in nlp(main_object).noun_chunks:
                if chunk.root.head.text.lower() == "of":
                    main_objects[-1] = [main_objects[-1][0] + " of " + chunk.text, 'AND']
                else:
                    main_objects.append([main_object,'AND'])
        
        if not main_objects:
            main_objects.append([main_object,'AND'])

        inputs = []
        outputs = []

        if pps:
            for pp in pps:
                if pp[0] == svo[1]:
                    if pp[2] == "input":
                            inputs.append([pp[1],pp[3],'defined'])
                    elif pp[2] == "output":
                            outputs.append([pp[1],pp[3],'defined'])
                    elif pp[2] == "role":
                            role = pp[2]
        if not inputs:
            if activity[1] == 'creating':
                for obj in main_objects:
                    input = obj[0] + " to " + main_activity
                    inputs.append([input,obj[1],'generated'])
            else:
                for obj in main_objects:
                    input = obj[0]
                    inputs.append([input,obj[1],'defined'])
        if not outputs:
            if activity[1] == 'creating':
                for obj in main_objects:
                    output = obj[0]
                    outputs.append([output,obj[1],'defined'])
            else:
                for obj in main_objects:
                    output = default_conjugator.conjugate(main_activity).conjug_info['indicative']['indicative present perfect']['3s'] + " " + obj[0]
                    outputs.append([output,obj[1],'generated'])    
        mapping = (activity,inputs,outputs,role)
        return mapping

    else:
        return None

def _map_advcl(row):
    tmp = row[1]
    if tmp:
        svo = tmp[0][0]
        pps = tmp[1]
        
        # Process SVO
        main_activity = svo[1]
        role = svo[0]
        main_object = svo[2]
        
        activity_label = main_activity + " " + main_object
        
        type = "processing"
        CREAT_VERB = {'create','assemble','make','generate','design','produce','construct','build'}
        if main_activity in CREAT_VERB:
            type = "creating" 
        
        activity = [activity_label,type]
        
        main_objects = []
        
        if len(main_object.split()) > 1:
            cnctr = 'AND'
            for tok in nlp(main_object):
                if tok.text.lower() == "or":
                    cnctr = "XOR"
                    break
            for chunk in nlp(main_object).noun_chunks:
                if chunk.root.head.text.lower() == "of":
                    main_objects[-1] = [main_objects[-1][0] + " of " + chunk.text, cnctr]
                else:
                        main_objects.append([chunk.text,cnctr])
        else:
            for chunk in nlp(main_object).noun_chunks:
                if chunk.root.head.text.lower() == "of":
                    main_objects[-1] = [main_objects[-1][0] + " of " + chunk.text, 'AND']
                else:
                    main_objects.append([main_object,'AND'])
        

        inputs = []
        outputs = []

        if pps:
            for pp in pps:
                if pp:
                    if pp[0] == svo[1]:
                        if pp[2] == "input":
                            inputs.append([pp[1],role,pp[3]])
                        elif pp[2] == "output":
                            outputs.append([pp[1],role,pp[3]])
                        elif pp[2] == "role":
                            role = pp[2]
        if not inputs:
            if activity[1] == 'creating':
                for obj in main_objects:
                        input = obj + " to " + main_activity
                        inputs.append([input,obj[1],'generated'])
            else:
                for obj in main_objects:
                    input = obj[0]
                    inputs.append([input,obj[1],'defined'])
        if not outputs:
            if activity[1] == 'creating':
                for obj in main_objects:
                        output = obj[0]
                        outputs.append([output,obj[1],'defined'])
            else:
                for obj in main_objects:
                        output = default_conjugator.conjugate(main_activity).conjug_info['indicative']['indicative present perfect']['3s'] + " " + obj[0]
                        outputs.append([output,obj[1],'generated'])
        type = tmp[2]             
        mapping = (tmp[2],activity,inputs,outputs,role)
        return mapping
    else:
        return None
        
def _map_cond(row):
    elements = []
    tmp = row[2]
    if row[0][0]:
        crow = row[0][0][0]
    else:
        crow = ["","",""]
    if tmp:
        refactvitiy_out = tmp[1]
        refactvitiy_in = crow[1] + " " + crow[2]
        adj = nlp(tmp[2])[0]
        noun = nlp(tmp[2])[1:]
        syn = adj.text + " " + noun.text
        ant = _get_antonym(adj) + " " + noun.text

        type = "before"
        input = [refactvitiy_in, [syn,'AND','defined']]
        outputs = [refactvitiy_out,[syn,'XOR','defined'],[ant,'XOR','generated']]
        element = [type,input,outputs,tmp[3]]
        elements.append(element)
        return elements
    else:
        return None

def _map_relcl(row):
    elements = []
    tmp = row[3]
    if tmp:
        refactivity = tmp[1][0][1] + " " + tmp[3]
        obj = default_conjugator.conjugate(tmp[2]).conjug_info['indicative']['indicative present perfect']['1s'] + " " + tmp[3]
        element = [refactivity, [obj, "defined", "AND"]]
        elements.append(element)
        return elements
    else:  
        return None 