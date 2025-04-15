import torch

def query_manager(query, visual_slides, text_patients):
    visual_tmp =[]
    text_tmp = []
    for item in visual_slides:
        if query in item:
            visual_tmp.append(item)
    for item in text_patients:
        if query in item:
            text_tmp.append(item)
    query_dict = {'visual': visual_tmp, 'text': text_tmp}
    return query_dict

def visual_search(visual_database, query_list, top_k=1):
    if len(query_list)==1:
        query_idx = visual_database['patient_ids'].index(query_list[0])
        query = visual_database['features'][query_idx]
        #sim = torch.nn.functional.cosine_similarity(query, visual_database['features'], dim=1, eps=1e-6)
        sim = torch.cdist(query.unsqueeze(0), visual_database['features'], p=2)
        #print(sim)
        topk = torch.topk(sim.squeeze(), k = top_k + 1, largest=False)
        topk_report = {'sim':topk.values, 'index':topk.indices}
    else:
        # feat = []
        # for q in query_list:
        #     query_idx = visual_database['slide_ids'].index(q)
        #     feat.append(visual_database['features'][query_idx])
        # query = torch.stack(feat, dim = 0)
        topk_report = {'sim':torch.tensor([]), 'index':torch.tensor([])}
    return topk_report

def text_search(text_database, query_list, top_k=1):
    query_idx = text_database['patient_ids'].index(query_list[0])
    query = text_database['features'][query_idx]
    #sim = torch.nn.functional.cosine_similarity(query, text_database['features'], dim=1, eps=1e-6)
    sim = torch.cdist(query.unsqueeze(0), text_database['features'], p=2)
    #print(torch.min(abs(text_database['features'])))
    topk = torch.topk(sim.squeeze(), k = top_k + 1, largest = False )
    topk_report = {'sim':topk.values, 'index':topk.indices}
    return topk_report

def database_matcher(visual_slides, text_patients):
    patient_slide_map = []
    slide_patient_map = []
    joint_patients = []
    for patient_id in text_patients:
        text_p = patient_id.split('.')[0]
        patient_to_slide = []
        for i in range(len(visual_slides)):
            slide = visual_slides[i]
            vis_p = slide[:12]
            if text_p == vis_p:
                patient_to_slide.append(i)
                if not patient_id in joint_patients:
                    joint_patients.append(patient_id)
        if len(patient_to_slide) > 0:
            patient_slide_map.append(patient_to_slide)
    for i in range(len(patient_slide_map)):
        for j in patient_slide_map[i]:
            slide_patient_map.append(i)
    overall_map = {'text2slide':patient_slide_map, 'slide2text':slide_patient_map}
    #print('patient_slide_map:', patient_slide_map)
    return overall_map, joint_patients

def map_visual_results(results, overall_map):
    indexes = results['index']
    updated_indexes = []
    for item in indexes:
        updated_indexes.append(overall_map['slide2text'][item])
    results['index'] = updated_indexes
    return results

def update_databases(visual_database, text_database, joint_patients):
    v_d = {'features': [], 'patient_ids': []}
    t_d = {'features': [], 'patient_ids': [], 'cancer_code':[]}
    for p in joint_patients:
        patient = p.split('.')[0]
        for item in visual_database['patient_ids']:
            if patient in item:
                ind = visual_database['patient_ids'].index(item)
                v_d['features'].append(visual_database['features'][ind])
                v_d['patient_ids'].append(item)
        ind_t = text_database['patient_ids'].index(p)
        t_d['features'].append(text_database['features'][ind_t])
        t_d['patient_ids'].append(p)
        t_d['cancer_code'].append(text_database['cancer_code'][ind_t])
    v_d['features'] = torch.stack(v_d['features'], dim = 0 ).double()
    t_d['features'] = torch.stack(t_d['features'], dim = 0).double()
    print(v_d['features'].shape, t_d['features'].shape )
    return v_d, t_d

def visual_slide2patient_database(visual_database):
    patient_ids = {}
    d_v = {'features':[], 'patient_ids':[]}
    for item in visual_database['slide_ids']:
        p = item[:12]
        if not p in list(patient_ids.keys()):
            patient_ids[p] = []
        patient_ids[p].append(item)
    for p in sorted(list(patient_ids.keys())):
        if len(patient_ids[p]) == 1:
            item = patient_ids[p][0]
            ind = visual_database['slide_ids'].index(item)
            d_v['features'].append(visual_database['features'][ind])
            d_v['patient_ids'].append(p)
        else:
            tmp = []
            for item in patient_ids[p]:
                ind = visual_database['slide_ids'].index(item)
                tmp.append(visual_database['features'][ind])
            tmp = torch.stack(tmp, dim=0)
            d_v['features'].append(torch.mean(tmp, dim=0))
            d_v['patient_ids'].append(p)
    return d_v