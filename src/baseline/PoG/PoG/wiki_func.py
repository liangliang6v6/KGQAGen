from prompt_list import *
from utils import *
import json
import openai
import re
import time

def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    # print("inside construct")
    prompt =  extract_relation_prompt_wiki % (3, 3)+question+'\nTopic Entity: '+entity_name+ '\nRelations:\n'+'\n'.join([f"{i}. {item}" for i, item in enumerate(total_relations, start=1)])+'A:'
    # print("organize prompt", prompt)
    return prompt


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    useless_relation_list = ["category's main topic", "topic\'s main category", "stack exchange site", 'main subject', 'country of citizenship', "commons category", "commons gallery", "country of origin", "country", "nationality"]
    if check_end_word(relation) or 'wikidata' in relation.lower() or 'wikimedia' in relation.lower() or relation.lower() in useless_relation_list:
        return True
    return False

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt_wiki.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client):
    relations = wiki_client.query_all("get_all_relations_of_an_entity", entity_id)
    # print("@@@@wiki get relations", relations)
    if relations == "Not Found!" or not isinstance(relations, dict):
        print(f"[WARN] No relations found for entity {entity_id}")
        return []
    head_relations = relations['head']
    tail_relations = relations['tail']

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if len(pre_relations)!=0 and pre_head !=-1:
        tail_relations = [rel for rel in pre_relations if pre_head and rel not in tail_relations]
        head_relations = [rel for rel in pre_relations if not pre_head and rel not in head_relations]

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    # print([head_relations, "tail", tail_relations])
    total_relations = head_relations+tail_relations
    # print("!!!!!!!before sort total relation",sorted(total_relations))
    total_relations=sorted(total_relations)  # make sure the order in prompt is always equal
    # print("!!!!!!!total relation",question, entity_name, total_relations, args)
    prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
    print("before run llm!!!!!!!")
    result, token_num = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    print("after run llm!!!!!!!")
    flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)
    print("return", retrieve_relations_with_scores) 

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    
def del_all_unknown_entity(entity_candidates_id, entity_candidates_name):
    if len(entity_candidates_name) == 1 and entity_candidates_name[0] == "N/A":
        return entity_candidates_id, entity_candidates_name

    new_candidates_id = []
    new_candidates_name = []
    for i, candidate in enumerate(entity_candidates_name):
        if candidate != "N/A":
            new_candidates_id.append(entity_candidates_id[i])
            new_candidates_name.append(candidate)

    return new_candidates_id, new_candidates_name

def all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)

def entity_search(entity, relation, wiki_client, head):

    rid = wiki_client.query_all("label2qid", relation)
    if not rid or rid == "Not Found!":
        return [], []
    
    rid_str = rid.pop()

    entities = wiki_client.query_all("get_tail_entities_given_head_and_relation", entity, rid_str)
    
    if head:
        entities_set = entities['tail']
    else:
        entities_set = entities['head']

    if not entities_set:
        values = wiki_client.query_all("get_tail_values_given_head_and_relation", entity, rid_str)
        return [], list(values)

    id_list = entities_set
    name_list = [
        wiki_client.query_all("qid2label", qid)[0] if wiki_client.query_all("qid2label", qid) != "Not Found!" else "Unnamed_Entity"
        for qid in entities_set
    ]
    return id_list, name_list

def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)

def entity_score(question, entity_candidates_id, entity_candidates, score, relation, args):
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)

    prompt = construct_entity_score_prompt(question, relation, entity_candidates, score)

    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    entity_scores = clean_scores(result, entity_candidates)
    if all_zero(entity_scores):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    else:
        return [float(x) * score for x in entity_scores], entity_candidates, entity_candidates_id

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)

def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates

# remove score
def update_history(entity_candidates, entity, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, value_flag):
    # if value_flag:
    #     scores = [1/len(entity_candidates) * entity['score']]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    # total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)


    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


# def generate_answer(question, cluster_chain_of_entities, args): 
#     print("inside generate answer")
#     prompt = answer_prompt_wiki + question + '\n'
#     chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
#     prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
#     result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
#     return result

def generate_answer(question, subquestions, cluster_chain_of_entities, args):
    # print("inside generate answer") 
    prompt = answer_prompt_wiki + question 
    # print("inside prompt", prompt)
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt
    # print("final prompt", prompt)
    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    print("return inside generate answer",result, token_num )
    return result, token_num

# def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
#     dict = {"question":question, "turbo_results": answer, "chains": cluster_chain_of_entities}
#     with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
#         json_str = json.dumps(dict)
#         outfile.write(json_str + "\n")


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, wiki_client):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))
    tops = [wiki_client.query_all("qid2label", entity_id).pop() if (entity_name := wiki_client.query_all("qid2label", entity_id)) != "Not Found!" else "Unname_Entity" for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads

# def reasoning(question, cluster_chain_of_entities, args):
#     prompt = prompt_evaluate_wiki + question
#     chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
#     prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

#     response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    
#     result = extract_answer(response)
#     if if_true(result):
#         return True, response
#     else:
#         return False, response

def reasoning(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args):
    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()

    prompt = answer_depth_prompt + question + '\nMemory: ' + his_mem

    chain_prompt = ''

    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                chain_prompt += entid_name[topic_e] + ', ' + rela + ', ' + str(sorted_e_list) + '\n'

    prompt += "\nKnowledge Triplets:\n" + chain_prompt

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    
    answer, reason, sufficient = extract_reason_and_anwer(response)
    return response, answer, sufficient, token_num


def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    
def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False

# def entity_condition_prune(question, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, ent_rel_ent_dict, entid_name, name_entid, args, model):
#     cur_call_time = 0
#     cur_token = {'total': 0, 'input': 0, 'output': 0}

#     new_ent_rel_ent_dict = {}
#     no_prune = ['time', 'number', 'date']
#     filter_entities_id, filter_tops, filter_relations, filter_candidates, filter_head = [], [], [], [], []
#     for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
#         for h_t, r_e_dict in sorted(h_t_dict.items()):
#             for rela, e_list in sorted(r_e_dict.items()):
#                 if is_all_digits(e_list) or rela in no_prune or len(e_list) <= 1:
#                     sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
#                     select_ent = sorted_e_list
#                 else:
#                     if all(entid_name[item].startswith('m.') for item in e_list) and len(e_list) > 10:
#                         e_list = random.sample(e_list, 10)

#                     if len(e_list) > 70:
#                         sorted_e_list = [entid_name[e_id] for e_id in e_list]
#                         topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 70)
#                         e_list = [name_entid[e_n] for e_n in topn_entities]
#                         print('sentence:', topn_entities)

#                     prompt = prune_entity_prompt + question +'\nTriples: '
#                     sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
#                     prompt += entid_name[topic_e] + ' ' + rela + ' ' + str(sorted_e_list)

#                     cur_call_time += 1
#                     result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
#                     for kk in token_num.keys():
#                         cur_token[kk] += token_num[kk]

#                     last_brace_l = result.rfind('[')
#                     last_brace_r = result.rfind(']')
                    
#                     if last_brace_l < last_brace_r:
#                         result = result[last_brace_l:last_brace_r+1]
                    
#                     try:
#                         result = eval(result.strip())
#                     except:
#                         result = result.strip().strip("[").strip("]").split(', ')
#                         result = [x.strip("'") for x in result]

#                     select_ent = sorted(result)
#                     select_ent = [x for x in select_ent if x in sorted_e_list]

#                 if len(select_ent) == 0 or all(x == '' for x in select_ent):
#                     continue

#                 if topic_e not in new_ent_rel_ent_dict.keys():
#                     new_ent_rel_ent_dict[topic_e] = {}
#                 if h_t not in new_ent_rel_ent_dict[topic_e].keys():
#                     new_ent_rel_ent_dict[topic_e][h_t] = {}
#                 if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
#                     new_ent_rel_ent_dict[topic_e][h_t][rela] = []
                
#                 for ent in select_ent:
#                     if ent in sorted_e_list:
#                         new_ent_rel_ent_dict[topic_e][h_t][rela].append(name_entid[ent])
#                         filter_tops.append(entid_name[topic_e])
#                         filter_relations.append(rela)
#                         filter_candidates.append(ent)
#                         filter_entities_id.append(name_entid[ent])
#                         if h_t == 'head':
#                             filter_head.append(True)
#                         else:
#                             filter_head.append(False)


#     if len(filter_entities_id) == 0:
#         return False, [], [], [], [], new_ent_rel_ent_dict, cur_call_time, cur_token


#     cluster_chain_of_entities = [[(filter_tops[i], filter_relations[i], filter_candidates[i]) for i in range(len(filter_candidates))]]
#     return True, cluster_chain_of_entities, filter_entities_id, filter_relations, filter_head, new_ent_rel_ent_dict, cur_call_time, cur_token

# def half_stop(question, cluster_chain_of_entities, args):
#     print("No new knowledge added during search depth %d, stop searching." % args.depth)
#     answer = generate_answer(question, cluster_chain_of_entities, args)
#     save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)

import random

def entity_condition_prune(question,
                           total_entities_id,
                           total_relations,
                           total_candidates,
                           total_topic_entities,
                           total_head,
                           ent_rel_ent_dict,
                           entid_name,
                           name_entid,
                           args,
                           model,
                           wiki_client):
    """
    Prune candidate entities using live Wikidata queries via wiki_client,
    preserving original signature plus wiki_client argument.

    Returns:
      flag, cluster_chain_of_entities, filter_entities_id,
      filter_relations, filter_head, new_ent_rel_ent_dict,
      cur_call_time, cur_token
    """
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}

    new_ent_rel_ent_dict = {}
    filter_entities_id, filter_tops, filter_relations, filter_candidates, filter_head = [], [], [], [], []

    for topic_e, h_t_dict in ent_rel_ent_dict.items():
        for h_t, r_e_dict in h_t_dict.items():
            for rela, e_list in r_e_dict.items():
                # Use wiki_client to fetch actual linked entities
                pid = rela  # assume rela is already PID string like 'P31'
                if h_t == 'tail':
                    # outgoing: topic -> relation -> obj
                    linked = wiki_client.query_all("get_tail_entities_given_head_and_relation", topic_e, pid)
                    candidates = set(linked['tail']) if isinstance(linked, dict) else set()
                else:
                    # incoming: subj -> relation -> topic
                    linked = wiki_client.query_all("get_tail_entities_given_head_and_relation", None, None)  # adjust if API supports reverse
                    # fallback: skip head pruning
                    candidates = set(e_list)

                # intersect with existing list
                select_ids = [eid for eid in e_list if eid in candidates]
                if not select_ids:
                    select_ids = e_list.copy()

                # record pruning
                new_ent_rel_ent_dict.setdefault(topic_e, {}).setdefault(h_t, {})[rela] = select_ids

                for ent_id in select_ids:
                    filter_entities_id.append(ent_id)
                    filter_tops.append(entid_name.get(topic_e, topic_e))
                    filter_relations.append(rela)
                    filter_candidates.append(entid_name.get(ent_id, ent_id))
                    filter_head.append(h_t == 'head')

    if not filter_entities_id:
        return False, [], [], [], [], new_ent_rel_ent_dict, cur_call_time, cur_token

    cluster_chain_of_entities = [
        list(zip(filter_tops, filter_relations, filter_candidates))
    ]

    return True, cluster_chain_of_entities, filter_entities_id, filter_relations, filter_head, new_ent_rel_ent_dict, cur_call_time, cur_token


def half_stop(question, question_string, subquestions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args):
    print("!!!!No new knowledge added during search depth %d, stop searching." % depth)
    print(question, subquestions, cluster_chain_of_entities, args)
    call_num += 1
    print('call generate answer')
    answer, token_num = generate_answer(question, subquestions, cluster_chain_of_entities, args)
    print("inside half stop", answer, token_num)
    for kk in token_num.keys():
        all_t[kk] += token_num[kk]

    save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)


def generate_without_explored_paths(question, args):
    prompt = generate_directly + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return response

def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'test':
        with open('../data/test.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found")
        exit(-1)
    return datas, question_string


def provide_triple(entity_candidates_id, relation):
    """
    Given a list of Wikidata QIDs, look up their English labels via wiki_client,
    then sort the (label, QID) pairs alphabetically by label.  
    Returns:
      - entity_candidates: [label1, label2, …]
      - entity_candidates_id: [qid1, qid2, …] in the same order
    """
    entity_candidates = []
    # Fetch labels in batch
    for qid in entity_candidates_id:
        # query_all returns a list of labels or "Not Found!"
        labels = wiki_client.query_all("qid2label", qid)
        label = labels[0] if isinstance(labels, list) and labels else qid
        entity_candidates.append(label)

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidates_id

    # Zip, sort by label, then unzip
    pairs = sorted(zip(entity_candidates, entity_candidates_id), key=lambda x: x[0])
    sorted_labels, sorted_ids = zip(*pairs)
    return list(sorted_labels), list(sorted_ids)

def add_pre_info(add_ent_list, depth_ent_rel_ent_dict, new_ent_rel_ent_dict, entid_name, name_entid, args):
    add_entities_id = sorted(add_ent_list)
    add_relations, add_head = [], []
    topic_ent = set()

    for cur_ent in add_entities_id:
        flag = 0
        for depth, ent_rel_ent_dict in depth_ent_rel_ent_dict.items():
            for topic_e, h_t_dict in ent_rel_ent_dict.items():
                for h_t, r_e_dict in h_t_dict.items():
                    for rela, e_list in r_e_dict.items():
                        if cur_ent in e_list:
                            if topic_e not in new_ent_rel_ent_dict.keys():
                                new_ent_rel_ent_dict[topic_e] = {}
                            if h_t not in new_ent_rel_ent_dict[topic_e].keys():
                                new_ent_rel_ent_dict[topic_e][h_t] = {}
                            if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
                                new_ent_rel_ent_dict[topic_e][h_t][rela] = []
                            if cur_ent not in new_ent_rel_ent_dict[topic_e][h_t][rela]:
                                new_ent_rel_ent_dict[topic_e][h_t][rela].append(cur_ent)
                            
                            if not flag:
                                add_relations.append(rela)
                                if h_t == 'head':
                                    add_head.append(True)
                                else:
                                    add_head.append(False)
                                flag = 1


        if not flag:
            print('none pre relation')
            print(cur_ent)
            flag = 1
            add_head.append(-1)
            add_relations.append('')
            if cur_ent not in new_ent_rel_ent_dict.keys():
                new_ent_rel_ent_dict[cur_ent] = {}

    return add_entities_id, add_relations, add_head, new_ent_rel_ent_dict

def update_memory(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args):
    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()
    prompt = update_mem_prompt + question + '\nSubobjectives: '+str(subquestions)+'\nMemory: ' + his_mem

    chain_prompt = ''
    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                chain_prompt += entid_name[topic_e] + ' ' + rela + ' ' + str(sorted_e_list) + '\n'

    prompt += "\nKnowledge Triplets:\n" + chain_prompt

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    
    mem = extract_memory(response)
    print(mem)
    with open(q_mem_f_path+'/mem', 'w', encoding='utf-8') as f:
        f.write(mem)
    return token_num

def if_topic_non_retrieve(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def is_all_digits(lst):
    for s in lst:
        if not s.isdigit():
            return False
    return True