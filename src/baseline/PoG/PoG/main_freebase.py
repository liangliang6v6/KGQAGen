from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import os
import pprint

def repeat_unanswer(dataset, datas, question_string, model_name):
    answered_set = set()
    new_data = []

    file_path = 'PoG_'+dataset+'_'+model_name+'.jsonl'
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line) 
            answered_set.add(data[question_string])

    for x in datas:
        if x[question_string] not in answered_set:
            new_data.append(x)
    print(len(new_data))

    return new_data

def get_one_data(datas, question_string, question):
    for data in datas:
        if data[question_string] == question:
            return [data]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=4096, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.3, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0.3, help="the temperature in reasoning stage.")
    parser.add_argument("--depth", type=int,
                        default=4, help="choose the search depth of PoG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")

    args = parser.parse_args()
    datas, question_string = prepare_dataset(args.dataset)
    datas = repeat_unanswer(args.dataset, datas, question_string, args.LLM_type)
    # if findone:
    #     datas = get_one_data(datas, question_string, 'Which countries both contain the Delnita River and fall in Eastern Europe?')
    #     print(datas)
    model = SentenceTransformer('../../../models/sentence-transformers/msmarco-distilbert-base-tas-b')
    part_q = False
    if part_q:
        q_set = []
        f = open('../eval/analysis_question', 'r', encoding='utf-8')
        for line in f.readlines():
            q_set.append(line.strip())

    print("Start Running PoG on %s dataset." % args.dataset)

    for data in tqdm(datas):
        if part_q and data[question_string] not in q_set:
            continue
        try:
            start_time = time.time()
            call_num = 0
            all_t = {'total': 0, 'input': 0, 'output': 0}

            question = data[question_string]
            print('New question start:', question)
            q_mem_f_path = '../mem/'+args.dataset+'/'+args.LLM_type+'/'+question[:255]
            if not os.path.exists(q_mem_f_path):
                os.makedirs(q_mem_f_path)
            with open(q_mem_f_path+'/mem', 'w', encoding='utf-8') as f:
                pass

            call_num += 1
            sub_questions, token_num = get_subquestions(q_mem_f_path, question, args)
            for kk in token_num.keys():
                all_t[kk] += token_num[kk]

            topic_entity = data['topic_entity']
            cluster_chain_of_entities = []
            depth_ent_rel_ent_dict = {}
            reverse_rec = {'time': 0, 'ent': []}

            entid_name = {}
            name_entid = {}
            for e_id, e_name in topic_entity.items():
                entid_name[e_id] = e_name
                name_entid[e_name] = e_id

            if len(topic_entity) == 0:
                call_num += 1
                results, token_num = generate_without_explored_paths(question, sub_questions, args)
                for kk in token_num.keys():
                    all_t[kk] += token_num[kk]
                
                new_e_rev_list = [entid_name[x] for x in reverse_rec['ent']]
                reverse_rec['ent'] = new_e_rev_list
                save_2_jsonl(question, question_string, results, [], call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)
                continue

            pre_relations = []
            pre_heads= [-1] * len(topic_entity)
            flag_printed = False
            for depth in range(1, args.depth+1):
                current_entity_relations_list = []
                i=0
                for entity in topic_entity:
                    if entity!="[FINISH_ID]":
                        call_num += 1
                        retrieve_relations, token_num = relation_search_prune(entity, sub_questions, topic_entity[entity], pre_relations, pre_heads[i], question, args)
                        for kk in token_num.keys():
                            all_t[kk] += token_num[kk]
                        current_entity_relations_list.extend(retrieve_relations)
                    i+=1
                total_candidates = []
                total_relations = []
                total_entities_id = [] 
                total_topic_entities = [] 
                total_head = []

                ent_rel_ent_dict = {} # e->head/tail->rel->ent
                for ent_rel in current_entity_relations_list:
                    if ent_rel['entity'] not in ent_rel_ent_dict.keys():
                        ent_rel_ent_dict[ent_rel['entity']] = {}

                    if ent_rel['head']:
                        head_or_tail = 'head'
                        entity_candidates_id = entity_search(ent_rel['entity'], ent_rel['relation'], True)
                    else:
                        head_or_tail = 'tail'
                        entity_candidates_id = entity_search(ent_rel['entity'], ent_rel['relation'], False)
                    
                    if len(entity_candidates_id) == 0:
                        print('the relations without tail entity:', ent_rel)
                        continue

                    entity_candidates, entity_candidates_id = provide_triple(entity_candidates_id, ent_rel['relation'])

                    name_entid.update(dict(zip(entity_candidates, entity_candidates_id)))
                    entid_name.update(dict(zip(entity_candidates_id, entity_candidates)))

                    if head_or_tail not in ent_rel_ent_dict[ent_rel['entity']].keys():
                            ent_rel_ent_dict[ent_rel['entity']][head_or_tail] = {}

                    if ent_rel['relation'] not in ent_rel_ent_dict[ent_rel['entity']][head_or_tail].keys():
                        ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']] = []

                    # store current entities into ent_rel_ent_dict
                    for retrive_ent in entity_candidates_id:
                        if retrive_ent not in ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']]:
                            ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']].append(retrive_ent)
                    
                    total_candidates, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, ent_rel, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head)
                
                depth_ent_rel_ent_dict[depth] = ent_rel_ent_dict
                
                pprint.pprint(convert_dict_name(ent_rel_ent_dict, entid_name))

                if len(total_candidates) == 0:
                    new_e_rev_list = [entid_name[x] for x in reverse_rec['ent']]
                    reverse_rec['ent'] = new_e_rev_list
                    half_stop(question, question_string, sub_questions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args)
                    flag_printed = True
                    break
                
                flag, chain_of_entities, entities_id, pre_relations, pre_heads, new_ent_rel_ent_dict,  cur_call_time, cur_token = entity_condition_prune(question, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, ent_rel_ent_dict, entid_name, name_entid, args, model)
                cluster_chain_of_entities.append(chain_of_entities)

                call_num += cur_call_time
                for kk in cur_token.keys():
                    all_t[kk] += cur_token[kk]

                pprint.pprint(convert_dict_name(new_ent_rel_ent_dict, entid_name))
                if flag:
                    call_num += 1
                    token_num = update_memory(question, sub_questions, new_ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args)
                    for kk in token_num.keys():
                        all_t[kk] += token_num[kk]

                    call_num += 1
                    results, answer, sufficient, token_num = reasoning(question, sub_questions, new_ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args)
                    for kk in token_num.keys():
                        all_t[kk] += token_num[kk]


                    if str(answer).lower() == 'null' or str(answer).lower() == 'none'  or str(answer).startswith('m.') or str(answer).startswith('[\"m.') or str(answer).startswith("['m.") or 'yes' not in str(sufficient).lower():
                        stop = False
                    else:
                        stop = True

                    if stop:
                        print("PoG stoped at depth %d." % depth)
                        new_e_rev_list = [entid_name[x] for x in reverse_rec['ent']]
                        reverse_rec['ent'] = new_e_rev_list
                        save_2_jsonl(question, question_string, results, cluster_chain_of_entities, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)
                        flag_printed = True
                        break
                    else:
                        print("depth %d still not find the answer." % depth)
                        add_ent_list = []
                        if reverse_rec['time']<5:
                            entities_id, add_ent_list, cur_call_time, cur_token = if_finish_list(question, entities_id, depth_ent_rel_ent_dict, entid_name, name_entid, q_mem_f_path, results, cluster_chain_of_entities, args, model)
                            call_num += cur_call_time
                            for kk in cur_token.keys():
                                all_t[kk] += cur_token[kk]
                            add_ent_list = [ent for ent in add_ent_list if ent not in reverse_rec['ent']]

                        # update: pre_relations, pre_heads, new_ent_rel_ent_dicts
                            if add_ent_list:
                                reverse_rec['time'] += 1
                                reverse_rec['ent'] += add_ent_list

                                add_ent_list, add_pre_relations, add_pre_heads, new_ent_rel_ent_dict = add_pre_info(add_ent_list, depth_ent_rel_ent_dict, new_ent_rel_ent_dict, entid_name, name_entid, args) 
                                pre_relations += add_pre_relations
                                pprint.pprint(convert_dict_name(ent_rel_ent_dict, entid_name))
                                pre_heads += add_pre_heads
                                entities_id += add_ent_list


                        if not entities_id or depth>5:
                            new_e_rev_list = [entid_name[x] for x in reverse_rec['ent']]
                            reverse_rec['ent'] = new_e_rev_list
                            half_stop(question, question_string, sub_questions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args)
                            flag_printed = True
                            break
                        else:
                            topic_entity = {}
                            for entity in entities_id:
                                if if_topic_non_retrieve(entity):
                                    continue
                                if entity.startswith("m."):
                                    topic_entity[entity] = entid_name[entity]

                else:
                    new_e_rev_list = [entid_name[x] for x in reverse_rec['ent']]
                    reverse_rec['ent'] = new_e_rev_list
                    half_stop(question, question_string, sub_questions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args)
                    flag_printed = True
                    break
            
            if not flag_printed:
                call_num += 1
                results, token_num = generate_without_explored_paths(question, sub_questions, args)
                for kk in token_num.keys():
                    all_t[kk] += token_num[kk]
                
                new_e_rev_list = [entid_name[x] for x in reverse_rec['ent']]
                reverse_rec['ent'] = new_e_rev_list
                save_2_jsonl(question, question_string, results, [], call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)
        except:
            continue
