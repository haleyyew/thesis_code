import json
import pandas as pd

import parse_surrey_dataset as pds
import build_matching_model as bmm
import schema_matchers as sch
import preprocess_topic as pt

def load_metadata(p, m):
    '''TODO there might be a correct mapping between input_topics and attributes of input_datasets'''

    dataset_metadata_f = open(p.dataset_metadata_p, 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    metadata_f = open(p.metadata_p, 'r')
    metadata_set = json.load(metadata_f)

    schema_f = open(p.schema_p, 'r')
    schema_set = json.load(schema_f, strict=False)

    dataset_metadata_f.close()
    metadata_f.close()
    schema_f.close()

    m.dataset_metadata_set = dataset_metadata_set
    m.metadata_set = metadata_set
    m.schema_set = schema_set

    return

from similarity.ngram import NGram
twogram = NGram(2)
fourgram = NGram(4)
import numpy as np

from similarity.metric_lcs import MetricLCS
metric_lcs = MetricLCS()
def build_local_similarity_matrix(source_schema, target_schema, r):
    source_schema_name = list(source_schema.keys())

    matrix= np.zeros((len(source_schema_name), len(target_schema)))

    for i in range(len(source_schema_name)):
            for j in range(len(target_schema)):
                # TODO call matcher
                sim_score = 1 - twogram.distance(source_schema_name[i],target_schema[j])
                # matrix[i,j] = np.int(100*SequenceMatcher(None,source_schema[i],target_schema[j]).ratio())
                matrix[i, j] = sim_score

                if matrix[i, j] >= r.ngram_threshold:
                    print('matrix[i, j]', source_schema_name[i], target_schema[j], matrix[i, j])

                # if target_schema[j] == 'tree_species':
                #     print(source_schema[i], target_schema[j], matrix[i, j])

    return matrix

import random
def create_attributes_contexts(datasets, m, p, r):
    contexts = {}

    for dataset in datasets:
        contexts[dataset] = {}

        schema = m.schema_set[dataset]
        attributes_list = [pds.clean_name(attr['name'], False, False) for attr in schema]



        dataset_existing_tags = m.dataset_metadata_set[dataset]['tags']
        dataset_existing_groups = m.dataset_metadata_set[dataset]['groups']
        dataset_notes = m.dataset_metadata_set[dataset]['notes']

        desc = ''
        for group in dataset_existing_groups:
            desc = ' ' + group['description']

        dataset_existing_tags = [tag['display_name'] for tag in dataset_existing_tags]
        dataset_existing_groups = [group['display_name'] for group in dataset_existing_groups]
        dataset_notes = [word for word in dataset_notes.split() if "http://" not in word]

        notes = ' '.join(dataset_notes)

        stats_f = open(p.dataset_stats + dataset + '.json', 'r')
        stats = json.load(stats_f)

        df_columns = list(stats.keys())

        attributes_list = [pds.clean_name(attr['name'], False, False) for attr in schema]
        cols_to_delete = bmm.find_attrs_to_delete(attributes_list, df_columns)
        attributes_list = [item for item in attributes_list if item not in cols_to_delete]


        for attr in attributes_list:
            # other_attrs = attributes_list.copy()
            # other_attrs.remove(attr)
            other_attrs = []

            attr_values = stats[attr].keys()
            # TODO get average of val length, place attr vals in notes if length is long

            length = 0
            if len(attr_values) > 0:


                if len(attr_values) > r.vals_truncate_sample:

                    num_to_select = r.vals_truncate_sample
                    attr_values = random.sample(attr_values, num_to_select)


                length = [len(val) for val in attr_values]
                length = sum(length)/len(attr_values)

            if r.sentence_threshold <= length:
                notes = notes + '. ' + '. '.join([val for val in attr_values])
                # print('>>>>>', notes)
            else:
                other_attrs.extend(attr_values)
                # print('>>>>>', other_attrs)

            pt.enrich_homonyms(dataset, attr, desc, notes, other_attrs)


    m.dataset_attributes_contexts = contexts
    return contexts

def topic_attribute_overlap(syn_attr_dict, syn_top_dict, all_topics):
    score = 0
    pair = None
    for attr in syn_attr_dict:
        for top in syn_top_dict:
            ctx_words_attr = syn_attr_dict[attr]
            ctx_words_top = syn_top_dict[top]
            try:
                score_ctx = len(list(ctx_words_attr & ctx_words_top))
                if score_ctx > score:
                    score = score_ctx
                    pair = (top, attr)
            except Exception:
                pass
    return score, pair

def topic_attribute_syn_similarity(syn_attr_dict, syn_top_dict, function, all_topics, source_name, topic_name):
    score = 0
    pair = None
    source_matched = None
    for attr in syn_attr_dict:
        for top in syn_top_dict:
            try:

                syn_attr = function.synset(attr)
                syn_top = function.synset(top)
                syn_score = syn_attr.path_similarity(syn_top)
                # TODO store all scores

                if syn_score > score:
                    score = syn_score
                    pair = (top, attr)

            except Exception:
                pass

    other_sources = {}
    if topic_name in all_topics:
        other_sources = all_topics[topic_name]
    for source in other_sources:
        if source_name == source:
            continue
        syns_dict = other_sources[source]
        for top in syns_dict:
            for attr in syn_attr_dict:
                try:

                    syn_attr = function.synset(attr)
                    syn_top = function.synset(top)
                    syn_score = syn_attr.path_similarity(syn_top)

                    if syn_score > 0.2: print('===', source_name, syn_attr, '<>', source, syn_top, ':', syn_score)

                    # TODO store all scores

                    if syn_score > score:
                        score = syn_score
                        pair = (top, attr)
                        source_matched = source

                except Exception:

                    print('ERROR: topic_attribute_syn_similarity', top, attr)
                    pass

    return score, pair, source_matched

# import requests

# def topic_attribute_fasttext(topic, attr, server_ip):
#     response = requests.get("http://" + server_ip + ":5000/similarity/" + topic+'__'+attr)
#     # print(topic+'__'+attr)
#     ret_val = 0
#     try:
#         ret_val = float(response.json())
#     except:
#         print('fasttext err:', topic+'__'+attr)
#         pass
#     return ret_val

def build_local_context_similarity_matrix_full(topics_contexts, attributes_contexts, source_name, function, all_topics, server_ip):

    topic_names = list(all_topics.keys())
    attribute_names = list(attributes_contexts.keys())

    matrix= np.zeros((len(topic_names), len(attribute_names)))
    matrix2 = np.zeros((len(topic_names), len(attribute_names)))

    pair_dict = {}

    for i in range(len(topic_names)):
        print('~~~', topic_names[i])
        for j in range(len(attribute_names)):

            pair_dict[(source_name, attribute_names[j], topic_names[i])] = None

            ds_with_topic = all_topics[topic_names[i]]

            best_val = 0
            best_ctx = None

            for k in ds_with_topic:
                sim_score_arr = [0, 0, 0]

                syn_attr_dict = attributes_contexts[attribute_names[j]]
                # print('~~~', topic_names[i], ds_with_topic[k])
                syn_top_dict = all_topics[topic_names[i]][k]

                sim_score_arr[1], pair1, source_matched = topic_attribute_syn_similarity(syn_attr_dict, syn_top_dict, function, {}, k, topic_names[i])

                if sim_score_arr[1] > best_val:
                    matrix[i, j] = sim_score_arr[1]
                    best_ctx = k

                    pair_dict[(source_name, attribute_names[j], topic_names[i])] = [pair1, best_ctx]  # TODO need the source of topic too

                # if matrix[i, j] >= 0.5:
                #     print('matrix[i, j]', topic_names[i], attribute_names[j], matrix[i, j])

            #matrix2[i, j] = topic_attribute_fasttext(topic_names[i], attribute_names[j], server_ip)

    return matrix, pair_dict, matrix2


def build_local_context_similarity_matrix(topics_contexts, attributes_contexts, source_name, function, all_topics):

    topic_names = list(topics_contexts[source_name].keys())
    attribute_names = list(attributes_contexts.keys())

    # ADDED
    topic_names.sort()
    attribute_names.sort()

    # print('=====', topic_names, attribute_names)

    matrix= np.zeros((len(topic_names), len(attribute_names)))
    matrix2 = np.zeros((len(topic_names), len(attribute_names)))

    pair_dict = {}

    for i in range(len(topic_names)):
            for j in range(len(attribute_names)):
                # call matchers

                # sim_score = 0
                sim_score_arr = [0,0,0]
                # sim_score_arr[0] = sch.matcher_name(topic_names[i],attribute_names[j], twogram)
                ## sim_score = 1 - twogram.distance()

                syn_attr_dict = attributes_contexts[attribute_names[j]]
                syn_top_dict = topics_contexts[source_name][topic_names[i]]
                # all topics are considered as well

                # TODO wordnet distance

                sim_score_arr[1], pair1, source_matched = topic_attribute_syn_similarity(syn_attr_dict, syn_top_dict, function, all_topics, source_name, topic_names[i])
                # TODO even if the matched synset is not from this dataset, the score is still kept for the topic of this dataset

                # TODO context words overlap
                sim_score_arr[2], pair2 = topic_attribute_overlap(syn_attr_dict, syn_top_dict, all_topics)

                # take the max

                # combine scores
                # TODO not only existing topics, map to new topics here too!
                # word2vec finding is easier
                # transfering attributes to different topic groups at pairwise table comparison
                # guiding table augmented with new topics, as well as other related tables

                matrix[i, j] = sim_score_arr[1]
                matrix2[i, j] = sim_score_arr[2]

                pair_dict[(source_name, attribute_names[j], topic_names[i])] = [pair1, source_matched, pair2]       # TODO need the source of topic too

                # if matrix[i, j] >= 0.5:
                #     print('matrix[i, j]', topic_names[i], attribute_names[j], matrix[i, j])

    return matrix, matrix2, pair_dict


def load_prematching_metadata(p, m, pds):
    all_topics = {}

    topic_contexts_f = open(p.enriched_topics_json_dir, 'r')
    topic_contexts = json.load(topic_contexts_f)
    all_topics = {}

    # TODO get all available topics
    for source_name in m.datasources_with_tag:
        source_name = pds.clean_name(source_name, False, False)
        for src_top in topic_contexts[source_name]:
            if src_top not in all_topics:
                all_topics[src_top] = {}

            all_topics[src_top][source_name] = topic_contexts[source_name][src_top] # synsets


    attrs_contexts_f = open(p.enriched_attrs_json_dir, 'r')
    attrs_contexts = json.load(attrs_contexts_f)

    return all_topics, attrs_contexts, topic_contexts

def load_per_source_metadata(p, m, datasources, source_name, pds, bmm):
    # path = datasets_path + source_name + '.csv'
    # dataset = pd.read_csv(path, index_col=0, header=0)
    stats_f = open(p.dataset_stats + source_name + '.json', 'r')
    stats = json.load(stats_f)
    df_columns = list(stats.keys())

    schema = m.schema_set[source_name]
    metadata = m.dataset_metadata_set[source_name]['tags']
    dataset = pd.DataFrame()

    # dataset = bmm.df_rename_cols(dataset)

    datasources[source_name] = (source_name, dataset, schema, metadata)

    print(source_name)
    # bmm.print_metadata_head(source_name, dataset, schema, metadata)

    # initialization schema matching
    tags_list = [tag['display_name'] for tag in metadata]
    # use enriched tags instead
    tags_list_enriched_f = open(p.enriched_topics_json_dir, 'r')
    tags_list_enriched = json.load(tags_list_enriched_f)
    tags_list_enriched_dataset = tags_list_enriched[source_name]
    tags_list_enriched_names = list(tags_list_enriched[source_name].keys())
    # TODO add non-overlapping homonyms to context

    attributes_list = [pds.clean_name(attr['name'], False, False) for attr in schema]
    cols_to_delete = bmm.find_attrs_to_delete(attributes_list, df_columns)
    attributes_list = [item for item in attributes_list if item not in cols_to_delete]

    return tags_list_enriched_dataset, tags_list_enriched_names, attributes_list, schema, datasources, stats


import pprint
import os
from operator import itemgetter
def initialize_matching_full(p, m, r):

    all_topics, attrs_contexts, topic_contexts = load_prematching_metadata(p, m, pds)

    wordnet = pt.load_dict()

    pair_dict_all = {}

    kb_curr_file = './outputs/kb_file.json' # TODO <=
    if not os.path.exists(kb_curr_file):
        with open(kb_curr_file, 'w') as fp:
            json.dump({}, fp, sort_keys=True, indent=2)

    fp = open(kb_curr_file, 'r')
    m.kbs = json.load(fp)


    len_all_ds = len(m.datasources_with_tag)

    datasources = {}
    for source_name in m.datasources_with_tag:
        if source_name in m.kbs:
            print('--already have local mapping ', source_name)
            continue       # already have local mapping, skip


        _, _, attributes_list, schema, _, _ = load_per_source_metadata(
            p, m, datasources, source_name, pds, bmm)

        attributes_list_orig = [pds.clean_name(attr['name'], False, False) for attr in schema]

        score_names = m.score_names

        # 700+ topics
        sim_matrix1 = build_local_similarity_matrix(all_topics, attributes_list, r)

        if source_name not in attrs_contexts:
            print('ERROR: DATASOURCE NOT FOUND', source_name, '\n', '---!--')
            continue
        attribute_contexts = attrs_contexts[source_name]

        # topic_contexts is all datasets, attribute_contexts is per dataset
        sim_matrix2, pair_dict, sim_matrix3 = build_local_context_similarity_matrix_full(topic_contexts, attribute_contexts, source_name, wordnet, all_topics, p.server_ip)

        pprint.pprint(pair_dict)

        sim_frame1 = pd.DataFrame(data=sim_matrix1, columns=attributes_list, index=all_topics.keys())
        sim_frame2 = pd.DataFrame(data=sim_matrix2, columns=list(attribute_contexts.keys()), index=all_topics.keys())
        sim_frame3 = pd.DataFrame(data=sim_matrix3, columns=list(attribute_contexts.keys()), index=all_topics.keys())

        pair_dict_all.update(pair_dict)
        attrs = list(sim_frame1.columns.values)

        if len(attrs) == 0:
            print('ERROR: empty dataset', source_name, '\n', '-----')
            continue

        for attr_i in range(len(schema)):
            if 'domain' not in schema[attr_i] or schema[attr_i]['domain'] == None:

                attr_name = schema[attr_i]['name']
                attr_name = pds.clean_name(attr_name, False, False)

                _, uniques = bmm.get_attr_stats(p.dataset_stats, source_name, attr_name)
                if uniques != None:
                    pass
                else:
                    continue

                uniques.sort()
                schema[attr_i]['coded_values'] = uniques

                schema[attr_i]['domain'] = 'coded_values_groupby'

        # init kb
        build_kb_json(all_topics, source_name, m)

        for attr_i in range(len(attrs)):
            scores1 = [[attr_i, attrs[attr_i], sim_frame1.loc[topic, attrs[attr_i]], topic, None] for topic in all_topics]
            scores2 = [[attr_i, attrs[attr_i], sim_frame2.loc[topic, attrs[attr_i]], topic, pair_dict[(source_name,pds.clean_name(attrs[attr_i]),topic)]] for topic in all_topics]
            scores3 = [[attr_i, attrs[attr_i], sim_frame3.loc[topic, attrs[attr_i]], topic, None] for topic in
                       all_topics]

            score_len = 0
            if len(scores1) != 0: score_len = len(scores1[0])

            # TODO change score to weighted average
            weights = [0.40,0.6,0.0]    # TODO train the weights
            scores = []
            for i in range(len(scores1)):

                scores_tmp = [scores1[i][2], scores2[i][2], scores3[i][2]]
                index, element = max(enumerate(scores_tmp), key=itemgetter(1))


                # if scores1[i][2] >= scores2[i][2]:
                #     scores.append([attr_i, attrs[attr_i],scores1[i][2],scores1[i][3], scores1[i][4], score_names[0]])
                # else:
                #     scores.append([attr_i, attrs[attr_i],scores2[i][2],scores2[i][3], scores2[i][4], score_names[1]])


                score_tmp = weights[0]*scores_tmp[0]+weights[1]*scores_tmp[1]+weights[2]*scores_tmp[2]

                scores.append([attr_i, attrs[attr_i], score_tmp, scores2[i][3], scores2[i][4], score_names[index]])


            scores = sorted(scores, key=lambda tup: tup[2])
            scores.reverse()
            scores_examples = []
            for attr_score in scores:
                # attr_splt = attr_score[1].split()
                ind = attributes_list_orig.index(attr_score[1])
                if 'coded_values' not in schema[ind]:
                    continue
                arg_max_examples_vals = schema[ind]['coded_values']
                arg_max_examples_vals.sort()
                scores_examples.append(attr_score + [arg_max_examples_vals] )


            top = 0
            output = []
            for score in scores_examples:
                if len(score) <= score_len:
                    continue
                if score[2] > r.topic_to_attr_threshold and top <= r.topic_to_attr_count:
                    output.append(score)
                    top += 1

            for match in output:
                kb_match_entry = {'concept': match[3],
                                  'datasource': source_name,
                                  'attribute': match[1],
                                  'match_score': match[2],
                                  'example_values': match[6],
                                  'topic_source': match[5],
                                  'data_type': schema[match[0]]['data_type'],
                                  'score_name': match[4]}

                update_kb_json(m.kbs[source_name], kb_match_entry)

                # for debugging:
                kb_match_entry['example_values'] = kb_match_entry['example_values'][:min(len(kb_match_entry['example_values']), 5)]
                pprint.pprint(kb_match_entry)


        with open(p.schema_p, 'w') as fp:
            json.dump(m.schema_set, fp, sort_keys=True, indent=2)

        with open(kb_curr_file, 'w') as fp:
            json.dump(m.kbs, fp, sort_keys=True, indent=2)

        print('done saving kb_file', source_name)
        print('^^^ PROGRESS', len(m.kbs)/len_all_ds)


    return

def initialize_matching(p, m, r):

    all_topics, attrs_contexts, topic_contexts = load_prematching_metadata(p, m, pds)

    wordnet = pt.load_dict()

    pair_dict_all = {}

    datasources = {}
    for source_name in m.datasources_with_tag:
        tags_list_enriched_dataset, tags_list_enriched_names, attributes_list, schema, _, _ = load_per_source_metadata(p, m, datasources, source_name, pds, bmm)

        score_names = m.score_names

        sim_matrix1 = build_local_similarity_matrix(tags_list_enriched_dataset, attributes_list, r)
        # TODO build_local_similarity_matrix using context

        if source_name not in attrs_contexts:
            print('ERROR: DATASOURCE NOT FOUND', source_name, '\n', '-----')
            continue
        attribute_contexts = attrs_contexts[source_name]

        # topic_contexts is all datasets, attribute_contexts is per dataset
        sim_matrix2, sim_matrix3, pair_dict = build_local_context_similarity_matrix(topic_contexts, attribute_contexts, source_name, wordnet, all_topics)

        sim_frame1 = pd.DataFrame(data=sim_matrix1, columns=attributes_list, index=tags_list_enriched_names)
        sim_frame2 = pd.DataFrame(data=sim_matrix2, columns=attributes_list, index=tags_list_enriched_names)
        sim_frame3 = pd.DataFrame(data=sim_matrix3, columns=attributes_list, index=tags_list_enriched_names)

        # chance of getting external topics
        pair_dict_all.update(pair_dict)

        # print(sim_frame.to_string())

        attrs = list(sim_frame1.columns.values)

        # if stats file is empty
        if len(attrs) == 0:
            print('ERROR: empty dataset', source_name, '\n', '-----')
            continue


        # get example values
        for attr_i in range(len(schema)):
            # print(attr_i)
            if 'domain' not in schema[attr_i] or schema[attr_i]['domain'] == None:

                attr_name = schema[attr_i]['name']
                attr_name = pds.clean_name(attr_name, False, False)

                # loading from stats file
                _, uniques = bmm.get_attr_stats(p.dataset_stats, source_name, attr_name)
                if uniques != None:
                    # print('uniques', len(uniques))
                    pass
                else:
                    continue

                # stat, _, uniques = bmm.groupby_unique(attrs[arg_i], dataset)

                uniques.sort()
                schema[attr_i]['coded_values'] = uniques
                # arg_max_examples_vals = schema[attr_i]['coded_values']

                # if len(arg_max_examples_vals) > 0: print('arg_max_examples_vals', arg_max_examples_vals[0])

                schema[attr_i]['domain'] = 'coded_values_groupby'

        # init kb
        build_kb_json(tags_list_enriched_names, source_name, m)

        # during new concepts stage, add second best tag and so on

        for attr_i in range(len(attrs)):
            scores1 = [[attr_i, attrs[attr_i], sim_frame1.loc[topic, attrs[attr_i]], topic] for topic in tags_list_enriched_names]
            scores2 = [[attr_i, attrs[attr_i], sim_frame2.loc[topic, attrs[attr_i]], topic] for topic in tags_list_enriched_names]
            scores3 = [[attr_i, attrs[attr_i], sim_frame3.loc[topic, attrs[attr_i]], topic] for topic in
                       tags_list_enriched_names]

            score_len = 0
            if len(scores1) != 0: score_len = len(scores1[0])


        # for topic in tags_list_enriched_names:
        #     scores1 = [[attr_i, attrs[attr_i], sim_frame1.loc[topic, attrs[attr_i]]] for attr_i in range(len(attrs))]
        #     scores2 = [[attr_i, attrs[attr_i], sim_frame2.loc[topic, attrs[attr_i]]] for attr_i in range(len(attrs))]

            scores = []
            for i in range(len(scores1)):
                if scores1[i][2] >= scores2[i][2]:
                    # print(scores2[i][2])
                    # print(scores3[i][2])
                    scores.append([attr_i, attrs[attr_i],scores1[i][2],scores1[i][3], score_names[0]])
                else:
                    multiplier = 1.0
                    if scores3[i][2] != 0: multiplier = scores3[i][2]
                    scores.append([attr_i, attrs[attr_i], min(scores2[i][2]*multiplier, 1.0), scores2[i][3], score_names[1]])

            scores = sorted(scores, key=lambda tup: tup[2])
            scores.reverse()
            scores_examples = []
            for attr_score in scores:
                # example_value = None
                # print(attr_score, attr_score[0], schema[attr_score[0]])
                if 'coded_values' not in schema[attr_score[0]]:
                    continue
                arg_max_examples_vals = schema[attr_score[0]]['coded_values']
                arg_max_examples_vals.sort()
                scores_examples.append(attr_score + [schema[attr_score[0]]['coded_values']] )
                # print('here')

            top = 0
            output = []
            for score in scores_examples:
                if len(score) <= score_len:
                    # print('skip', score)
                    continue
                # print('topic_to_attr_count', score[2], top)
                if score[2] > r.topic_to_attr_threshold and top <= r.topic_to_attr_count:
                    # print('topic_to_attr_count', r.topic_to_attr_count)
                    output.append(score)
                    top += 1
            # if len(output) == 0:
            #     output.append(scores_examples[0])

            # max_score = 0
            # arg_max_score = None
            # arg_i = -1
            # for attr_i in range(len(attrs)):
            #     attr = attrs[attr_i]
            #     score = sim_frame.loc[topic, attr]
            #     if score > max_score:
            #         max_score = score
            #         arg_max_score = attr
            #         arg_i = attr_i



            # if len(arg_max_examples_vals) > 0: example_value = arg_max_examples_vals[0]
            # print('best match:', topic, arg_max_score, max_score, example_value)

            # print('=====output', output)

            for match in output:
                kb_match_entry = {'concept': match[3],
                                  'datasource': source_name,
                                  'attribute': match[1],
                                  'match_score': match[2],
                                  'example_values': match[5],
                                  'data_type': schema[match[0]]['data_type'],
                                  'score_name': match[4]}

                update_kb_json(m.kbs[source_name], kb_match_entry)

                # for debugging:
                kb_match_entry['example_values'] = kb_match_entry['example_values'][:min(len(kb_match_entry['example_values']), 5)]
                pprint.pprint(kb_match_entry)

        print('-----')

    m.pair_dict_all = pair_dict_all

    # done initialization

    return True

def build_kb_json(list_of_concepts, dataset_name, m):

    kb = {}
    for concept in list_of_concepts:
        concept_name = concept
        if concept_name not in kb:
            kb[concept_name] = {}
            # kb[concept_name]['datasources'] = datasources
            # kb[concept_name]['matches'] = {}
        else:
            kb_concept = kb[concept_name]
            # kb_concept['datasources'].extend(datasources)
        # TODO remove duplicates

    m.kbs[dataset_name] = kb
    return

def update_kb_json(kb, match_entry):
    concept = match_entry['concept']
    datasource = match_entry['datasource']
    attribute = match_entry['attribute']
    match_score = match_entry['match_score']
    example_values = match_entry['example_values']
    data_type = match_entry['data_type']

    if concept not in kb: kb[concept] = {} # TODO this is work around
    kb_concept = kb[concept]
    # kb_concept_matches = kb_concept['matches']
    # kb_concept_matches[datasource] =
    kb_concept[attribute] = {'attribute': attribute, 'match_score' : match_score, 'example_values' : example_values, 'data_type' : data_type}

    if 'topic_source' in match_entry: kb_concept[attribute]['topic_source'] = match_entry['topic_source']
    return

class Paths:
    datasets_path = './thesis_project_dataset_clean/'
    dataset_stats = './inputs/dataset_statistics/'

    dataset_metadata_p = './inputs/datasource_and_tags.json'
    metadata_p = './inputs/metadata_tag_list_translated.json'
    schema_p = './inputs/schema_complete_list.json'

    matching_output_p = './outputs/instance_matching_output/'
    from time import gmtime, strftime
    curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    kb_file_p = "./outputs/kb_file_"+curr_time+".json"
    kb_file_const_p = "./outputs/kb_file_local_mapping.json"
    pair_dict_all_p = "./outputs/pair_dict_all_" + curr_time + ".json"
    dataset_topics_p = "./outputs/dataset_topics_"+curr_time+".json"

    # new_concepts_p = "./outputs/new_concepts.json"
    # new_concepts_f = './outputs/new_concepts.csv'

    enriched_attrs_json_dir = './outputs/dataset_attrs_enriched.json'
    enriched_topics_dir = './outputs/enriched_topics/'
    enriched_topics_json_dir = "./outputs/dataset_topics_enriched.json"

    server_ip = '34.222.58.64'

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

p = Paths()

class Metadata:
    dataset_metadata_set = None
    metadata_set = None
    schema_set = None
    datasources_with_tag = None
    kbs = {}
    pair_dict_all = None
    score_names = ['ngram', 'wordnet', 'fasttext']

    dataset_topics_contexts = None
    dataset_attributes_contexts = None

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

m = Metadata()

class Parameters:
    topic_to_attr_threshold = 0.5
    topic_to_attr_count = 3
    ngram_threshold = 0.6

    sentence_threshold = 30
    vals_truncate_sample = 100

    table_similarity_thresh = 0.4

    num_iters = 5

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

r = Parameters()

def local_mappings_full(p,m,r):
    initialize_matching_full(p, m, r)




def local_mappings(p,m,r):
    initialize_matching(p, m, r)
    with open(p.schema_p, 'w') as fp:
        json.dump(m.schema_set, fp, sort_keys=True, indent=2)

    print('done saving schema_set')

    with open(p.kb_file_p, 'w') as fp:
        json.dump(m.kbs, fp, sort_keys=True, indent=2)

    print('done saving kb_file')

    # with open(p.pair_dict_all_p, 'w') as fp:
    #     json.dump(m.pair_dict_all, fp, sort_keys=True, indent=2)
    #
    # print('done saving pair_dict_all')


    keys = list(m.pair_dict_all.keys())

    for key in keys:
        pair = m.pair_dict_all[key]
        if pair[1] == None:
            del m.pair_dict_all[key]
    pprint.pprint(m.pair_dict_all)

    # TODO some match scores are 1 for unrelated attr-topic pair, which is clearly wrong

if __name__ == "__main__":

    load_metadata(p, m)
    local_mappings_full(p, m, r)

    exit(0)

    # GLAV mapping for each dataset
    # m.datasources_with_tag = ['aquatic hubs','drainage 200 year flood plain','drainage water bodies','park specimen trees','parks']
    # m.datasources_with_tag = ['park screen trees']
    m.datasources_with_tag = ['aquatic hubs', 'drainage 200 year flood plain', 'drainage water bodies',
                              'park specimen trees', 'parks', 'park screen trees']
    load_metadata(p, m)

    ## run preprocess topic enrich_homonyms()
    ## TODO call this to generate enriched attrs before running the rest
    # print('create_attributes_contexts:')
    # create_attributes_contexts(m.datasources_with_tag, m, p, r)
    # exit(0)
    ## then run script_enriched_topics_to_json.py
    ## then run the code below
    ## next run build_matching_model_new_global

    local_mappings(p,m,r)