import json
import pandas as pd

import build_matching_model_new as bmmn
import parse_surrey_dataset as pds
import build_matching_model as bmm
import preprocess_topic as pt
import pprint
import schema_matchers as sch
import classification_evaluation as ce
import time
import script_enriched_topics_to_json as settj

class Metadata2:
    guiding_table_name = None

    kbs = {}
    pair_dict_all = None

    enriched_attrs = None
    enriched_topics = None

    attrs_contexts = None
    topic_contexts = None
    all_topics = None

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

m2 = Metadata2()

class TableMetadata:
    table_name = None

    tags_list_enriched_dataset = None
    tags_list_enriched_names = None

    attributes_list = None
    schema = None
    attribute_contexts = None

    exposed_topics = None
    dataset_stats = None
    exposed_topics_groups = {}

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# for compare guiding table topics with other tables topics and get most similar table
def process_scores(sim_matrix1, sim_matrix2, sim_matrix3, table_metadata_topics, source, table_scores, gtm_topics):
    sim_frame1 = pd.DataFrame(data=sim_matrix1, columns=table_metadata_topics, index=gtm_topics)
    sim_frame2 = pd.DataFrame(data=sim_matrix2, columns=table_metadata_topics, index=gtm_topics)
    sim_frame3 = pd.DataFrame(data=sim_matrix3, columns=table_metadata_topics, index=gtm_topics)

    # print(sim_frame2.head())

    frames = [sim_frame1, sim_frame2, sim_frame3]
    # TODO sim_frame3 doesn't look right

    cols = list(sim_frame1.columns.values)

    list_of_scores = []

    curr_max = 0
    arg_cur_max = []
    for i in range(len(frames)):
        frame = frames[i]
        for col in cols:
            max_row = frame[col].idxmax()
            max_val = frame.loc[max_row, col]

            # if max_val > 0: print([i, max_row, col, max_val])
            list_of_scores.append([max_val, [i, max_row, col]])

            if max_val > curr_max:
                arg_cur_max = [i, max_row, col]
                curr_max = max_val

    #  NOTE: need update all topics, not the best one
    print(source, ' : ', [curr_max, arg_cur_max])

    if curr_max > bmmn.r.table_similarity_thresh:
        if source not in table_scores:
            table_scores[source] = []
        table_scores[source].extend(list_of_scores)

    return table_scores, list_of_scores

# for transfer new similar topics
def topic_topic_update(dataset, info, score, comparing_pairs, gtm, gtm_kb, dataset_kb, added_topics):
    # for fine grained matching later
    if info[1] not in gtm_kb: gtm_kb[info[1]] = {}
    if info[2] not in dataset_kb: dataset_kb[info[2]] = {}

    comparing_pairs[(info[1], dataset, info[2])] = (gtm_kb[info[1]].copy(), dataset_kb[info[2]].copy())

    # transfer
    update = dataset_kb[info[2]].copy()
    for attr in update:
        if 'source_dataset' not in dataset_kb[info[2]][attr]:
            update[attr]['source_dataset'] = [dataset]
        else:
            update[attr]['source_dataset'].append(dataset)

    gtm_kb[info[1]].update(update)
    gtm_kb[info[2]] = gtm_kb[info[1]]  # NOTE: put in one group instead

    # put dataset's topic into the same group as guiding table
    if info[1] not in gtm.exposed_topics_groups:
        gtm.exposed_topics_groups[info[1]] = []
    gtm.exposed_topics_groups[info[1]].append((dataset, info[2], score))

    print('----- transfer -----')
    pprint.pprint(gtm_kb[info[1]].keys())
    # pprint.pprint(gtm_kb[info[2]].keys())
    print('----------')

    if dataset not in added_topics:
        added_topics[dataset] = []
    # print('add', dataset, info[2])

    if info[2] not in added_topics[dataset]:
        added_topics[dataset].append(info[2])  # these topics will be removed later

    return comparing_pairs, gtm, gtm_kb, added_topics

# for coverage
# NOTE: check to see if any topics in other table can be mapped to non-assigned attrs in guiding table. use attr-topic wordnet, ngram name
def gt_nonmapped_attr_to_new_topic(table_scores, table_metadata, guiding_table, wordnet):
    gt_attrs = guiding_table.attributes_list
    gt_exposed_topics = guiding_table.exposed_topics

    gt_attrs_mapped = []

    gtm_kb = m2.kbs[m2.guiding_table_name]

    for topic in gtm_kb:
        for attr in gtm_kb[topic]:

            if 'source_dataset' in gtm_kb[topic][attr] and m2.guiding_table_name in gtm_kb[topic][attr][
                'source_dataset']:
                gt_attrs_mapped.append(attr)
            else:
                gt_attrs_mapped.append(attr)

    # pprint.pprint(gt_attrs_mapped)

    candidate_attrs = list(set(guiding_table.attribute_contexts.keys()) - set(gt_attrs_mapped))

    attribute_contexts_temp = guiding_table.attribute_contexts.copy()
    attribute_contexts_temp_keys = list(attribute_contexts_temp.keys())
    for attr in attribute_contexts_temp_keys:
        if attr not in candidate_attrs:
            del attribute_contexts_temp[attr]

    # print(attribute_contexts_temp.keys())

    table_scores_attr_topic = {}

    for dataset in table_scores:
        scores = sorted(table_scores[dataset], key=lambda x: x[0])
        scores.reverse()

        candidates = []
        for item in scores:
            # if item[0] < bmmn.r.topic_to_attr_threshold:
            #     break
            if item[0] == 0:
                break
            candidates.append(item[1][2])

        not_added = list(set(candidates) - set(gt_exposed_topics))

        curr_exposed_topics = list(table_metadata[dataset].tags_list_enriched_dataset.keys())

        # print(dataset, curr_exposed_topics, not_added, candidates, gt_exposed_topics)

        tags_list_enriched_temp = table_metadata[dataset].tags_list_enriched_dataset.copy()
        for topic in curr_exposed_topics:
            if topic not in not_added:
                del tags_list_enriched_temp[topic]

        table_scores_attr_topic[dataset] = [[0, None]]

        if len(tags_list_enriched_temp.keys()) == 0: continue
        # print(dataset, tags_list_enriched_temp.keys())

        sim_matrix2, sim_matrix3, _ = bmmn.build_local_context_similarity_matrix({dataset: tags_list_enriched_temp},
                                                                                 attribute_contexts_temp, dataset,
                                                                                 wordnet, {})

        sim_matrix1 = bmmn.build_local_similarity_matrix(tags_list_enriched_temp,
                                                         list(attribute_contexts_temp.keys()), bmmn.r)

        table_scores_attr_topic, list_of_scores = process_scores(sim_matrix1, sim_matrix2, sim_matrix3,
                                                                 list(attribute_contexts_temp.keys()), dataset,
                                                                 table_scores_attr_topic,
                                                                 list(tags_list_enriched_temp.keys()))

    return table_scores_attr_topic

def find_source_of_topic(added_topics, topic):
    compose_ctx = []

    for dataset in added_topics:
        if topic in added_topics[dataset]:
            compose_ctx.append(dataset)
    return compose_ctx

# for importance
# NOTE: check to see if any non-assigned attrs in other table can be mapped to guiding table topics. use attr-topic wordnet, ngram name
def gt_topic_to_dataset_nonmapped_attr(dataset, gtm, added_topics, table_scores_topic_attr, topic_from, table_metadata, wordnet):
    added_attrs = []
    dataset_attrs = table_metadata[dataset].attribute_contexts

    gtm_kb = m2.kbs[m2.guiding_table_name]

    for topic in gtm_kb:
        for attr in gtm_kb[topic]:

            if 'source_dataset' in gtm_kb[topic][attr] and dataset in gtm_kb[topic][attr]['source_dataset']:
                added_attrs.append(attr)

    candidate_attrs = list(set(dataset_attrs.keys()) - set(added_attrs))

    gtm_topics = gtm.exposed_topics
    gtm_contexts = gtm.tags_list_enriched_dataset

    # contexts can be from any dataset, not just gt
    compose_topic_ctx = {}

    for topic in gtm_topics:
        if topic == None: continue
        # aux info used later
        if topic not in topic_from:
            topic_from[topic] = []

        # the context could be from the gt or from another dataset
        if topic in gtm_contexts:
            compose_topic_ctx[topic] = gtm_contexts[topic].copy()

            topic_from[topic].append(m2.guiding_table_name)
        else:
            datasets_with_topic = find_source_of_topic(added_topics, topic)
            topic_ctx = {}
            for ds in datasets_with_topic:
                update = table_metadata[ds].tags_list_enriched_dataset[topic].copy()
                topic_ctx.update(update)
            compose_topic_ctx[topic] = topic_ctx

            topic_from[topic].append(m2.guiding_table_name)

    attr_ctx = {}
    for attr in candidate_attrs:
        attr_ctx[attr] = dataset_attrs[attr]

    # pprint.pprint(compose_topic_ctx)
    # pprint.pprint(attr_ctx)

    sim_matrix2, sim_matrix3, _ = bmmn.build_local_context_similarity_matrix({dataset: compose_topic_ctx}, attr_ctx,
                                                                             dataset, wordnet, {})

    print(compose_topic_ctx.keys())
    sim_matrix1 = bmmn.build_local_similarity_matrix(compose_topic_ctx, list(attr_ctx.keys()), bmmn.r)

    table_scores_attr_topic, list_of_scores = process_scores(sim_matrix1, sim_matrix2, sim_matrix3,
                                                             list(attr_ctx.keys()), dataset,
                                                             table_scores_topic_attr,
                                                             list(compose_topic_ctx.keys()))

    return table_scores_attr_topic, topic_from

def preprocesss_attr_values(values):
    splits = []

    if values == None: return ''

    for value in values:
        value = str(value)  # TODO get alpha to numeric ratio, if all numbers then skip
        value.replace('-', '')
        value.replace('.', '')
        val_spt = pt.splitter.split(value.lower())
        val_spt = [val for val in val_spt if 'http' not in val]
        # print(val_spt)
        val_spt_merge = []
        for item in val_spt:
            val_spt_merge.extend(item)
        splits.append(' '.join(val_spt_merge))

    return ' '.join(splits)
# preprocesss_attr_values(['I am splitting this text.','This some nonsense text qwertyuiop'])


# add more topics that have no attrs mapped, NOTE: every dataset in group gets all topics in group
def add_more_topics(exposed_topics_groups, dataset_topics):
    for key in exposed_topics_groups:
        items_list = exposed_topics_groups[key]

        # print(key, items_list)

        unique_topics = []
        unique_datasets = []
        for tupl in items_list:
            dataset, topic, score = tupl[0], tupl[1], tupl[2]

            if key not in unique_topics:
                unique_topics.append(key)

            if topic not in unique_topics and topic != None:
                unique_topics.append(topic)

            if dataset not in unique_datasets:
                unique_datasets.append(dataset)

        # print(unique_topics, unique_datasets)

        # for tupl in items_list:
        #     dataset, topic, score = tupl[0], tupl[1], tupl[2]
        for dataset in unique_datasets:
            if dataset not in dataset_topics:
                dataset_topics[dataset] = []
            if m2.guiding_table_name not in dataset_topics:
                dataset_topics[m2.guiding_table_name] = []

            for topic in unique_topics:

                if topic not in dataset_topics[dataset]:
                    dataset_topics[dataset].append(topic)
                if topic not in dataset_topics[m2.guiding_table_name]:
                    dataset_topics[m2.guiding_table_name].append(topic)
    return

def reverse_dict(dict):
    reversed = {}
    for k in dict:
        for val in dict[k]:
            if val not in reversed:
                reversed[val] = []
            if k not in reversed[val]:
                reversed[val].append(k)
    return reversed

def one_full_run(guiding_table_name, datasources_with_tag):



    # bmmn.m.datasources_with_tag = ['aquatic hubs','drainage 200 year flood plain','drainage water bodies','park specimen trees', 'parks', 'park screen trees'] #
    bmmn.m.datasources_with_tag = datasources_with_tag

    bmmn.load_metadata(bmmn.p, bmmn.m)

    m2.all_topics, m2.attrs_contexts, m2.topic_contexts = bmmn.load_prematching_metadata(bmmn.p, bmmn.m, pds)

    kb_file_f = open(bmmn.p.kb_file_const_p, 'r')
    m2.kbs = json.load(kb_file_f)
    kb_file_f.close()

    enriched_attrs_f = open(bmmn.p.enriched_attrs_json_dir, 'r')
    m2.enriched_attrs = json.load(enriched_attrs_f)
    enriched_attrs_f.close()

    enriched_topics_f = open(bmmn.p.enriched_topics_json_dir, 'r')
    m2.enriched_topics = json.load(enriched_topics_f)
    enriched_topics_f.close()

    wordnet = pt.load_dict()

    # load guiding table
    # m2.guiding_table_name = 'parks'
    m2.guiding_table_name = guiding_table_name

    gtm = TableMetadata(table_name=m2.guiding_table_name)
    gtm.tags_list_enriched_dataset, gtm.tags_list_enriched_names, gtm.attributes_list, gtm.schema, _, _ = bmmn.load_per_source_metadata(bmmn.p, bmmn.m, {}, m2.guiding_table_name, pds, bmm)
    gtm.attribute_contexts = m2.attrs_contexts[m2.guiding_table_name]
    gtm.exposed_topics = gtm.tags_list_enriched_names
    gtm.dataset_stats = bmm.get_table_stats(bmmn.p.dataset_stats, m2.guiding_table_name)

    # TODO add enriched topics to dataset if the topic is in vocab
    # TODO find more topics for nonmapped attrs

    # load all other tables


    table_metadata = {}
    added_topics = {}
    num_iters = 0
    # start of iteration:
    break_out = False

    while not break_out:
        print('===== num_iters', num_iters, '=====')
        # compare guiding table topics with other tables topics and get most similar table

        table_scores = {}
        for source in bmmn.m.datasources_with_tag:
            if source == m2.guiding_table_name: continue
            if source not in table_metadata:
                tm = TableMetadata(table_name=source)
                tm.tags_list_enriched_dataset, tm.tags_list_enriched_names, tm.attributes_list, tm.schema, _, _ = bmmn.load_per_source_metadata(
                    bmmn.p, bmmn.m, {}, source, pds, bmm)

                if source not in m2.attrs_contexts: continue
                tm.attribute_contexts = m2.attrs_contexts[source]
                table_metadata[source] = tm

                tm.exposed_topics = tm.tags_list_enriched_names
                # TODO update tags_list_enriched_dataset when new attrs are added to exposed_topics. Remove attrs from exposed_topics at end of iter

                tm.dataset_stats = bmm.get_table_stats(bmmn.p.dataset_stats, source)

            table_scores[source] = [[0, None]]

            # dataset = pd.read_csv(bmmn.p.datasets_path + source + '.csv', index_col=0, header=0)
            # dataset = bmm.df_rename_cols(dataset)

            compose_topic_ctx = {}
            gtm_contexts = gtm.tags_list_enriched_dataset

            for topic in gtm.exposed_topics:

                # the context could be from the gt or from another dataset
                if topic in gtm_contexts:
                    compose_topic_ctx[topic] = gtm_contexts[topic].copy()

                else:
                    datasets_with_topic = find_source_of_topic(added_topics, topic)
                    topic_ctx = {}
                    for ds in datasets_with_topic:
                        update = table_metadata[ds].tags_list_enriched_dataset[topic].copy()
                        topic_ctx.update(update)
                    compose_topic_ctx[topic] = topic_ctx

            dataset_contexts = table_metadata[source].tags_list_enriched_dataset
            ds_topic_ctx = {}
            for topic in table_metadata[source].exposed_topics:

                # the context can only be from the dataset
                if topic in dataset_contexts:
                    ds_topic_ctx[topic] = dataset_contexts[topic].copy()

            # print('|||||', source, len(dataset_contexts), len(table_metadata[source].exposed_topics))

            sim_matrix2, sim_matrix3, _ = bmmn.build_local_context_similarity_matrix({source: compose_topic_ctx},ds_topic_ctx, source, wordnet, {})

            sim_matrix1 = bmmn.build_local_similarity_matrix(compose_topic_ctx, table_metadata[source].exposed_topics, bmmn.r)

            # print(table_metadata[source].exposed_topics, gtm.exposed_topics)
            # print(gtm.tags_list_enriched_dataset.keys(), table_metadata[source].exposed_topics)

            # print(sim_matrix1.shape, sim_matrix2.shape, sim_matrix3.shape)

            table_scores, list_of_scores = process_scores(sim_matrix1, sim_matrix2, sim_matrix3, table_metadata[source].exposed_topics, source, table_scores, gtm.exposed_topics)

        print('===== table search =====')
        pprint.pprint(table_scores)
        print('==========')


        # add new similar topics, transfer guiding table attrs to new topic group, and transfer other table attrs to existing guiding table topic group. use topic-topic wordnet, ngram name
        comparing_pairs = {}
        this_iteration_change = 0
        for dataset in table_scores:
            scores = sorted(table_scores[dataset], key=lambda x: x[0])
            scores.reverse()

            # print(':::::', dataset, scores)

            print(len(m2.kbs))
            if m2.guiding_table_name not in m2.kbs:
                m2.kbs[m2.guiding_table_name] = m2.kbs['parks'] # TODO <==

            gtm_kb = m2.kbs[m2.guiding_table_name]
            dataset_kb = m2.kbs[dataset]

            for item in scores:
                if item[0] < bmmn.r.topic_to_attr_threshold:
                    break

                this_iteration_change += 1

                score = item[0]
                info = item[1]
                if info[1] == info[2]:  # dataset and guiding have the same topic
                    #  NOTE: still need to add it
                    if info[1] not in gtm_kb:
                        gtm_kb[info[1]] = {}


                    for attr in gtm_kb[info[1]]:
                        if 'source_dataset' not in gtm_kb[info[1]][attr]:
                            gtm_kb[info[1]][attr]['source_dataset'] = [m2.guiding_table_name]
                    pass

                    if dataset not in added_topics:
                        added_topics[dataset] = []  # TODO work around, check this is correct

                # comparing_pairs, gtm, gtm_kb, added_topics = \
                topic_topic_update(dataset, info, score, comparing_pairs, gtm, gtm_kb, dataset_kb, added_topics)

        if this_iteration_change == 0: break

        print('----- end of transfer -----')
        pprint.pprint(gtm.exposed_topics_groups)
        print('----------')

        for key in gtm.exposed_topics_groups:
            for tupl in gtm.exposed_topics_groups[key]:
                topic = tupl[1]
                if topic not in gtm.exposed_topics:
                    gtm.exposed_topics.append(topic)
        # pprint.pprint(gtm.exposed_topics)



        # ===begin coverage===
        table_scores_attr_topic = gt_nonmapped_attr_to_new_topic(table_scores, table_metadata, gtm, wordnet)
        # print('///// topics attributes /////')
        # pprint.pprint(table_scores_attr_topic)
        # print('//////////')

        for dataset in table_scores_attr_topic:
            if dataset == m2.guiding_table_name: continue

            scores = sorted(table_scores_attr_topic[dataset], key=lambda x: x[0])
            scores.reverse()

            # print(':::::', dataset, scores)

            gtm_kb = m2.kbs[m2.guiding_table_name]
            dataset_kb = m2.kbs[dataset]
            # print(dataset_kb.keys())

            for item in scores:
                if item[0] < bmmn.r.topic_to_attr_threshold:
                    break

                score = item[0]
                info = item[1]

                if info[1] not in gtm_kb:
                    gtm_kb[info[1]] = {}

                attr = info[2]
                # print('/////', dataset, info, score)

                # the kb concept did not exist before
                gtm_kb[info[1]][attr] = {}

                dataset_schema = gtm.schema
                # print(len(dataset_schema))
                index = -1
                search = info[2].replace(' ', '_')
                datatype = None
                examples = None
                for sch_attr in dataset_schema:
                    if sch_attr['name'] == search or sch_attr['alias'] == info[2]:

                        if 'coded_values' in sch_attr:
                            examples = sch_attr['coded_values']

                        datatype = sch_attr['data_type']

                kb_match_entry = {'concept': info[1],
                                  'datasource': dataset,       # where the concept comes from
                                  'attribute': info[2],
                                  'match_score': score,
                                  'example_values': examples,
                                  'data_type': datatype,
                                  'score_name': bmmn.m.score_names[info[0]]}

                bmmn.update_kb_json(gtm_kb, kb_match_entry)

                # kb_match_entry['example_values'] = kb_match_entry['example_values'][
                #                                    :min(len(kb_match_entry['example_values']), 5)]
                # pprint.pprint(kb_match_entry)

                if dataset not in added_topics: added_topics[dataset] = []  # TODO this is work around for dataset not found error
                if info[1] not in added_topics[dataset]:
                    added_topics[dataset].append(info[1])

        print('----- end of coverage -----')
        pprint.pprint(added_topics)
        print('----------')


        # ===begin importance===
        table_scores_topic_attr = {}
        topic_from = {}
        for dataset in table_scores:
            table_scores_topic_attr, topic_from = gt_topic_to_dataset_nonmapped_attr(dataset, gtm, added_topics, table_scores_topic_attr, topic_from, table_metadata, wordnet)

        for topic in topic_from:
            topic_from[topic] = list(set(topic_from[topic]))

        # print(len(table_scores_topic_attr))

        for dataset in table_scores_topic_attr:
            if dataset == m2.guiding_table_name: continue

            scores = sorted(table_scores_topic_attr[dataset], key=lambda x: x[0])
            scores.reverse()

            gtm_kb = m2.kbs[m2.guiding_table_name]
            dataset_kb = m2.kbs[dataset]
            # print(dataset_kb.keys())

            for item in scores:
                if item[0] < bmmn.r.topic_to_attr_threshold:
                    break

                # print(item)

                score = item[0]
                info = item[1]

                attr = info[2]

                dataset_schema = table_metadata[dataset].schema
                # print(len(dataset_schema))
                index = -1
                search = info[2].replace(' ', '_')
                datatype = None
                examples = None
                for sch_attr in dataset_schema:
                    if sch_attr['name'] == search or sch_attr['alias'] == info[2]:

                        if 'coded_values' in sch_attr:
                            examples = sch_attr['coded_values']

                        datatype = sch_attr['data_type']


                kb_match_entry = {'concept': info[1],
                                  'datasource': topic_from[info[1]],
                                  'attribute': info[2],
                                  'match_score': score,
                                  'example_values': examples,
                                  'data_type': datatype,
                                  'score_name': bmmn.m.score_names[info[0]]}

                # print('before', len(gtm_kb[info[1]]))

                bmmn.update_kb_json(gtm_kb, kb_match_entry)

                # print('after', len(gtm_kb[info[1]]))
                # if info[1] != 'trees':
                #     print(dataset)
                    # kb_match_entry['example_values'] = kb_match_entry['example_values'][
                    #                                    :min(len(kb_match_entry['example_values']), 5)]
                    # pprint.pprint(kb_match_entry)
                    # pprint.pprint(gtm_kb[info[1]])

                if info[1] not in gtm.exposed_topics_groups:
                    gtm.exposed_topics_groups[info[1]] = []
                    # print(info[1], gtm.exposed_topics_groups[info[1]])
                gtm.exposed_topics_groups[info[1]].append([dataset, None, score])
                # print(gtm.exposed_topics_groups[info[1]])

        print('---- end of importance -----')
        pprint.pprint(gtm.exposed_topics_groups)
        print('----------')

        '''
        # compute attr-attr similarity matrix (just append one more column). use attr-attr comparison: tf-idf pair of document of values, TODO ngram per val, attr name wordnet
        sim_dict = {}

        # ===fine grained matching===
        for pair in comparing_pairs:
            gtm_top_name, dataset,dataset_top_name  = pair
            gtm_top, dataset_top = comparing_pairs[pair]

            for gtm_attr in gtm_top:
                for dataset_attr in dataset_top:

                    text1 = preprocesss_attr_values(gtm_top[gtm_attr]['example_values'])
                    text2 = preprocesss_attr_values(dataset_top[dataset_attr]['example_values'])

                    if len(text1) == 0 or len(text2) == 0: continue

                    score = sch.matcher_instance_document(text1, text2)

                    if score > 0:
                        pass
                        # print('>>>>>', gtm_attr, len(text1) ,' | ', dataset, dataset_attr, len(text2),  ' || ',  score)
                        # print('<<<<<')
        # TODO clustering. split out attrs from topics (break ties usig avg attr-topic score), merge attrs into (newly added only) groups if possible
        '''


        # ===remove added topics from the other table===
        print('=====remove added topics=====')
        pprint.pprint(added_topics)
        for dataset in added_topics:
            rm_topics = added_topics[dataset]
            exposed_topics = table_metadata[dataset].exposed_topics
            for top in rm_topics:
                # print('rm', dataset, top)
                if top not in exposed_topics: continue  # need to fix this
                exposed_topics.remove(top)

            pprint.pprint(table_metadata[dataset].exposed_topics)

        for key in gtm.exposed_topics_groups:
            items_list = gtm.exposed_topics_groups[key]
            for tupl in items_list:
                dataset, topic, score = tupl[0], tupl[1], tupl[2]

                if key not in gtm.exposed_topics:
                    gtm.exposed_topics.append(key)

                if topic not in gtm.exposed_topics and topic != None:
                    gtm.exposed_topics.append(topic)

        if None in gtm.exposed_topics:
            gtm.exposed_topics.remove(None)


        # TODO repeat comparing table topics
        num_iters += 1
        if num_iters == bmmn.r.num_iters:
            break_out = True




    # transfer topics to tables related to guiding table using topic-attr mappings
    dataset_topics = ce.kb_to_topics_per_dataset(m2.kbs[m2.guiding_table_name], m2.guiding_table_name)

    # TODO remember to remove topics from here if found some topic should be split
    add_more_topics(gtm.exposed_topics_groups, dataset_topics)

    print('=====transfer topics=====')
    pprint.pprint(dataset_topics)

    # eval accuracy
    ground = {'parks' : ['green', 'trees', 'parks'], 'park specimen trees' : ['green', 'trees']}

    accu = ce.compute_precision_and_recall(reverse_dict(ground), [reverse_dict(dataset_topics)])

    print('=====accuracy=====')
    print(accu)

    with open(bmmn.p.kb_file_p, 'w') as fp:
        json.dump(m2.kbs, fp, sort_keys=True, indent=2)

    with open(bmmn.p.dataset_topics_p, 'w') as fp:
        json.dump(dataset_topics, fp, sort_keys=True, indent=2)




import pathlib
if __name__ == "__main__":
    import sys

    old_stdout = sys.stdout
    log_file = open("message.log", "w")
    sys.stdout = log_file
    print(    "this will be written to message.log")


    table_topics_p = 'outputs/table_topics.json'
    table_setup_p = 'outputs/table_setup.json'

    f = open(table_topics_p)
    table_topics = json.load(f)

    f = open(table_setup_p)
    table_setup = json.load(f)

    # datasources_with_tag = ['aquatic hubs', 'drainage 200 year flood plain', 'drainage water bodies',
    #                         'park specimen trees', 'parks', 'park screen trees']
    # guiding_table_name = 'parks'


    # datasources_with_tag = table_setup['tables']  # TODO change to this later
    datasources_with_tag = [table_setup['guiding_tables'][item][0] for item in table_setup['guiding_tables']]
    datasources_with_tag = [datasources_with_tag[0]]

    for k,table in enumerate(table_setup['guiding_tables']):
        if k != 0: continue # TODO for now just first table

        dataset_name = table_setup['guiding_tables'][table][0]
        dataset_name = 'parks' # TODO <=
        print(dataset_name)

        # TODO change scope of datasets, per sample size per guiding table
        print('[[[', 'global mappings', dataset_name, ']]]')    # GUIDING TABLE

        for plan in table_topics[dataset_name]['samples']:
            if int(plan) > 10:  # TODO for now just try sample size 10
                continue
            mixes = table_topics[dataset_name]['samples'][plan]
            for mix in mixes:
                datasources_with_tag += mixes[mix][0] + mixes[mix][1]
    datasources_with_tag = list(set(datasources_with_tag))

    datasources_with_tag = ['parks', 'heritage sites', 'water utility facilities', 'sanitary lift stations', 'drainage dyke infrastructure', 'park outdoor recreation facilities', 'park sports fields', 'water assemblies', 'road row requirements downtown']  # TODO <=

    datasources_with_tag = ['parks', 'park specimen trees', 'park screen trees', 'park outdoor recreation facilities', "park structures"]

    # TODO kb_local_mapping does not have any data after local mapping!

    print(len(datasources_with_tag))
    print(datasources_with_tag)

    done = False # TODO <=
    if not done:
        print('enrich topics')
        pt.enrich_topics_full_run(datasources_with_tag)
        print('create attributes contexts')
        bmmn.load_metadata(bmmn.p, bmmn.m)
        bmmn.m.datasources_with_tag = datasources_with_tag
        bmmn.create_attributes_contexts(datasources_with_tag, bmmn.m, bmmn.p, bmmn.r)
        # then move the json files to the appropriate dirs
        print('contexts to json')

        t0 = time.time()

        settj.one_full_run()

        t1 = time.time()
        total = t1 - t0
        print('>>>>>>>>>>>>>>>>prematching time %s sec<<<<<<<<<<<<<<<' % (total))

    t0 = time.time()

    print('local mappings')
    bmmn.load_metadata(bmmn.p, bmmn.m)
    bmmn.m.datasources_with_tag = datasources_with_tag
    # bmmn.local_mappings(bmmn.p, bmmn.m, bmmn.r)
    bmmn.local_mappings_full(bmmn.p, bmmn.m, bmmn.r)

    t1 = time.time()
    total = t1 - t0
    print('>>>>>>>>>>>>>>>>local time %s sec<<<<<<<<<<<<<<<' % (total))

    dataset_metadata_f = open('./inputs/datasource_and_tags.json', 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    #
    #exit(0) # TODO <=




    for k,table in enumerate(table_setup['guiding_tables']):
        if k % 2 != 0: continue # TODO for now just try 5 tables

        dataset_name = table_setup['guiding_tables'][table][0]
        temp1 = dataset_name
        dataset_name = 'parks' # TODO <=
        print(dataset_name, '!!!')

        # TODO change scope of datasets, per sample size per guiding table
        print('[[[', 'global mappings', dataset_name, ']]]')    # GUIDING TABLE

        for plan in table_topics[dataset_name]['samples']:
            if int(plan) > 10:  # TODO for now just try sample size 10
                continue
            mixes = table_topics[dataset_name]['samples'][plan]
            temp2 = mixes
            mixes = {'1+4': [['parks'], ['heritage sites', 'water utility facilities', 'sanitary lift stations', 'drainage dyke infrastructure']], '3+2': [['parks', 'park outdoor recreation facilities', 'park sports fields'], ['water assemblies', 'road row requirements downtown']]} # TODO <=

            mixes = {'5+0': [['parks', 'park specimen trees', 'park screen trees', 'park outdoor recreation facilities', "park structures"],
                             []]} # TODO <=
            for mix in mixes:
                datasources_with_tag = mixes[mix][0] + mixes[mix][1]
                print('one_full_run:', dataset_name, plan, mix, datasources_with_tag)

                json_kb_save_name = "./outputs/kb_file_v2_" + '{0}' + ".json"
                json_kb_save_name = json_kb_save_name.replace('{0}', dataset_name+'_'+plan+'_'+mix)

                bmmn.p.kb_file_p = json_kb_save_name
                bmmn.p.dataset_topics_p = "./outputs/dataset_topics_v2_" + dataset_name+'_'+plan+'_'+mix + ".json"


                my_file = pathlib.Path(json_kb_save_name)
                if my_file.exists():
                    continue

                t0 = time.time()

                one_full_run(dataset_name, datasources_with_tag)

                t1 = time.time()
                total = t1 - t0
                print('>>>>>>>>>>>>>>>>%s matching time %s sec<<<<<<<<<<<<<<<' % (dataset_name, total))

    sys.stdout = old_stdout
    print("this will not be written to message.log")
    log_file.close()
