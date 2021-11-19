DEBUG_MODE = False

# import os
# import sys
import networkx
import itertools
import pandas as pd
# from numpy import array

# scriptpath = "./parse_dataset.py"
# sys.path.append(os.path.abspath(scriptpath))
import parse_surrey_dataset as surrey

import math
import collections

# import sklearn.cluster as cluster
import numpy as np
# import tkinter
# import matplotlib.pyplot

import pprint
pp = pprint.PrettyPrinter(indent=2)


class GraphicalModel(object):
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.texts_vectors = []
        self.edge_ids = []
        self.similarities = []


    # def cluster(self):
    #     similarities_array = np.asarray(self.similarities, dtype=np.float32)

    #     k_means = cluster.KMeans(n_clusters=2)
    #     k_means.fit(similarities_array.reshape(-1, 1))
    #     # print(k_means.labels_)

    #     for j in range(len(k_means.labels_)):
    #         label = k_means.labels_[j]
    #         similarity = self.similarities[j]
    #         edge = self.edges[self.edge_ids[j]]
    #         print('(', edge.node0, edge.node1, ')', similarity, ':', label)


def build_graph(data_model):
    graphical_model = GraphicalModel()

    nodes = {}
    edges = {}

    texts_vectors = []

    for key in data_model.datasets:
        data_instance = data_model.datasets[key]
        print('Data Instance: ' + key)

        for resource in data_instance.resources:
            print(resource['format'])

            data = resource['data']
            first_row = data[0]
            for attribute_name in first_row:
                # print('\t' + attribute_name)

                text = attribute_name
                textvec = word2vec(attribute_name)

                node = AttributeNode()
                node.resource_name = resource
                node.attribute_name_vec = textvec
                node.attribute_name = text

                texts_vectors.append(node.id)
                nodes[node.id] = node

    S = complete_graph_from_list(texts_vectors)

    edge_ids = []
    similarities = []
    for key in S.edges.keys():
        # print('key', key)
        # print(key[0], 'and', key[1])
        # print('value', S.edges.get(key))
        edge = AttributeEdge()
        edge.node0 = key[0]
        edge.node1 = key[1]
        # print(nodes[edge.node0].attribute_name)
        # print(nodes[edge.node1].attribute_name)
        edge.similarity = cosdis(nodes[edge.node0].attribute_name_vec, nodes[edge.node1].attribute_name_vec)
        edges[edge.id] = edge
        # print('(', edge.node0, edge.node1, ')',  edges[key].similarity)
        edge_ids.append(edge.id)
        similarities.append(edge.similarity)

    graphical_model.nodes = nodes
    graphical_model.edges = edges
    graphical_model.texts_vectors = texts_vectors
    graphical_model.edge_ids = edge_ids
    graphical_model.similarities = similarities

    return graphical_model

def complete_graph_from_list(L, create_using=None):
    G = networkx.empty_graph(len(L),create_using)
    if len(L)>1:
        if G.is_directed():
            edges = itertools.permutations(L,2)
        else:
            edges = itertools.combinations(L,2)
        G.add_edges_from(edges)
    return G


def word2vec(word):
    # count the characters in word
    cw = collections.Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = math.sqrt(sum(c*c for c in cw.values()))

    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


class AttributeNode(object):
    newid = 0

    def __init__(self):
        self.id = AttributeNode.newid
        AttributeNode.newid += 1
        self.resource_name = ''
        self.attribute_name = ''
        self.attribute_name_vec = ''

class AttributeEdge(object):
    newid = 0

    def __init__(self):
        self.id = AttributeEdge.newid
        AttributeEdge.newid += 1

        self.node0 = -1
        self.node1 = -1
        self.similarity = 0


from difflib import SequenceMatcher
from similarity.ngram import NGram
twogram = NGram(2)
fourgram = NGram(4)

from similarity.metric_lcs import MetricLCS
metric_lcs = MetricLCS()
def build_local_similarity_matrix(source_schema, target_schema):
    matrix= np.zeros((len(source_schema), len(target_schema)))

    for i in range(len(source_schema)):
            for j in range(len(target_schema)):
                # TODO call matcher
                sim_score = 1 - twogram.distance(source_schema[i],target_schema[j])
                # matrix[i,j] = np.int(100*SequenceMatcher(None,source_schema[i],target_schema[j]).ratio())
                matrix[i, j] = sim_score

                DEBUG_MODE = False # TODO <=
                if matrix[i, j] >= 0.5 and DEBUG_MODE:
                    print('matrix[i, j]', source_schema[i], target_schema[j], matrix[i, j])

                # if target_schema[j] == 'tree_species':
                #     print(source_schema[i], target_schema[j], matrix[i, j])

    return matrix


def match_table_by_values_beta(source_instance, target_instance, source_schema, target_schema):
    src_values = []
    tar_values = []

    # src_key = source_instance.keys()[0]
    src_val_len = len(source_instance[source_schema[0]])
    # tar_key = target_instance.keys()[0]
    tar_val_len = len(target_instance[target_schema[0]])

    source_keys = source_schema
        # source_instance.keys()
    # source_keys.sort()
    target_keys = target_schema
        # target_instance.keys()
    # target_keys.sort()

    start_ind = {}
    attr_ind = 0
    val_ind = 0
    for key in source_keys:
        # print('src matrix dimension ' + str(src_val_len) + ' ' + str(len(source_instance[key])))
        # assert src_val_len == len(source_instance[key])
        src_values.extend(source_instance[key])
        for val in source_instance[key]:
            start_ind[val_ind] = attr_ind
            val_ind += 1
        attr_ind =+ 1

    for key in target_keys:
        # print('tar matrix dimension ' + str(src_val_len) + ' ' + str(len(target_instance[key])))
        assert tar_val_len == len(target_instance[key])
        tar_values.extend(target_instance[key])

    sim_matrix = np.zeros((len(source_schema), len(target_schema)))
    for i in range(len(src_values)):
        src_value = src_values[i]
        src_ind = start_ind[i]
        src_attr = source_keys[src_ind]
        for j in range(len(tar_values)):
            tar_value = tar_values[j]
            tar_ind = j // tar_val_len
            tar_attr = target_keys[tar_ind]
            # sim_score = np.int( SequenceMatcher(None, str(src_value), str(tar_value)).ratio())
            sim_score = 1 - twogram.distance(str(src_value), str(tar_value))

            if str(src_value) == 'None' or str(tar_value) == 'None':
                sim_score = 0
            sim_matrix[src_ind, tar_ind] += sim_score

            if sim_score >= 0.5 and DEBUG_MODE:
                print('sim_score >= 0.5', src_attr, tar_attr, src_value, tar_value, sim_score)

    return sim_matrix

def find_potential_matches(sim_matrix, threshold, src_attrs, tar_attrs, src_name, tar_name):

    matches = []
    for i in range(len(sim_matrix)):
        row = sim_matrix[i]
        max_score = max(row)
        max_indices = [ind for ind, val in enumerate(row) if val == max_score]
        max_rm = [val for ind, val in enumerate(row) if ind not in max_indices]
        matched = all(val < max_score/threshold for val in max_rm)
        matched = matched or max_score >= 0.5 and not (max_score == 0)
        if matched:
            for ind in max_indices:
                matches.append({'i': i, 'j': ind, 'val[i]': src_attrs[i], 'val[j]': tar_attrs[ind], 'src_name': src_name, 'tar_name': tar_name, 'sim_score': sim_matrix[i][ind]})

    return matches

def populate_concept_with_samples(matches_list, dataset, kb_concepts):
    for matches in matches_list:
        for match in matches:
            kb_concept_name = match['val[i]']
            attr_name = match['val[j]']
            datasource_name = match['src_name'][0]

            metadata = dataset[datasource_name][0]
            values = dataset[datasource_name][1]

            # print(kb_concept_name, attr_name, datasource_name)
            concept_source = None
            kb_concept_sources = kb_concepts[kb_concept_name]
            for source in kb_concept_sources:
                if source['source_name'] == datasource_name:
                    concept_source = source

            coded_values = False
            for attr in metadata:
                if attr['name'] == attr_name and attr['domain'] != None and attr['domain'] == 'coded_values':
                    concept_source['coded_values'] = attr['coded_values']
                    coded_values = True

            # use summarization and sampling
            if coded_values == False:
                # DO NOT DO THIS, if no coded values, then do not populate
                # concept_source['coded_values'] = values[attr_name]
                concept_source['coded_values'] = []
            concept_source['sim_score'] = match['sim_score']

import nltk
import platform
pltfm = platform.system()
if pltfm == 'Linux':
    nltk.data.path.append('/media/haleyyew/e2660490-a736-4bb9-b3dd-5c0f3871a2f2/thesis_code/python_venv/nltk_data/')
else:
    nltk.data.path.append('/Users/haoran/Documents/nltk_data/')
from nltk.corpus import wordnet
def find_synonyms_antonyms(word):
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                for ant in l.antonyms():
                    antonyms.append(ant.name())
    return (synonyms, antonyms)

'''
def toy_example():
    import toy.data as td
    kb_concepts = parse_dataset.collect_concepts_beta()
    # tags = kb_concepts.keys()

    # local matches
    PARKS_metadata_tags = list(td.PARKS_metadata['tags'])
    IMPORTANTTREES_metadata_tags = list(td.IMPORTANTTREES_metadata['tags'])
    PARKSPECIMENTREES_metadata_tags = list(td.PARKSPECIMENTREES_metadata['tags'])
    PARKS_metadata_category = [td.PARKS_metadata['category']]
    IMPORTANTTREES_metadata_category = [td.IMPORTANTTREES_metadata['category']]
    PARKSPECIMENTREES_metadata_category = [td.PARKSPECIMENTREES_metadata['category']]
    PARKS_schema_attr = list(td.PARKS_schema.keys())
    IMPORTANTTREES_schema_attr = list(td.IMPORTANTTREES_schema.keys())
    PARKSPECIMENTREES_schema_attr = list(td.PARKSPECIMENTREES_schema.keys())

    schema_sources = [PARKS_metadata_tags,
        IMPORTANTTREES_metadata_tags,
        PARKSPECIMENTREES_metadata_tags,
        PARKS_metadata_category,
        IMPORTANTTREES_metadata_category,
        PARKSPECIMENTREES_metadata_category,
        PARKS_schema_attr,
        IMPORTANTTREES_schema_attr,
        PARKSPECIMENTREES_schema_attr]

    schema_sources_clean = []
    for source in schema_sources:
        schema_source = [val.lower() for val in source]
        schema_source.sort()
        schema_sources_clean.append(schema_source)
        # print(schema_source)

    schema_source_names = [('PARKS', 'tag'), ('IMPORTANTTREES', 'tag'), ('PARKSPECIMENTREES', 'tag'), ('PARKS', 'category'), ('IMPORTANTTREES', 'category'), ('PARKSPECIMENTREES', 'category'), ('PARKS', 'attributes'), ('IMPORTANTTREES', 'attributes'), ('PARKSPECIMENTREES', 'attributes')]

    # initialization
    # for a dataset, between the pair of (its metadata tags, its schema)
    matching_tasks = [(0, 6), (1, 7), (2, 8), (3, 6), (4, 7), (5, 8)]

    similarity_matrices = []
    for task in matching_tasks:
        similarity_matrices.append(
            (build_local_similarity_matrix(schema_sources_clean[task[0]], schema_sources_clean[task[1]]),
             schema_sources_clean[task[0]],
             schema_sources_clean[task[1]],
             schema_source_names[task[0]],
             schema_source_names[task[1]]))



    matches_list = []
    for matrix in similarity_matrices:
        matches = find_potential_matches(matrix[0], 3, matrix[1], matrix[2], matrix[3], matrix[4])
        matches_list.append(matches)

    print('---matches_list---')
    pp.pprint(matches_list)
    # TODO get all matches, include probabilities

    metadata_files = ['211 Important Trees.json', '239 Park Specimen Trees.json', '244 Parks.json']
    metadata_files = ['./metadata/'+file for file in metadata_files]
    metadata_it = parse_dataset.parse_metadata(metadata_files[0])
    metadata_pst = parse_dataset.parse_metadata(metadata_files[1])
    metadata_parks = parse_dataset.parse_metadata(metadata_files[2])

    # populate each concept with some coded values, based on matches; if no coded values, get a sample of values from instance

    values_it = td.IMPORTANTTREES_data
    values_pst = td.PARKSPECIMENTREES_data
    values_parks = td.PARKS_data

    dataset = {'PARKS': (metadata_parks, values_parks), 'IMPORTANTTREES': (metadata_it, values_it), 'PARKSPECIMENTREES': (metadata_pst, values_pst)}

    populate_concept_with_samples(matches_list, dataset, kb_concepts)

    # print('[kb_concepts]')
    # pp.pprint(kb_concepts)
    # exit(0)

    # print(similarity_matrices[0])
    # print(metadata_parks)
    # # print(schema_sources_clean[0])
    # # print(schema_sources_clean[6])
    # print(matches_list[0])
    # print(values_parks)

    # for concept in kb_concepts:
    #     concept_sources = kb_concepts[concept]
    #     for source in concept_sources:
    #         if 'coded_values' in source:
    #             print(concept, source['source_name'], source['coded_values'], source['sim_score'])




    # match each set of {source attribute values} with each set of {global knowledge base concept values}
    kb_concept_values = {}
    for concept in kb_concepts:
        concept_sources = kb_concepts[concept]
        for source in concept_sources:
            if 'coded_values' in source:
                kb_concept_values[(concept, source['source_name'])] = source['coded_values']

    print('---kb_concept_values---')
    pp.pprint(kb_concept_values)

    # find synonyms for concept
    synonyms_antonyms = {}
    for concept in kb_concept_values.keys():
        if len(kb_concept_values[concept]) < 1:
            continue
        synonyms_antonyms[concept] = find_synonyms_antonyms(concept[0])
        # print('synonyms', concept, ': ', synonyms_antonyms[concept])


    print('---match_table_by_values_beta---')
    for data_source_name in dataset:
        data_source = dataset[data_source_name]
        src = kb_concept_values
        tar = data_source[1]

        src_schema = [key for key in src.keys()]
        tar_schema = [key for key in tar.keys()]
        matrix = match_table_by_values_beta(src, tar, src_schema, tar_schema)
        pp.pprint({'name':data_source_name, 'src':src_schema, 'tar':tar_schema, 'matrix':matrix})
'''

def serialize_for_rdf(str):
    return str.replace(' ', '-')



def build_kb_json(list_of_concepts, kb):

    for concept in list_of_concepts:
        concept_name = concept[0]
        datasources = concept[1]
        if concept_name not in kb:
            kb[concept_name] = {}
            kb[concept_name]['datasources'] = datasources
            kb[concept_name]['matches'] = {}
        else:
            kb_concept = kb[concept_name]
            kb_concept['datasources'].extend(datasources)
        # TODO remove duplicates
    return kb

def update_kb_json(kb, match_entry):
    concept = match_entry['concept']
    datasource = match_entry['datasource']
    attribute = match_entry['attribute']
    match_score = match_entry['match_score']
    example_values = match_entry['example_values']
    data_type = match_entry['data_type']

    kb_concept = kb[concept]
    kb_concept_matches = kb_concept['matches']
    kb_concept_matches[datasource] = {'attribute': attribute, 'match_score' : match_score, 'example_values' : example_values, 'data_type' : data_type}
    return

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def match_table_by_values(df_src, df_tar, match_threshold, comparison_count_o, stats,
                          sample_ratio, sample_min_count, sample_max_count):
    comparison_count = comparison_count_o[0]
    schema_tar = list(df_tar.columns.values)
    schema_src = list(df_src.columns.values)

    print('matching:', schema_src, 'with', schema_tar)

    src_values = []
    tar_values = []
    src_val_len = 0
    tar_val_len = 0
    num_rows_src = 0
    num_rows_taken_src = 0
    for attr in schema_src:
        attr_vals = list(df_src[attr])
        num_rows_src = len(attr_vals)
        num_rows_taken_src = int(sample_ratio * num_rows_src)
        if num_rows_taken_src < sample_min_count:
            num_rows_taken_src = sample_min_count
        if num_rows_taken_src > sample_max_count:
            num_rows_taken_src = sample_max_count
        attr_vals = attr_vals[:num_rows_taken_src]      # TODO: need to sample, not just take the top rows
        src_values.extend(attr_vals)
        src_val_len = len(attr_vals)

    num_rows_tar = 0
    num_rows_taken_tar = 0
    for attr in schema_tar:
        # randomly sample some values, and return a probability of instance match ratio
        attr_vals = list(df_tar[attr])
        num_rows_tar = len(attr_vals)
        num_rows_taken_tar = int(sample_ratio * num_rows_tar)
        if num_rows_taken_tar < sample_min_count:
            num_rows_taken_tar = sample_min_count
        if num_rows_taken_src > sample_max_count:
            num_rows_taken_src = sample_max_count
        attr_vals = attr_vals[:num_rows_taken_tar]
        tar_values.extend(attr_vals)
        tar_val_len = len(attr_vals)

    # might not be able to match anything
    if len(schema_tar) == 0 or len(schema_tar) == 0:
        return pd.DataFrame(columns=schema_tar, index=schema_src), -1

    sim_matrix = np.zeros((len(schema_src), len(schema_tar)))

    confidence = 0
    if num_rows_tar * num_rows_src > 0:
        confidence = (num_rows_taken_src * num_rows_taken_tar)/(num_rows_tar * num_rows_src)
    else:
        df_sim_matrix = pd.DataFrame(data=sim_matrix, columns=schema_tar, index=schema_src)
        return df_sim_matrix, confidence

    for i in range(len(src_values)):
        src_value = src_values[i]
        src_ind = i // src_val_len      # modulo to find out what is its attribute header
        src_attr = schema_src[src_ind]

        for j in range(len(tar_values)):
            tar_value = tar_values[j]
            tar_ind = j // tar_val_len
            tar_attr = schema_tar[tar_ind]

            sim_score = 0
            if str(src_value) == 'None' or str(tar_value) == 'None':
                sim_score = 0

            elif RepresentsInt(src_value) != RepresentsInt(tar_value):
                sim_score = 0

            else:
                sim_score = 1 - twogram.distance(str(src_value), str(tar_value))
                comparison_count += 1
                if comparison_count % 50000 == 0:
                    print('comparison_count=', comparison_count)


            if sim_score > match_threshold:
                reps = stats[tar_attr][tar_value]
                sim_matrix[src_ind, tar_ind] += sim_score * reps
                if DEBUG_MODE:
                    print('|sim_score %0.2f > %0.2f: %s(%s) <=> %s(%s) * %d|' % (sim_score, match_threshold, src_attr, src_value, tar_attr, tar_value, reps))

    df_sim_matrix = pd.DataFrame(data=sim_matrix, columns=schema_tar, index=schema_src)
    comparison_count_o[0] = comparison_count

    return df_sim_matrix, confidence

def groupby_unique(attr, df):
    stat = {}
    print(attr, df.head(2))
    groups = df.groupby([attr])[attr]
    for key, item in groups:
        key_str = key
        if not isinstance(key, str): key_str = str(key)
        stat[key_str] = len(groups.get_group(key).values)

    return stat, groups, list(groups.groups.keys())

def find_attrs_to_delete(tar_schema, df_columns):
    cols_to_delete = ['latitude', 'longitude']
    attrs_schema = []
    for attr in tar_schema:
        name = attr

        if type(attr) is dict:
            name = attr['name']
        if type(attr) is str:
            name = attr

        attrs_schema.append(name)

    for col in df_columns:
        if col not in attrs_schema:
            cols_to_delete.append(col)

    for col in attrs_schema:
        if col not in df_columns:
            cols_to_delete.append(col)

    return cols_to_delete

def compare_datatypes(src_datatype, tar_schema, df_columns):
    # all_datatypes = parse_dataset.collect_all_datatypes('./schema_complete_list.json')
    datatype_groups = {'esriFieldTypeBlob': 1,
                         'esriFieldTypeDate': 2,
                         'esriFieldTypeDouble': 3,
                         'esriFieldTypeGeometry': 4,
                         'esriFieldTypeGlobalID': 5,
                         'esriFieldTypeInteger': 3,
                         'esriFieldTypeOID': 5,
                         'esriFieldTypeSingle': 3,
                         'esriFieldTypeSmallInteger': 3,
                         'esriFieldTypeString': 1}

    cols_to_delete = []
    src_group = datatype_groups[src_datatype]
    attrs_schema = []
    for attr in tar_schema:
        name = attr['name']
        data_type = attr['data_type']
        attr_group = datatype_groups[data_type]
        if src_group != attr_group:
            cols_to_delete.append(name)

    cols_to_delete.extend(find_attrs_to_delete(tar_schema, df_columns))

    return list(set(cols_to_delete))

def df_rename_cols(dataset):
    dataset_col_names = [surrey.clean_name(x) for x in list(dataset.columns.values)]
    col_rename_dict = {i: j for i, j in zip(list(dataset.columns.values), dataset_col_names)}
    dataset.rename(columns=col_rename_dict, inplace=True)
    return dataset

def df_delete_cols(df, cols_to_delete):
    # print(cols_to_delete)
    tar_attrs = list(df.columns.values)
    for col in cols_to_delete:
        if col in tar_attrs and col in df.columns:
            # print('drop', col)
            df.drop(col, axis=1, inplace=True)

    # print(df.head())
    return df

def print_metadata_head(source_name, dataset, schema, metadata):
    print('dataset_name:', source_name)
    print('dataset_values.head \n', dataset.head())
    print('dataset_schema[0]', surrey.clean_name(schema[0]['name'], False, False))
    print('dataset_schema[1]', surrey.clean_name(schema[1]['name'], False, False))
    print('dataset_tags[0]', metadata[0]['display_name'])

import pathlib
def gather_statistics(schema_set, datasources_with_tag, stats_output_p, datasets_input_p):

    for source_name in datasources_with_tag:
        all_stats = {}

        file_path = stats_output_p + source_name + '.json'
        my_file = pathlib.Path(file_path)
        if my_file.exists():
            continue

        path = datasets_input_p + source_name + '.csv'
        dataset = pd.read_csv(path, index_col=0, header=0)
        dataset = df_rename_cols(dataset)

        attr_schema = schema_set[source_name]
        df_columns = list(dataset.columns.values)

        # print(attr_schema)
        attr_schema = [{'name': surrey.clean_name(attr['name'], False, False)} for attr in attr_schema]
        df_columns = [surrey.clean_name(attr, False, False) for attr in df_columns]

        cols_to_delete = find_attrs_to_delete(attr_schema, df_columns)
        dataset = df_delete_cols(dataset, cols_to_delete)

        schema = schema_set[source_name]
        attributes_list = [surrey.clean_name(attr['name'], False, False) for attr in schema]
        attributes_list = [item for item in attributes_list if item not in cols_to_delete]

        print(source_name, attributes_list, cols_to_delete)

        for attr in attributes_list:
            stat, groups, uniques = groupby_unique(attr, dataset)

            all_stats[attr] = stat

            # TODO more types of stats needed


        with open(file_path, 'w') as fp:
            json.dump(all_stats, fp, sort_keys=True, indent=2)

    return

def get_table_stats(dataset_stats, source_name):

    stats_f = open(dataset_stats + source_name + '.json', 'r')
    stats = json.load(stats_f)


    stats_f.close()
    return stats


def get_attr_stats(dataset_stats, source_name, attr):

    stats_f = open(dataset_stats + source_name + '.json', 'r')
    stats = json.load(stats_f)

    if attr not in stats:
        print('get_attr_stats not found', attr)
        return None, None

    stat = stats[attr]
    uniques = list(stats[attr].keys())

    stats_f.close()

    return stat, uniques

import json
import numpy as np
import pandas as pd
import time
if __name__ == "__main__":
    '''
    1. kb values for concept - probabilities generation and propagation
    2. use thesaurus wordnet to find related concepts names, cluster
    3. training with imperfect labels
    4. correct knowledge base errors by imperfect training
    5. get new concepts from summarization
    6.  iterative until no affinity between clusters is strong enough
    
    1.  use numpy and pandas and rdf
        use actual complete datasource values
        //fix potential matrix bug
        *** file name as first clustering step? ***
    2.  better distance functions
        ignore the many-to-one mapping constraint when mapping
        *** output of clusters becomes imperfect labels for value-pair matches for imperfect training***
    3.  train classifier
        create mappings as training data
        *** output of imperfect labels predictions becomes actual training data ***
    4.  improve schema matching?
    5.  summarization:
        get unique values for an attribute column
        rank the attribute names by num of non-null values in column

    overcome np-completeness
    use iterative algorithm
    use greedy algorithms
    run in small batches
    need run on server!
        
    dataset: flat csv or json
    metadata concepts
    schema has sample values
    need some training data!
    measuring precision and recall
        need to know real concepts and mappings
        need to find differences in terms of number of concepts and number of wrong mappings
    '''

    # data_model = parse_dataset.parse_models()
    # graphical_model = build_graph(data_model)
    # graphical_model.cluster()

    # build_similarity_matrices(data_model, kb_concepts)
    # -----

    t0 = time.time()

    dataset_metadata_f = open('./datasource_and_tags.json', 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    metadata_f = open('./metadata_tag_list_translated.json', 'r')
    metadata_set = json.load(metadata_f)

    schema_f = open('./schema_complete_list.json', 'r')
    schema_set = json.load(schema_f, strict=False)

    # out of the 749 tags, start with one just as an example
    # TODO do the other 748
    topic = 'trees'
    datasources_with_tag = metadata_set['tags'][topic]['sources']
    # the knowledge base to be updated
    # kb = build_kb([(topic, datasources_with_tag)])
    kb = {}
    kb = build_kb_json([(topic, datasources_with_tag)], kb)
    # used for computing probability
    len_datasources = len(datasources_with_tag)

    # exit(0)

    datasets_path = './thesis_project_dataset_clean/'
    datasources = {}
    for source_name in datasources_with_tag:
        dataset = pd.read_csv(datasets_path + source_name + '.csv', index_col=0, header=0)
        schema = schema_set[source_name]
        metadata = dataset_metadata_set[source_name]['tags']

        dataset = df_rename_cols(dataset)

        datasources[source_name] = (source_name, dataset, schema, metadata)
        print_metadata_head(source_name, dataset, schema, metadata)

        # initialization schema matching
        tags_list = [tag['display_name'] for tag in metadata]
        attributes_list = [surrey.clean_name(attr['name'], False, False) for attr in schema]
        sim_matrix = build_local_similarity_matrix(tags_list, attributes_list)
        sim_frame = pd.DataFrame(data=sim_matrix, columns=attributes_list, index=tags_list)
        print(sim_frame.to_string())

        tree_matches = sim_frame.loc['trees']
        attrs = list(sim_frame.columns.values)
        max_score = 0
        arg_max_score = None
        arg_i = -1
        for attr_i in range(len(attrs)):
            attr = attrs[attr_i]
            score = sim_frame.loc['trees', attr]
            if score > max_score:
                max_score = score
                arg_max_score = attr
                arg_i = attr_i

        arg_max_examples_vals = None
        example_value = None
        if schema[arg_i]['domain'] != None:
            arg_max_examples_vals = schema[arg_i]['coded_values']
            arg_max_examples_vals.sort()
            example_value = arg_max_examples_vals[0]
        else:
            stat, groups, uniques = groupby_unique(attrs[arg_i], dataset)
            uniques.sort()
            schema[arg_i]['coded_values'] = uniques
            arg_max_examples_vals = schema[arg_i]['coded_values']
            print('arg_max_examples_vals', arg_max_examples_vals)
            schema[arg_i]['domain'] = 'coded_values_groupby'

        print('best match to trees:', arg_max_score, max_score, example_value)

        kb_match_entry = {'concept': 'trees',
                          'datasource': source_name,
                          'attribute': arg_max_score,
                          'match_score': max_score,
                          'example_values': arg_max_examples_vals,
                          'data_type': schema[arg_i]['data_type']}
        # update_kb_with_match(kb, kb_match_entry)
        update_kb_json(kb, kb_match_entry)
        print('-----')

    # done initialization
    dataset_metadata_f.close()
    metadata_f.close()
    schema_f.close()

    # Match by value:

    comparison_count = 0
    comparison_count_o = [comparison_count]

    for source_name in datasources_with_tag:
        t2 = time.time()

        dataset = pd.read_csv(datasets_path + source_name + '.csv', index_col=0, header=0)
        dataset = df_rename_cols(dataset)

        schema = schema_set[source_name]
        metadata = dataset_metadata_set[source_name]['tags']

        schema_attr_names = []
        for attr in schema:
            attr['name'] = surrey.clean_name(attr['name'], False, False)
            schema_attr_names.append(attr['name'])
        schema_attr_names.sort()

        match_threshold = 0.6
        sample_ratio = 0.01
        sample_min_count = 20
        sample_max_count = 100
        for concept in kb:
            for datasource in kb[concept]['matches']:
                src_attr = kb[concept]['matches'][datasource]['attribute']
                src_vals = kb[concept]['matches'][datasource]['example_values']
                # do not match with self
                if source_name == datasource:
                    continue
                # do not match if no populated values
                if src_vals == None:
                    continue

                src_data = pd.DataFrame({src_attr: src_vals})
                print("[concept:%s, datasource:%s(%s) <=> dataset:%s]" % (concept, datasource, src_attr, source_name))

                # groupby values for each column and obtain count for each unique value, then multiply counts when comparison succeeds
                tar_schema = list(dataset.columns.values)
                attrs_stat = {}
                max_len = 0
                for attr in tar_schema:
                    stat, groups, uniques = groupby_unique(attr, dataset)
                    uniques.sort()

                    # save for later
                    try:
                        arg_i = schema_attr_names.index(attr)
                        if schema[arg_i]['domain'] == None:
                            schema[arg_i]['coded_values'] = uniques
                            schema[arg_i]['domain'] = 'coded_values_groupby'
                    except:
                        pass

                    attrs_stat[attr] = (stat, groups, uniques)
                    if len(uniques) > max_len:
                        max_len = len(uniques)
                tar_df = pd.DataFrame()
                for attr in tar_schema:
                    uniques = attrs_stat[attr][2]
                    attrs_stat[attr] = attrs_stat[attr][0]
                    attr_vals = uniques + ['None'] * (max_len - len(uniques))
                    tar_df[attr] = attr_vals

                # collect stats first, also compare data types
                src_datatype = kb[concept]['matches'][datasource]['data_type']
                attr_schema = schema_set[datasource]
                cols_to_delete = compare_datatypes(src_datatype, attr_schema, tar_schema)
                tar_df = df_delete_cols(tar_df, cols_to_delete)

                sim_matrix, confidence = match_table_by_values(src_data, tar_df, match_threshold, comparison_count_o, attrs_stat,
                                                               sample_ratio, sample_min_count, sample_max_count)
                print(sim_matrix.to_string())

                # save similarity matrices
                filename = '%s|%s|%s||%s.csv' % (concept, datasource, src_attr, source_name)
                sim_matrix.to_csv(filename, sep=',', encoding='utf-8')

        t3 = time.time()
        total = t3 - t2
        print('time %s sec' % (total))
        print('-----')

    # TODO:
    # use wordnet, get semantic closeness
    # get new labels
    # compute composite score (probabilities multiplied)
    # update kb splitting and merging
    # imperfect label prediction as train data
    # hybrid matching and deep neuralnet train model
    # stop iteration if percent of data are covered in kb

    # TODO change score to percentage
    # TODO once in cluster, add example values of every attribute in cluster
    # TODO in final iteration, delete empty concepts
    # TODO matching based on both char and semantics!

    # kb.serialize(format='turtle', destination='./knowledge base.txt')
    kb_file = open("kb_file.json", "w")
    json.dump(kb, kb_file, indent=2, sort_keys=True)
    # pprint.pprint(kb)

    with open('./schema_complete_list.json', 'w') as fp:
        json.dump(schema_set, fp, sort_keys=True, indent=2)

    t1 = time.time()
    total = t1 - t0
    print('time %s sec' % (total))


