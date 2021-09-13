import zipfile
import os
import shutil
import json
import csv
import copy
import asyncio

class DataModel(object):
    def __init__(self):
        self.datasets = {}
        self.tags = []
        # self.tag_map = {}
        self.groups = []

    def add_dataset(self, name, dataset):
        self.datasets[name] = dataset

    def print_data_model(self):
        print('Data Model:')
        print('tags: ' + str(self.tags))
        print('groups: ' + str(self.groups))
        print('-----')
        for key in  self.datasets:
            dataset = self.datasets[key]
            print_data_instance(dataset)
            print('-----')

class DataInstance(object):
    def __init__(self):
        self.name = ''
        self.notes = ''
        self.tags = []
        self.groups = []
        # self.group_descriptions = []
        # self.resources = []
        self.source_filename = []


    def set_name(self, name):
        self.name = name

    def add_resource(self, resource):
        self.resources.append(resource)

    def set_metadata(self, notes, tags, groups, group_descriptions):
        self.notes = notes
        self.tags = tags
        self.groups = groups
        self.group_descriptions = group_descriptions

    def set_tags(self, notes, tags, groups):
        self.notes = notes
        self.tags = tags
        self.groups = groups

    def set_source_filename(self, name):
        self.source_filename.append(name)

    def to_json(self):
        return {'name':self.name,
                'notes':self.notes,
                'tags':self.tags,
                'groups':self.groups,
                # 'group_descriptions':self.group_descriptions,
                'source_filename':self.source_filename
                }

def print_data_instance(data_instance):
    print('Data Instance:')
    print('name: ' + data_instance.name)
    print('notes: ' + data_instance.notes)
    print('tags: ' + str(data_instance.tags))
    print('groups: {')
    for i in range(len(data_instance.groups)):
        group = data_instance.groups[i]
        group_description = data_instance.group_descriptions[i]
        print('\t' + '(' + str(group) + ', ' + group_description + ')')
    print('}')

    print('resources: {')
    for i in range(len(data_instance.resources)):
        resource = data_instance.resources[i]
        print('\t' + 'format: ' + resource['format'])
        # for i in range(len(resource['data'])):
        for i in range(10):
            value = resource['data'][i]
            print('\t' + '\t' + 'value: ' + str(value))

    print('}')


    return

def create_tags(tags_json, data_model_tags):
    tags = []
    for tag in tags_json:
        index = -1
        try:
            tag_name = tag['name']
            index = data_model_tags.index[tag_name]
        except Exception as e:
            print('note: create_tags ' + str(e))
            data_model_tags.append(tag['name'])
            index = len(data_model_tags) - 1

        if len(tags) <= index:
            grow_len = index - (len(tags) - 1)
            for i in range(grow_len):
                tags.append(0)

        tags[index] = 1

    return tags

def create_groups(groups_json, data_model_groups):
    groups = []
    groups_descriptions = []
    for group in groups_json:
        index = -1
        try:
            group_name = group['display_name']
            index = data_model_groups.index[group_name]
        except Exception as e:
            print('note: create_groups ' + str(e))
            data_model_groups.append(group['display_name'])
            index = len(data_model_groups) - 1

        if len(groups) <= index:
            grow_len = index - (len(groups) - 1)
            for i in range(grow_len):
                groups.append(0)
                groups_descriptions.append('')
        groups[index] = 1
        groups_descriptions[index] = group['description']

    return groups, groups_descriptions

def create_resource_from_csv(resource_csv, data_instance, data_model):
    resource = {}
    resource['format'] = 'csv'
    resource['data'] = resource_csv

    return resource

def create_resource_from_json(resource_json, data_instance, data_model):
    resource = {}
    resource['format'] = 'json'
    resource['data'] = []

    # cnt = 0
    for element in resource_json['features']:
        row = copy.deepcopy(element['properties'])
        row['geometry_type'] = element['geometry']['type']
        resource['data'].append(row)

        # if cnt < 10:
        #     print(resource['data'][-1])
        # cnt += 1

    return resource

def unzip_and_rename():
    # zip_ref = zipfile.ZipFile('../thesis_project_dataset/parks/parks_JSON.zip', 'r')
    # zip_ref.extractall('../thesis_project_dataset/parks')
    # zip_ref.close()

    for root, dirs, files in os.walk("../thesis_project_dataset"):
        path = root.split(os.sep)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            # print(len(path) * '---', file)
            # print(root + '/' + file)
            filename, file_extension = os.path.splitext(file)
            if file == '.DS_Store':
                continue
            # 	print('skip')
            # print(file_extension)
            try:
                if file_extension == '.zip':
                    # print(root + '/' + file)
                    zip_ref = zipfile.ZipFile(root + '/' + file, 'r')
                    zip_ref.extractall(root)
                    zip_ref.close()
                if file_extension == '':
                    # print(root + '/' + file)
                    shutil.copyfile(root + '/' + file, root + '/' + file + '.csv')
            except Exception as e:
                print('error: unzip_and_rename ' + str(e))


def select_datasources():
    import os
    ls = os.listdir('./metadata')
    ls_dict = {}
    for st in ls:
        st = st.replace('-', ' ')
        st = st.replace('.', ' ')
        st = st.split(' ')
        st = st[1:-1]
        st = ' '.join(st)
        st = st.lower()
        ls_dict[st] = {'csv':[], 'json':[]}

    from similarity.ngram import NGram
    twogram = NGram(2)
    metadata_sources = ls_dict.keys()

    for root, dirs, files in os.walk("../thesis_project_dataset"):
        curr_dir_path = root.split("/")
        curr_dir_name = curr_dir_path[-1]
        for file in files:
            filename, file_extension = os.path.splitext(file)
            dataset = root.split('/')
            dataset = dataset[2:3]
            if len(dataset) != 0 and dataset[0] != '.git':
                dataset = dataset[0]
                dataset = dataset.replace('-', ' ')

                found = False
                found_val = None
                curr_score = 0
                found_datasource = None

                if dataset in ls_dict:
                    found = True

                if found:
                    dataset_collection = ls_dict[dataset]
                    found = True
                    found_val = dataset_collection
                    found_datasource = dataset
                    curr_score = 1

                if not found:
                    curr_score = 0
                    for metadata_source in metadata_sources:
                        dist = 1 - twogram.distance(dataset, metadata_source)
                        if dist < 0.85:
                            print('skip', root + '/' + file)
                            continue
                        if dist > curr_score:
                            found = True
                            found_val = ls_dict[metadata_source]
                            curr_score = dist
                            found_datasource = metadata_source

                            print('found', found, found_datasource, curr_score, file_extension, root + '/' + file)

                if not found:
                    continue

                if file_extension == '.json':
                    found_val['json'].append((root + '/' + file, curr_score))

                if file_extension == '.csv':
                    found_val['csv'].append((root + '/' + file, curr_score))



    print(ls_dict)
    for key in ls_dict:
        val = ls_dict[key]
        val['csv'] = sorted(val['csv'], key=lambda x: x[1])
        val['json'] = sorted(val['json'], key=lambda x: x[1])

    import json
    with open('datasource_and_metadata.json', 'w') as fp:
        json.dump(ls_dict, fp, sort_keys=True, indent=2)


def parse_models():
    import json
    # data_to_parse = ['important-trees', 'parks', 'park-specimen-trees']
    data_model = DataModel()

    with open('./downloadResourceURL.json', 'r') as f:

        data = json.load(f)
        for key in data:
            # if key not in data_to_parse:
            #     continue

            data_instance = DataInstance()
            value = data[key]

            # print(key)
            key = key.replace('-', ' ')
            key = key.lower()


            data_instance.set_name(key)
            # tags = create_tags(value['tags'], data_model.tags)
            # groups, group_descriptions = create_groups(value['groups'], data_model.groups)
            # data_instance.set_metadata(value['notes'], tags, groups, group_descriptions)

            data_instance.set_tags(value['notes'], value['tags'], value['groups'])

            data_model.add_dataset(key, data_instance)

    error_list = []
    file_list = []

    for root, dirs, files in os.walk("../thesis_project_dataset"):
        curr_dir_path = root.split("/")
        # curr_dir_name = curr_dir_path[-1]

        dataset = root.split('/')

        if len(dataset) < 3:
            print('  error', dataset)
            continue
        curr_dir_name = dataset[2:3]
        dataset_name = curr_dir_name[0]
        if dataset_name == '.git':
            continue

        for file in files:

            file_list.append(root + '/' + file)

            filename, file_extension = os.path.splitext(file)

            dataset_name = dataset_name.replace('-', ' ')
            dataset_name = dataset_name.lower()


            if dataset_name not in data_model.datasets:
                # print('error dataset_name', root + '/' + file)
                error_list.append(root + '/' + file)
                print('error', dataset_name)
                continue

            data_inst = data_model.datasets[dataset_name]
            data_inst.set_source_filename(root + '/' + file)
            continue

    datasets_json_obj = {}
    for key in data_model.datasets:
        dataset = data_model.datasets[key]
        dataset_json = dataset.to_json()
        datasets_json_obj[dataset_json['name']] = dataset_json

    print(datasets_json_obj)
    error_list.sort()
    print(error_list)
    print(len(error_list))

    file_list.sort()
    print(file_list)
    print(len(file_list))

    with open('datasource_and_tags.json', 'w') as fp:
        json.dump(datasets_json_obj, fp, sort_keys=True, indent=2)

        # else:
        #     continue

    # data_model.print_data_model()
    # TODO: json doesn't print

    return data_model

import nltk
nltk.data.path.append('/home/haoran/Documents/venv/nltk_data/')
from nltk.corpus import words
from nltk.corpus import wordnet


import json
def collect_concepts():
    list_of_tags = []
    list_of_groups = []
    with open('./metadata/tag_list.json', 'r') as f:
        data = json.load(f)
        # list_of_tags = [translate_to_english(key) for key in data['result']]
        list_of_tags = dict((el, translate_to_english(el)) for el in data['result'])

        # cnt = 0
        # for key in data['result']:
        #     list_of_tags.append(translate_to_english(key))
        #     cnt += 1
        #     if cnt > 10:
        #         break
        with open('./tag_list_translated.json', 'w') as fp:
            json.dump(list_of_tags, fp, sort_keys=True, indent=2)

    with open('./metadata/group_list.json', 'r') as f:
        data = json.load(f)
        # list_of_groups = [translate_to_english(key) for key in data['result']]
        list_of_groups = dict((el, translate_to_english(el)) for el in data['result'])

        # cnt = 0
        # for key in data['result']:
        #     list_of_groups.append(translate_to_english(key))
        #     cnt += 1
        #     if cnt > 10:
        #         break
        with open('./group_list_translated.json', 'w') as fp:
            json.dump(list_of_groups, fp, sort_keys=True, indent=2)


# TODO convert to RDF
def collect_concepts_from_metadata():
    list_of_tags = {}
    list_of_groups = {}

    with open('./datasource_and_tags.json', 'r') as f:
        data = json.load(f)
        for datasource_key in data:
            datasource = data[datasource_key]
            groups = datasource['groups']
            tags = datasource['tags']

            for group in groups:
                display = group['display_name']
                # descr = group['description']
                if display not in list_of_groups:
                    sources = [datasource_key]
                    list_of_groups[display] = {'sources': sources}
                    list_of_groups[display].update(translate_to_english(display))
                else:
                    group_elem = list_of_groups[display]
                    group_elem['sources'].append(datasource_key)
            for tag in tags:
                display = tag['display_name']
                if display not in list_of_tags:
                    sources = [datasource_key]
                    list_of_tags[display] = {'sources': sources}
                    list_of_tags[display].update(translate_to_english(display))
                else:
                    tag_elem = list_of_tags[display]
                    tag_elem['sources'].append(datasource_key)

        with open('./metadata_tag_list_translated.json', 'w') as fp:
            json.dump({'tags': list_of_tags, 'groups': list_of_groups}, fp, sort_keys=True, indent=2)

def parse_metadata(file):

    metadata = []

    with open(file, 'r') as f:
        # print(file)
        data = json.load(f, strict=False)

        if data['body']['fields'] == None:
            return []

        for item in data['body']['fields']:
            attr = {}
            if item['name'] != None:
                attr['name'] = item['name'].lower()
            if item['alias'] != None and item['alias'] == item['name']:
                attr['alias'] = None
            elif item['alias'] != None:
                attr['alias'] = item['alias'].lower()
            try:
                if item['domain'] == None:
                    attr['domain'] = None
                elif item['domain']['type'] == 'codedValue':
                    attr['domain'] = 'coded_values'
                    attr['coded_values'] = [value['name'] for value in item['domain']['codedValues']]
            except Exception as e:
                print('error', 'parse_metadata domain None', file, e)

            try:
                attr['data_type'] = item['type']
            except Exception as e:
                print('error', 'parse_metadata type None', file, e)

            attr['datasource'] = file

            metadata.append(attr)

    return metadata

def clean_name(st, has_prefix=False, has_extension=False):
    st = st.replace('-', ' ')
    st = st.replace('_', ' ')
    st = st.replace('.', ' ')
    st = st.split(' ')
    if has_prefix:
        st = st[1:]
    if has_extension:
        st = st[0:-1]
    st = ' '.join(st)
    st = st.lower()

    st = st.strip()
    return st

def parse_metadata_files():
    all_metadata = {}
    for root, dirs, files in os.walk("./metadata"):
        for file in files:
            if file == '.DS_Store':
                continue

            st, file_extension = os.path.splitext(file)
            if st == 'group_list' or st == 'tag_list':
                continue
            metadata = parse_metadata(root + '/' + file)
            key = clean_name(st, True, False)
            all_metadata[key] = metadata
            print('done metadata', root + '/' + file)

    with open('./metadata_complete_list.json', 'w') as fp:
        json.dump(all_metadata, fp, sort_keys=True, indent=2)

    return

import numpy
import pandas as pd

def format_json(json_data):
    resource_json = []
    keys = None

    first = json_data[0]
    keys = list(first['properties'].keys())
    keys.sort()

    for row in json_data:
        # row = json_data[element]
        row_vals = []
        for key in keys:
            value = row['properties'][key]
            if isinstance(value, dict) or type(value)==list:
                continue
            row_vals.append(value)
        resource_json.append(row_vals)

    y = numpy.array([numpy.array(xi) for xi in resource_json])

    # print(y[0])
    # print(keys)
    dataframe = pd.DataFrame(data=y, columns=keys)

    return dataframe


def format_csv(csv_data):
    y = numpy.array([numpy.array(xi) for xi in csv_data])
    # print(y[0, 0:])
    dataframe = pd.DataFrame(data=y[1:, 0:], columns=y[0, 0:])

    return dataframe

def check_added():
    added = {}
    for root, dirs, files in os.walk("../thesis_project_dataset_clean"):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            added[filename] = True

    # with open('./datasource_and_metadata.json', 'r') as f:
    #     metadata = json.load(f)
    #     for source in metadata:
    #         if source in added:
    #             print('already added', source)
    return added

def to_csv_format():

    added = check_added()
    with open('./datasource_and_metadata.json', 'r') as f:
        metadata = json.load(f)

        for source in metadata:
            if source in added:
                print('already added', source)
                continue

            num_datasources = 0
            num_csv = metadata[source]['csv']
            num_datasources += len(num_csv)
            num_json = metadata[source]['json']
            num_datasources += len(num_json)

            datasource = None
            resource = None
            type = None
            if num_datasources == 0:
                continue
            if len(num_csv) == 0:
                # then convert json to csv
                datasource = num_json[-1]
                # print(datasource)
                datasource = ''.join(datasource[0])

                print('parsing json', datasource)
                with open(datasource, mode='r') as json_file:
                    json_data = json.load(json_file, strict=False)
                    # multiple datasets here
                    candidate_json = None
                    if isinstance(json_data, dict):
                        candidate_json = json_data
                    elif not isinstance(json_data, dict) and isinstance(json_data, list):
                        for dataset in json_data:
                            if not isinstance(dataset['features'], dict):
                                continue
                            features = dataset['features']
                            if 'properties' not in features:
                                continue
                            candidate_json = dataset
                    if candidate_json == None:
                        print('[skip]', datasource)
                        continue
                    resource = format_json(candidate_json['features'])
                type = 'json'

            elif len(num_csv) > 0:
                datasource = num_csv[-1]
                # print(datasource)
                datasource = ''.join(datasource[0])

                print('parsing csv', datasource)
                csv_data = []
                with open(datasource, mode='r', encoding='unicode_escape') as csv_file:
                    # csv_reader = csv.reader(csv_file)
                    csv_data = list(list(rec) for rec in csv.reader(csv_file, delimiter=','))

                    # csv_reader = csv.DictReader(csv_file)
                    # for row in csv_reader:
                    #     csv_data.append(row)

                # print(csv_data[0])
                resource = format_csv(csv_data)
                type = 'csv'

            resource.to_csv('../thesis_project_dataset_clean/' + source + '.csv', sep=',', encoding='utf-8')
            print('write', type, '../thesis_project_dataset_clean/' + source + '.csv')

    return

import pprint
def collect_all_datatypes(path):
    schema_f = open(path, 'r')
    schema_set = json.load(schema_f, strict=False)
    data_types = {}

    for item in schema_set:
        item_data = schema_set[item]
        for data in item_data:
            if data['data_type'] not in data_types:
                data_types[data['data_type']] = 1
            else:
                data_types[data['data_type']] = data_types[data['data_type']] + 1
    # pprint.pprint(data_types)
    return data_types


if __name__ == "__main__":
    print("running parse_dataset")
    # unzip_and_rename()
    # select_datasources()
    # parse_models()
    # to_csv_format()
    # parse_metadata_files()
    # collect_concepts_from_metadata()
    # collect_all_datatypes()
    pass