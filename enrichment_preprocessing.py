import nltk

DEBUG_MODE = 0

from similarity.ngram import NGram
twogram = NGram(2)
def matcher_name(src, tar, function):
    sim_score = 1 - function.distance(src, tar)
    return sim_score

def load_dict(nltk_path):
    nltk.data.path.append(nltk_path)
    from nltk.corpus import wordnet
    dictionary = wordnet
    return dictionary

import pandas as pd
import inflection
import math
def matcher_name_meaning_by_thesaurus(src, tar, dictionary, threshold):

    # threshold = 0.2
    # top_rows = 0.05

    top_rows = 1.0

    src_word_vec = src.split(' ')
    tar_word_vec = tar.split(' ')

    src_word_enrich = {word: {} for word in src_word_vec}
    tar_word_enrich = {word: {} for word in tar_word_vec}

    for word1 in tar_word_vec:
        word1 = inflection.singularize(word1)

        w1 = None
        try:
            w1 = dictionary.synsets(word1, pos=dictionary.NOUN)
            tar_word_enrich[word1] = w1
        except Exception:
            continue

    for word2 in src_word_vec:
        word2 = inflection.singularize(word2)

        w2 = None
        try:
            w2 = dictionary.synsets(word2, pos=dictionary.NOUN)
            src_word_enrich[word2] = w2
        except Exception:
            continue

    # sem_sim_score = 0
    sims_list = []
    for word1 in tar_word_vec:
        for word2 in src_word_vec:
            if len(tar_word_enrich[word1]) == 0 or len(src_word_enrich[word2]) == 0:
                continue

            # synonym approach 1
            # intersect = set(tar_word_enrich[word1].lemma_names).intersection(set(src_word_enrich[word2].lemma_names))
            # print(word1, tar_word_enrich[word1], word2, src_word_enrich[word2], intersect)

            # synonym approach 2
            sims = find_similarity_between_synset_pairs(tar_word_enrich[word1], src_word_enrich[word2], dictionary)
            # print(sims.to_string())

            num_pairs = len(sims.index)
            sims = sims.head(max(int(math.ceil(num_pairs * top_rows)), 1))
            # sims = sims.head(1)

            append_pair = False
            for k, v in sims.iterrows():
                lemmas_1 = show_all_lemmas(v['synset_1'], [])
                lemmas_2 = show_all_lemmas(v['synset_2'], [])

                hyperset_1 = set([i for i in v['synset_1'].closure(lambda s:s.hypernyms())])
                hyperset_2 = set([i for i in v['synset_2'].closure(lambda s: s.hypernyms())])

                if DEBUG_MODE:
                    print(lemmas_1, lemmas_2)
                    print(v['synset_2'] in hyperset_1)
                    print(v['synset_1'] in hyperset_2)




                if v['sim'] > threshold:
                    append_pair = True

            if append_pair == True:
                sims_list.append((word1, word2, sims))


    for word1 in tar_word_vec:
        for word2 in src_word_vec:
            if len(tar_word_enrich[word1]) == 0 or len(src_word_enrich[word2]) == 0:
                sim2 = matcher_name(word1, word2, twogram)
                # if 'businesses' in word1 or 'businesses' in word2:
                #     print('===', word1, word2, ' : ', sim2)

                if sim2 > threshold*2:
                    df = pd.DataFrame(columns=['sim', 'word1', 'word2'])
                    df = df.append({'sim': sim2, 'word1': word1, 'word2': word2}, ignore_index=True)
                    sims_list.append((word1, word2, df))


    ## compute score method 1
    #         sem_sim = w1.wup_similarity(w2)
    #         sem_sim_score += sem_sim
    #         # print(w1, w2, sem_sim)
    #
    # sem_sim_score = sem_sim_score / (len(src_word_vec) * len(tar_word_vec))
    # return sem_sim_score

    if len(sims_list) == 0:
        # print('empty: ', src, tar)
        return 0, []

    scores = []
    for sims_tuple in sims_list:
        sims = sims_tuple[2]
        for k, v in sims.iterrows():

            # word1 = sims_tuple[0]
            # word2 = sims_tuple[1]
            # sim2 = matcher_name(word1, word2, twogram)
            # if 'businesses' in word1 or 'businesses' in word2:
            #     print(word1, word2, ' : ', v['sim'], sim2)
            # v['sim'] = max(v['sim'], sim2)

            sim_score = v['sim']
            scores.append(sim_score)

    return max(scores), sims_list

def find_similarity_between_synset_pairs(synsets_1, synsets_2, wn):
    df = pd.DataFrame(columns=['sim', 'synset_1', 'synset_2'])

    for synset_1 in synsets_1:
        for synset_2 in synsets_2:
            sim = wn.path_similarity(synset_1, synset_2)

            # sim = w1.wup_similarity(w2)
            if sim is not None:

                df = df.append({'sim': sim, 'synset_1': synset_1, 'synset_2': synset_2}, ignore_index=True)

    df = df.sort_values(by=['sim'], ascending=False)
    return df

def show_all_lemmas(synset, exclude):

    lemmas = []
    lemmas += [str(lemma.name()) for lemma in synset.lemmas()]
    lemmas = [synonym.replace("_", " ") for synonym in lemmas]
    lemmas = list(set(lemmas))
    lemmas = [synonym for synonym in lemmas if synonym not in exclude]
    return lemmas


import pprint
import numpy as np
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as ssd
import scipy

def hierarchical_cluster_linkage(features, decision_threshold):
    if DEBUG_MODE: print('hierarchical_cluster_linkage:')
    if DEBUG_MODE: print(features)
    if len(features) < 2:
        return []

    arr = scipy.array(features)
    pd = ssd.pdist(arr, metric='cosine')
    z = hac.linkage(pd)

    if DEBUG_MODE: pprint.pprint(pd)

    part = hac.fcluster(z, decision_threshold, 'distance')
    return part


import pickle
import json
import numpy as np

def cluster_topics_prep(metadata_f, dir):

    dictionary = load_dict()
    metadata_set = json.load(metadata_f)

    total_topics_num = len(metadata_set['groups']) + len(metadata_set['tags'])

    groups = list(metadata_set['groups'].keys())
    tags = list(metadata_set['tags'].keys())

    groups.sort()
    tags.sort()

    index_of = 0
    topics_dict = {}
    duplicate_list = []
    topics_new = groups.copy()
    for group in groups:
        if group in tags:
            # print('error', group)
            duplicate_list.append(group)
            topics_new.remove(group)
            continue
        topics_dict[group] = index_of
        index_of += 1

    for tag in tags:
        topics_dict[tag] = index_of
        index_of += 1
    topics_new.extend(tags)

    sim_matrix = np.zeros(shape=(index_of,index_of))

    num_topics = len(topics_new)
    row_index = 0
    for topic in topics_new:

        for col_index in range(row_index, num_topics, 1):
            target = topics_new[col_index]

            threshold = 0.34
            score, sims_list = matcher_name_meaning_by_thesaurus(topic, target, dictionary, threshold)
            if score > threshold:
                print(topic, '<=>', target, ' : ', score)
            sim_matrix[row_index][col_index] = score

        np.savetxt(dir + "topic_sims"+str(row_index)+".csv", sim_matrix, delimiter=",")
        print('done topic #', row_index)
        row_index += 1

    np.savetxt(dir+"topic_sims.csv", sim_matrix, delimiter=",")

    with open(dir+'topics.txt', 'wb') as fp:
        pickle.dump(topics_new, fp)

    return

def cluster_topics_prep_matrix(dir):
    sim_matrix = np.loadtxt(dir+"topic_sims756.csv", delimiter=',')

    with open(dir+'topics.txt', 'rb') as fp:
        topics_new = pickle.load(fp)

    for i in range(len(topics_new)):
        for j in range(len(topics_new)):
            sim_matrix[i][j] = round(sim_matrix[i][j], 4)
            sim_matrix[j][i] = sim_matrix[i][j]

    np.savetxt(dir + "topic_sims.csv", sim_matrix, delimiter=",")
    print('done saving')

    return

def cluster_topics(dir):
    sim_matrix = np.loadtxt(dir+"topic_sims.csv", delimiter=',')

    decision_threshold = 0.45
    part = hierarchical_cluster_linkage(sim_matrix, decision_threshold)
    pprint.pprint(part)

    part_indexes = [[part[i], i] for i in range(len(part))]
    part_indexes_df = pd.DataFrame(part_indexes, columns=['group', 'index'])
    groups_df = part_indexes_df.groupby('group')['index'].apply(list)

    print(groups_df.head())

    groups_df.to_csv(dir + 'topics_groups.csv', sep=',', encoding='utf-8', index=False)

    with open(dir+'topics.txt', 'rb') as fp:
        topics_new = pickle.load(fp)

    reverse_topic_cluster = {topics_new[item]: index for index, row in groups_df.iteritems() for item in row}
    topic_cluster = {index: list(row) for index, row in groups_df.iteritems()}

    return groups_df, reverse_topic_cluster, topic_cluster

def isint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def open_updated_topics(dir, source_name):
    updated_topics_file = dir + 'new_topics_[' + source_name + '].txt'
    if os.path.isfile(updated_topics_file):
        with open(updated_topics_file, 'rb') as fp:
            updated_topics_this = pickle.load(fp)

            return True, updated_topics_this

    return False, None

import textwrap

def print_datasets_with_topic(dataset_metadata_set, dataset_with_topic, dir, brief, dataset_path, table_stats):

    if not os.path.isfile(dataset_path+dataset_with_topic+'.csv'):
        return False
    if not os.path.isfile(table_stats+dataset_with_topic+'.json'):
        return False

    print('     dataset: ', dataset_with_topic)
    dataset_existing_tags = dataset_metadata_set[dataset_with_topic]['tags']
    dataset_existing_groups = dataset_metadata_set[dataset_with_topic]['groups']
    dataset_notes = dataset_metadata_set[dataset_with_topic]['notes']

    dataset_existing_tags = [tag['display_name'] for tag in dataset_existing_tags]
    dataset_existing_groups = [group['display_name'] for group in dataset_existing_groups]
    dataset_notes = [word for word in dataset_notes.split() if "http://" not in word]

    print('     =existing in original=')
    print('     tags: ', textwrap.fill(str(dataset_existing_tags), 120))
    print('     groups: ', textwrap.fill(str(dataset_existing_groups), 120))
    # print('     notes: ', textwrap.fill(' '.join(dataset_notes), 120))
    print('     notes: ', ' '.join(dataset_notes))


    # get more from updated list
    print('     =updated topics already stored=')
    update, updated_topics_this = open_updated_topics(dir, dataset_with_topic)
    if update: print(textwrap.fill(str(updated_topics_this)), 120)

    if brief: return True

    print('     =stats for table=')

    dataset_f = open(table_stats+dataset_with_topic+'.json', 'r')
    dataset_stats = json.load(dataset_f)
    prt_strs = ''
    for attr in dataset_stats:
        sorted_dataset_stats = sorted(dataset_stats[attr].items(), key=lambda kv: kv[1])
        sorted_dataset_stats.reverse()
        len_stats = min(2, len(sorted_dataset_stats))
        prt_str = attr + ' :: ' + str(sorted_dataset_stats[:len_stats])
        prt_strs += prt_str + '  ||  '

    print('     attrs: ', textwrap.fill(prt_strs, 120))
    print('--')

    return True

def input_from_command(add_list, delete_list, topics_new):
    option = input("==add topics==")
    adding = option.split(',')
    # print(adding)
    # print(adding)
    adding2_tmp = [str.replace('\'', '') for str in adding if not isint(str) and str.replace('\'', '') in topics_new]
    adding2 = [topics_new[topics_new.index(str)] for str in adding2_tmp if not isint(str) and str in topics_new]

    adding3 = [topics_new[int(num)] for num in adding if isint(num) and int(num) < len(topics_new)]
    adding3_raw = [num for num in adding if isint(num) and int(num) < len(topics_new)]

    # print(adding, adding2)
    add_list.extend(adding3)
    add_list.extend(adding2)
    diff = list(set(adding) - set(add_list) - set(adding3_raw))
    if len(diff) > 0 and not (len(diff) == 1 and '' in diff): print('not added:', diff)

    option = input("==del topics==")
    deleting = option.split(',')
    deleting = [topics_new[int(num)] for num in deleting if isint(num)]
    delete_list.extend(deleting)

    return add_list, delete_list



import parse_surrey_dataset as pds
import build_matching_model as bmm
import os.path

def recommend_labels(dataset_metadata_set, metadata_set, schema_set, datasets_path, datasources_with_tag, dir, dataset_stats):
    import json
    import pandas as pd

    bmm.gather_statistics(schema_set, datasources_with_tag, dataset_stats, datasets_path)

    with open(dir+'topics.txt', 'rb') as fp:
        topics_new = pickle.load(fp)

    sim_matrix = np.loadtxt(dir + "topic_sims.csv", delimiter=',')
    clusters_of_concepts, reverse_topic_cluster, topic_cluster = cluster_topics()

    exclude_list = ['comments','condition','shape', 'material']
    exact_match = {'location':'location', 'status':'status', 'owner':'owners', 'node number':['node','nodes'], 'facilityid':['facility', 'facilities'], 'project number':['plan number','projects'], 'image':['image', 'imagery'], 'address': ['address', 'addresses']}

    topic_similarities = {}
    for topic in topics_new:
        sims = sim_matrix[topics_new.index(topic)]
        non_zero = [(i,topics_new[i]) for i, e in enumerate(sims) if e != 0]
        topic_similarities[topic] = non_zero


    for source_name in datasources_with_tag:
        print('-----', source_name, '-----')
        dataset = pd.read_csv(datasets_path + source_name + '.csv', index_col=0, header=0)

        data_samples = dataset.head()
        col_names = schema_set[source_name]
        dataset_existing_tags = dataset_metadata_set[source_name]['tags']
        dataset_existing_groups = dataset_metadata_set[source_name]['groups']
        dataset_notes = dataset_metadata_set[source_name]['notes']

        # display all datasets for each topic
        datasets_with_tag = {}
        for tag in dataset_existing_tags:
            tag_name = tag['display_name']
            print(tag_name)
            # print(metadata_set['tags'][tag_name])
            datasets_with_tag[tag_name] = metadata_set['tags'][tag_name]['sources']

        # get more topics from updated list
        updated_topics_this = open_updated_topics(dir, source_name)
        print(updated_topics_this)

        # allow break out from current source
        option = input("==skip this dataset? (y)==")
        if option == "y":
            continue

        op = input("==skip topic<->attr? (y)==")

        dataset_existing_tag_names = [item['display_name'] for item in dataset_existing_tags]


        add_list = []
        # recommend topics close to dataset
        for col in col_names:
            if op == "y": continue
            if col['alias'] != None:
                col_name = col['alias']
            else:
                col_name = col['name']

            if col_name in exact_match:
                if isinstance(exact_match[col_name], list): add_list.extend(exact_match[col_name])
                if isinstance(exact_match[col_name], str): add_list.append(exact_match[col_name])

                continue

            threshold = 0.5
            for topic in topics_new:
                if topic in dataset_existing_tag_names: continue

                score = matcher_name(col_name, topic, twogram)
                # TODO also semantic score

                if col_name in exclude_list: continue
                if score > threshold:
                    print(topic, "<=>", col_name, score)
                    # print("similar topics")
                    print("/similar topics/    ", topic_similarities[topic])
                    # print("topic cluster")
                    cluster_for_topic = topic_cluster[reverse_topic_cluster[topic]]
                    # topic ids to name
                    print('/topic cluster/    ',[topics_new[id] for id in cluster_for_topic])
                    # print("datasets with topic")
                    if topic in metadata_set['tags']:
                        print("/datasets with topic/    ", metadata_set['tags'][topic]['sources'])
                    elif topic in metadata_set['groups']:
                        print("/datasets with topic/    ",metadata_set['groups'][topic]['sources'])
                    else:
                        print("error: topic cannot be found")
                    # print("data samples")
                    # print(data_samples)
                    # print("data description")
                    print("/data description/    ",dataset_notes)
                    # print("data topics")
                    print("/data topics/    ",dataset_existing_tag_names)
                    # print("data groups")
                    print("/data groups/    ",[item['display_name'] for item in dataset_existing_groups])
                    print("-----")

                    option = input("==accept or reject (y)==")
                    if option == "a" or option == "y":
                        print("accept", topic)
                        add_list.append(topic)
                    else:
                        print("reject", topic)

                    print('----====')

            print('----------')
            print()
            print()
            print()
        print('==========')
        print()
        print()
        print()
        print()
        print()

        op2 = input("==skip topic<->datasets (y)==")

        visited_datasets = []
        delete_list = []
        visited_topics = []
        for topic in dataset_existing_tags:
            if op2 == "y": continue
            print("[",topic['display_name'],"]")
            similar_topics = topic_similarities[topic['display_name']]
            print("similar topics:")

            # show all datasets with topic, and additional topics each dataset has
            # TODO do not show duplicate datasets
            for topic_sim in similar_topics:
                topic_sim_name = topic_sim[1]
                if topic_sim_name in visited_topics or topic_sim_name in add_list: continue
                visited_topics.append(topic_sim_name)

                print()
                print('=    ', topic_sim)

                datasets_with_topic = None

                brief = False
                if topic_sim_name in metadata_set['groups']:
                    # print('found in groups')
                    datasets_with_topic = metadata_set['groups'][topic_sim_name]['sources']
                    brief = True

                if topic_sim_name in metadata_set['tags']:
                    # print('found in tags')
                    datasets_with_topic = metadata_set['tags'][topic_sim_name]['sources']

                if topic_sim_name in metadata_set['groups'] or topic_sim_name in metadata_set['tags']:
                    count_sim_topics_printed = 0

                    for dataset_with_topic in datasets_with_topic:
                        if dataset_with_topic == source_name:
                            continue
                        if dataset_with_topic in visited_datasets: continue
                        visited_datasets.append(dataset_with_topic)

                        count_sim_topics_printed += 1
                        printed = print_datasets_with_topic(dataset_metadata_set, dataset_with_topic, dir+'updated_topics/', brief)
                        if not printed: continue
                        # TODO also print table values
                        if count_sim_topics_printed % 5 == 0:
                            add_list, delete_list = input_from_command(add_list, delete_list, topics_new)

                print('-===-')

                add_list, delete_list = input_from_command(add_list, delete_list, topics_new)


                print("-add_list-", textwrap.fill(str(add_list), 120))
                print("-delete_list-", textwrap.fill(str(delete_list), 120))
                old_list = [item['display_name'] for item in dataset_existing_tags]
                print("-old_list-", textwrap.fill(str(old_list), 120))

        while True:
            print("==add additional topics or delete existing topics==")
            add_list, delete_list = input_from_command(add_list, delete_list, topics_new)

            option = input("done? (e)")
            if option == "e":
                break

            continue

        print("-add_list-", add_list)
        print("-delete_list-", delete_list)


        print("[save]")

        existing_tags = [item['display_name'] for item in dataset_existing_tags]
        existing_groups = [item['display_name'] for item in dataset_existing_groups]

        print("-existing data topics-", existing_tags)
        print("-existing data groups-", existing_groups)

        updated_topics_path = source_name
        updated_exists, updated_topics = open_updated_topics(dir+'updated_topics/', updated_topics_path)
        updated_exists2, updated_topics2 = open_updated_topics(dir, updated_topics_path)
        if not updated_exists: updated_topics = []
        if not updated_exists2: updated_topics2 = []
        updated_topics = list(updated_topics)
        updated_topics2 = list(updated_topics2)
        # print(updated_topics, updated_topics2)

        new_topics_set = set([*add_list, *existing_tags, *existing_groups, *updated_topics, *updated_topics2])

        for item in delete_list:
            if item not in new_topics_set: continue
            new_topics_set.remove(item)
        print(textwrap.fill(str(new_topics_set), 120))

        with open(dir + 'new_topics_['+source_name+'].txt', 'wb') as fp:
            pickle.dump(new_topics_set, fp)

    return


class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        """
        out : ['What', 'can', 'I', 'say', 'about', 'this', 'place', '.']
        """
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("_", " ")

        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens


class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self,tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]


        # lemmatization using pos tagg
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        pos_tokens = [ [(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), pos_tag) for (word,pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens



def enrich_homonyms_test(nltk_path):
    import nltk
    nltk.data.path.append(nltk_path)
    from nltk.corpus import wordnet
    from nltk.stem.wordnet import WordNetLemmatizer

    # synonyms = []
    # antonyms = []
    #
    # for syn in wordnet.synsets("good"):
    #     for l in syn.lemmas():
    #         synonyms.append(l.name())
    #         if l.antonyms():
    #             for ant in l.antonyms():
    #                 antonyms.append(ant.name())
    #
    # for syn in wordnet.synsets("bad"):
    #     for l in syn.lemmas():
    #         antonyms.append(l.name())
    #         if l.antonyms():
    #             for ant in l.antonyms():
    #                 synonyms.append(ant.name())

    ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    word = "park"
    syns = wordnet.synsets(word)

    syns_n = wordnet.synsets(word, pos='n')
    print('[syn.n]', syns_n)

    # plural or tense recognized
    print('[syn][morph]', wordnet.morphy(word+'s', wordnet.NOUN))

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize('parking', wordnet.VERB))

    print('=====')

    for syn in syns:
        print('[syn]', syn.name())
        print('[def]', syn.definition())
        print('[ex]', syn.examples())

        synset = wordnet.synset(syn.name())

        hypo = []
        for s in synset.closure(lambda s: s.hyponyms()):
            hypo.append(s)

        hyper = []
        for s in synset.closure(lambda s: s.hypernyms()):
            hyper.append(s)

        hyper1 = [s for s in synset.closure(lambda s: s.hypernyms(), depth=1)]
        hypo1 = [s for s in synset.closure(lambda s: s.hyponyms(), depth=1)]

        # print(syn.hypernyms())
        # print(syn.hyponyms())
        if len(hyper1) > 0: print('[syn][clos1][hyper]', list(hyper1))
        if len(hyper) > 0: print('[syn][clos][hyper]', list(hyper))
        if len(hypo1) > 0: print('[syn][clos1][hypo]', list(hypo1))
        if len(hypo) > 0: print('[syn][clos][hypo]', list(hypo))

        if len(syn.topic_domains()) > 0: print('___[syn][topic]', syn.topic_domains())
        if len(syn.region_domains()) > 0: print('___[syn][region]', syn.region_domains())
        if len(syn.usage_domains()) > 0: print('___[syn][usage]', syn.usage_domains())

        print('+++lemmas+++', len(syn.lemmas()))
        for lem in syn.lemmas():
            print('[syn][lem]', lem.name())
            # frequency in corpus
            print('[syn][lem][count]', lem.count())
            # verb and noun are related
            if len(lem.derivationally_related_forms()) > 0: print('___[syn][lem][rel]', lem.derivationally_related_forms())

            if lem.antonyms():
                print('+++antonyms+++', len(lem.antonyms()))
                for ant in lem.antonyms():

                    print('[ant]', ant.name())
                    if len(ant.synset().lowest_common_hypernyms(syn)) > 0: print('___[ant][com]', ant.synset().lowest_common_hypernyms(syn))
                    print('[ant][syn]', ant.synset().path_similarity(syn))
                print('+++++')

            print('-----')
        print('=====\n')


    return

import pprint
def compare_source_target(source_context, target_context):
    enriched_topic_prob = {}

    # print(source_context["desc"])
    # print(source_context["notes"])
    # print(target_context["defn"])
    # print(target_context["expl"])

    # pprint.pprint(source_context)
    # pprint.pprint(target_context)
    # print('-----')

    source_words = []
    # print(source_context['dataset_name'])
    source_words.extend([name for item in source_context['dataset_name'] for name in item])
    source_words.extend(source_context['desc'])
    source_words.extend(source_context['notes'])
    source_words.extend(source_context['other_topics'])
    source_words.extend(source_context['topic'])
    source_words.extend(source_context['topic_singl'])

    target_words = []
    target_words.extend([name for item in target_context['antonyms'] for name in item])
    target_words.extend([name for item in target_context['lemmas'] for name in item])
    target_words.extend([name for item in target_context['deriv_rel'] for name in item])
    target_words.extend(target_context['defn'])
    target_words.extend(target_context['expl'])
    target_words.extend([name for ex in target_context['hyper1_lemmas'] for item in ex for name in item])
    target_words.extend([name for ex in target_context['hypo1_lemmas'] for item in ex for name in item])
    target_words.extend([name for ex in target_context['region_dom'] for item in ex for name in item])
    target_words.extend([name for ex in target_context['topic_dom'] for item in ex for name in item])
    target_words.extend([name for ex in target_context['usage_dom'] for item in ex for name in item])

    # print(source_words)
    # print(target_words)
    # print('-----')

    source_words = {word for word in  set(source_words) if word != source_context['topic_singl']}
    target_words = {word for word in  set(target_words) if word != source_context['topic_singl']}

    return {'source_words': list(source_words), 'target_words': list(target_words), 'overlap': list(source_words & target_words)}

def get_lemma_name_from_synsets(synsets, splitter):
    lemmas = []
    for syn in synsets:
        for lem in syn.lemmas():
            lemmas.append(splitter.split(lem.name()))
    return lemmas


def enrich_topic_words(topic, top_context, wordnet):
    list_of_enriched = []
    for word in topic:
        enriched_topic, enriched_topic_prob = enrich_topic(word, top_context, wordnet)
        list_of_enriched.append([enriched_topic, enriched_topic_prob])
    return list_of_enriched

def enrich_topic(topic, top_context, wordnet):
    lemmatizer = WordNetLemmatizer()
    splitter = Splitter()
    lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()

    print('======', top_context['desc'])

    if len(top_context["desc"]) != 0 and type(top_context["desc"][0]) is str:
        pass
    else:
        top_context["desc"] = [tok[1] for sent in top_context['desc'] for tok in sent if 'NN' in tok[2]]

    if len(top_context["notes"]) != 0 and type(top_context["notes"][0]) is str:
        pass
    else:
        top_context["notes"] = [tok[1] for sent in top_context['notes'] for tok in sent if 'NN' in tok[2]]

    enriched_topic = {}
    enriched_topic_prob = {}

    syns = wordnet.synsets(topic, pos='n')
    topic_m = wordnet.morphy(topic, wordnet.NOUN)
    syns_m = []
    if topic_m != None:
        syns_m = wordnet.synsets(topic_m, pos='n')

    if len(syns) >= len(syns_m):
        pass
    else:
        syns = syns_m


    for syn in syns:
        defn = syn.definition()
        expl = syn.examples()
        syn_name = syn.name()

        synset = wordnet.synset(syn.name())

        hyper1 = [s for s in synset.closure(lambda s: s.hypernyms(), depth=1)]
        hypo1 = [s for s in synset.closure(lambda s: s.hyponyms(), depth=1)]
        hyper = [s for s in synset.closure(lambda s: s.hypernyms())]
        hypo = [s for s in synset.closure(lambda s: s.hyponyms())]

        syn_context = {'syn_name': syn_name,
                       'expl': expl,
                       'defn': defn,
                       'hyper': hyper,
                       'hypo': hypo,
                       'hyper_num_depth1': len(hyper1),
                       'hypo_num_depth1': len(hypo1),
                       'topic_dom': syn.topic_domains(),
                       'region_dom': syn.region_domains(),
                       'usage_dom': syn.usage_domains(),
                       'lemmas': [],
                       'lem_deriv_rel': [],
                       'anto': []}

        for lem in syn.lemmas():
            lem_name = lem.name()

            syn_context['lemmas'].append(lem)
            # syn_context['lemmas'].append(syn_name+'.'+lem_name)
            # print(syn_name+'.'+lem_name)
            # lemma = wordnet.lemma(syn_name+'.'+lem_name)
            # print(lemma)

            deriv_rel = lem.derivationally_related_forms()
            syn_context['lem_deriv_rel'].extend(deriv_rel)

            if not lem.antonyms():
                continue

            for ant in lem.antonyms():
                ant_name = ant.name()
                syn_context['anto'].append(ant)

        # pprint.pprint(syn_context)
        enriched_topic[syn_name] = syn_context


        defn = splitter.split(defn)
        expl = [splitter.split(ex) for ex in expl]

        if len(defn) != 0: defn = lemmatization_using_pos_tagger.pos_tag(defn)
        if len(expl) != 0:
            expl = [lemmatization_using_pos_tagger.pos_tag(ex) for ex in expl]

        hyper1_lemmas = get_lemma_name_from_synsets(hyper[:syn_context['hyper_num_depth1']], splitter)
        hypo1_lemmas = get_lemma_name_from_synsets(hypo[:syn_context['hypo_num_depth1']], splitter)

        lemmas = [lem.name().split('_') for lem in syn_context['lemmas']]
        antonyms = [ant.name().split('_') for ant in syn_context['anto']]

        deriv_rel = [lem.name().split('_') for lem in syn_context['lem_deriv_rel']]
        topic_dom = get_lemma_name_from_synsets(syn_context['topic_dom'], splitter)
        region_dom = get_lemma_name_from_synsets(syn_context['region_dom'], splitter)
        usage_dom = get_lemma_name_from_synsets(syn_context['usage_dom'], splitter)


        target_context = {'defn': defn,
                          'expl': expl,
                          'hyper1_lemmas': hyper1_lemmas,
                          'hypo1_lemmas': hypo1_lemmas,
                          'lemmas': lemmas,
                          'antonyms': antonyms,
                          'deriv_rel': deriv_rel,
                          'topic_dom': topic_dom,
                          'region_dom': region_dom,
                          'usage_dom': usage_dom}

        target_context["defn"] = [tok[1] for sent in target_context['defn'] for tok in sent if 'NN' in tok[2]]
        target_context["expl"] = [tok[1] for ex in target_context['expl'] for sent in ex for tok in sent if 'NN' in tok[2]]

        enriched_topic_prob_i = compare_source_target(top_context, target_context)
        enriched_topic_prob[syn_name] = enriched_topic_prob_i

        lowest_common_hypernyms = None
        path_similarity = None

    for syn_context_k in enriched_topic:
        syn_context = enriched_topic[syn_context_k]
        syn_context['anto'] = [anto.name() for anto in syn_context['anto']]
        syn_context['hyper'] = [hyper.name() for hyper in syn_context['hyper']]
        syn_context['hypo'] = [hypo.name() for hypo in syn_context['hypo']]
        syn_context['lem_deriv_rel'] = [syn_context['syn_name']+'.'+lem_deriv_rel.name() for lem_deriv_rel in syn_context['lem_deriv_rel']]
        syn_context['lemmas'] = [syn_context['syn_name']+'.'+lemmas.name() for lemmas in syn_context['lemmas']]
        syn_context['region_dom'] = [region_dom.name() for region_dom in syn_context['region_dom']]
        syn_context['topic_dom'] = [topic_dom.name() for topic_dom in syn_context['topic_dom']]
        syn_context['usage_dom'] = [usage_dom.name() for usage_dom in syn_context['usage_dom']]

    # pprint.pprint(enriched_topic)

    return enriched_topic, enriched_topic_prob

# import nltk
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
splitter = Splitter()
lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()


import os.path


def enrich_homonyms(dataset_name, topic, desc, notes, other_topics, dir):

    fname = dir + '[' + dataset_name + '_' + ' '.join(topic) + '].json'
    if os.path.isfile(fname):
        print('[' + dataset_name + '_' + ' '.join(topic) + '].json', 'already exists')
        return

    other_topics_singl = []
    for top in other_topics:
        try:
            other_topics_singl.append(inflection.singularize(top))
        except Exception:
            other_topics_singl.append(top)

    topic_singl = []
    for word in topic.split():
        try:
            word_singl = inflection.singularize(word)
            topic_singl.append(word_singl)
        except Exception:
            topic_singl.append(word)

    topic = topic.split()

    # step 1 split document into sentence followed by tokenization
    desc_tokens = splitter.split(desc)
    # step 2 lemmatization using pos tagger
    desc_pos_token = lemmatization_using_pos_tagger.pos_tag(desc_tokens)
    # TODO prune out words not in any other context items

    # step 1 split document into sentence followed by tokenization
    notes_tokens = splitter.split(notes)
    # step 2 lemmatization using pos tagger
    notes_pos_token = lemmatization_using_pos_tagger.pos_tag(notes_tokens)

    # step 1 split document into sentence followed by tokenization
    name_tokens = splitter.split(dataset_name)

    top_context = {'dataset_name': name_tokens,
                   'topic': topic,
                   'topic_singl': topic_singl,
                   'desc': desc_pos_token,
                   'notes': notes_pos_token,
                   'other_topics': other_topics_singl}

    # pprint.pprint(top_context)
    # print('-----')

    # negative_topic = ['car', 'parking']
    # threshold = 0.3
    set_of_enriched = enrich_topic_words(topic, top_context, wordnet)

    import json
    with open(dir + '[' + dataset_name + '_' + ' '.join(topic) + '].json', 'w') as outfile:
        json.dump([top_context, set_of_enriched], outfile)

    for enriched_topic, enriched_topic_prob in set_of_enriched:
        # pprint.pprint(enriched_topic)

        ranked = {syn: (
        enriched_topic_prob[syn]['overlap'], enriched_topic[syn]['defn'], len(enriched_topic_prob[syn]['overlap'])) for
                  syn in enriched_topic_prob}
        ranked = sorted(ranked.items(), key=lambda kv: kv[1][2])
        ranked.reverse()
        pprint.pprint(ranked)

    return


wordnet = load_dict()
# cluster_topics_prep_matrix()
# cluster_topics()




def enrich_topics_full_run(datasources_with_tag, dir):
    for dataset_name in datasources_with_tag:


        dataset_existing_tags = dataset_metadata_set[dataset_name]['tags']
        dataset_existing_groups = dataset_metadata_set[dataset_name]['groups']
        dataset_notes = dataset_metadata_set[dataset_name]['notes']

        desc = ''
        for group in dataset_existing_groups:
            desc = ' ' + group['description']

        dataset_existing_tags = [tag['display_name'] for tag in dataset_existing_tags]
        dataset_existing_groups = [group['display_name'] for group in dataset_existing_groups]
        dataset_notes = [word for word in dataset_notes.split() if "http://" not in word]

        notes = ' '.join(dataset_notes)

        for topic in dataset_existing_tags:
            other_topics = dataset_existing_tags.copy()
            other_topics.remove(topic)

            enrich_homonyms(dataset_name, topic, desc, notes, other_topics, dir)

    return

if __name__ == "__main__":

    dataset_metadata_f = open('./inputs/datasource_and_tags.json', 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    metadata_f = open('./inputs/metadata_tag_list_translated.json', 'r')
    metadata_set = json.load(metadata_f)

    schema_f = open('./inputs/schema_complete_list.json', 'r')
    schema_set = json.load(schema_f, strict=False)

    datasets_path = './thesis_project_dataset_clean/'

    dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/'
    nltk_path = '/Users/haoran/Documents/nltk_data/'

    dataset_path = '/Users/haoran/Documents/thesis_schema_integration/thesis_project_dataset_clean/'
    table_stats = '/Users/haoran/Documents/thesis_schema_integration/inputs/dataset_statistics/'

    # group = 'environmental services'
    # datasources_with_tag = metadata_set['groups'][group]['sources']
    # print(datasources_with_tag)

    datasources_with_tag = []
    # datasources_with_tag = [datasource_file for datasource_file in datasources_with_tag if
    #                         os.path.isfile(datasets_path + datasource_file + '.csv')]
    # datasources_with_tag = ['aquatic hubs','drainage 200 year flood plain','drainage water bodies','park specimen trees','parks', 'park screen trees', ]
    # datasources_with_tag = ['park natural areas', 'terrestrial hubs', 'park structures']
    # datasources_with_tag = [ 'drainage dyke infrastructure', 'drainage erosion protection works', 'drainage flood control', 'drainage sub catchments', 'ecosystem corridors', 'ecosystem sites', 'park passive grass', 'park paths and trails', 'park unimproved parkland']
    # datasources_with_tag = ['collection secondary suites', 'collection strata complexes', 'collection route boundaries', 'collection rear laneways', 'recycling toter collection complexes', 'litter containers', 'schools', 'elementary school catchments', 'garbage recycling collection days', 'secondary school catchments']
    #
    # datasources_with_tag = ['park potential donation bench locations', 'park sports fields', 'walking routes', 'park horticultural beds', 'heritage routes', 'heritage sites', 'park playgrounds', 'park horticultural zones', 'park outdoor recreation facilities', 'park trans canada trail', 'important trees']
    #
    # datasources_with_tag = ['drainage detention ponds', 'sanitary lift stations', 'water valves', 'sanitary flow system nodes', 'water service areas']
    # datasources_with_tag = [  'drainage manholes', 'park catch basins', 'drainage open channels', 'water pipe bridges', 'sanitary catchments']
    # datasources_with_tag = ['drainage service connections', 'water meters', 'sanitary chambers', 'drainage major catchments', 'water pressure zones', 'signs']
    # datasources_with_tag =[ 'drainage pump stations', 'water chambers', 'water sampling stations', 'drainage laterals']
    # datasources_with_tag = ['sanitary manholes', 'water fittings', 'sanitary valves', 'drainage monitoring stations', 'water assemblies', 'water utility facilities', 'drainage devices', 'park lights']
    # datasources_with_tag = ['sanitary nodes', 'sanitary laterals', 'drainage catch basins']
    #
    # datasources_with_tag = ['truck routes', 'sidewalks', 'vehicular bridges', 'bike routes', 'poles', 'road centrelines', 'traffic calming', 'medians', 'traffic signals', 'adopt a street']
    # datasources_with_tag = ['trails and paths', 'historic roads', 'curbs', 'road edges', 'greenways', 'pay parking stations', 'road surface', 'road row requirements downtown', 'barriers']
    # datasources_with_tag = ['railway crossings']


    # TODO: RUN THIS TO CREATE GOLD STANDARD
    # recommend_labels(dataset_metadata_set, metadata_set, schema_set, datasets_path, datasources_with_tag)
    # exit(0)


    # enrich_homonyms_test()
    # dataset_name = 'parks'
    # topic = 'parks'
    # desc = "the city of surrey is committed to protecting and enhancing natural and environmentally sensitive areas from harmful development. policies and regulations with respect to environmentally sensitive development are contained in city plans and by-laws as well as in provincial and federal acts."
    # notes = "this dataset includes parks in surrey. for more information please visit the [surrey"
    # other_topics = ['activities', 'environment', 'green', 'health', 'nature', 'walk', 'youth'] # ,'parks'
    #
    # # topic = 'activities'

    enrich_topics_full_run(datasources_with_tag, dir)

