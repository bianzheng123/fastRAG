import numpy as np
import os
import sys
import re
import string
import json

'''
used to count the exact match
'''


def read_exact_match_gnd(dataset: str, username: str):
    raw_data_path = f'/home/{username}/Dataset/vector-set-similarity-search/RawData'
    gnd_jsonl_filename = os.path.join(raw_data_path, f'{dataset}/document/queries_short_answer.gnd.jsonl')
    end2end_gnd_m = {}
    with open(gnd_jsonl_filename, 'r') as f:
        for line in f:
            query_gnd_json = json.loads(line)
            queryID = int(query_gnd_json['query_id'])
            passageID_l = query_gnd_json['answers']
            end2end_gnd_m[queryID] = passageID_l
    return end2end_gnd_m


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def count_exact_match(answer_m: dict, exact_match_gnd_m: dict):
    em_l = []
    for queryID in answer_m.keys():
        assert queryID in exact_match_gnd_m.keys()
        answer_string_l = [normalize_answer(str) for str in answer_m[queryID]]
        gnd_str_l = [normalize_answer(str) for str in exact_match_gnd_m[queryID]]
        is_correct = False
        for answer_str in answer_string_l:
            is_correct = True if answer_str in gnd_str_l else False
            if is_correct:
                break
        em = 1 if is_correct else 0
        em_l.append(em)
    assert len(em_l) == len(answer_m.keys()) and len(em_l) <= len(exact_match_gnd_m.keys())
    return em_l


'''
used to count the end2end and vector set search accuracy
'''


def compute_exact_match(result_l: list, queryID_l: list, queryID2gnd_m: dict):
    assert len(result_l) == len(queryID_l)
    answer_m = {}
    for queryID, result in zip(queryID_l, result_l):
        answer_m[queryID] = result

    exact_match_l = count_exact_match(answer_m=answer_m, exact_match_gnd_m=queryID2gnd_m)
    recall_performance = {
        'exact_match_p5': '{:.3f}'.format(np.percentile(exact_match_l, 5)),
        'exact_match_p50': '{:.3f}'.format(np.percentile(exact_match_l, 50)),
        'exact_match_p95': '{:.3f}'.format(np.percentile(exact_match_l, 95)),
        'exact_match_max': '{:.3f}'.format(np.percentile(exact_match_l, 100)),
        'exact_match_mean': '{:.3f}'.format(np.average(exact_match_l)),
    }

    return recall_performance, exact_match_l


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-500-gnd'
    topk_l = [10]
    method_name = 'dessert'
    # first is the fixed name suffix, second is the variable suffix
    method_config_m = {
        'dessert': {
            'build_index': {
                'n_table': 128,
            },
            'retrieval': [
                {'initial_filter_k': 32, 'remove_centroid_dupes': True, "nprobe_query": 8},
                # {'initial_filter_k': 128, 'remove_centroid_dupes': True, "nprobe_query": 8},
                # {'initial_filter_k': 512, 'remove_centroid_dupes': True, "nprobe_query": 8}
            ]
        },
    }
    dessert_build_index_suffix = f'n_table_{method_config_m["dessert"]["build_index"]["n_table"]}'
    answer_suffix_l = [[f'initial_filter_k_{32}-nprobe_query_{8}-remove_centroid_dupes_{True}',
                        # f'initial_filter_k_{128}-nprobe_query_{8}-remove_centroid_dupes_{True}',
                        # f'initial_filter_k_{512}-nprobe_query_{8}-remove_centroid_dupes_{True}',
                        ]]
    answer_config_l = [[{'initial_filter_k': 32, 'nprobe_query': 8,
                         'remove_centroid_dupes': True
                         }
                        #    , {'initial_filter_k': 128, 'nprobe_query': 8,
                        #       'remove_centroid_dupes': True
                        #       },
                        # {'initial_filter_k': 512, 'nprobe_query': 8,
                        #  'remove_centroid_dupes': True
                        #  }
                        ]]
