from info import HATECHECK_F20, HATECHECK_F21, HATECHECK_F18, HATECHECK_F19, HATECHECK_F23
import numpy as np


def get_counter_quote_instances():
    examples = list()
    with open(HATECHECK_F20, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def get_counter_ref_instances():
    examples = list()
    with open(HATECHECK_F21, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def get_ident_neutral_instances():
    examples = list()
    with open(HATECHECK_F18, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def get_ident_pos_instances():
    examples = list()
    with open(HATECHECK_F19, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def get_target_indiv_nh_instances():
    examples = list()
    with open(HATECHECK_F23, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def get_functionality_instances(functionality: str):
    examples = list()
    with open(f"{functionality}.tsv", 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def evaluate_on_hatecheck_functionality(predictor, functionality):
    if functionality == 'f20':
        examples = get_counter_quote_instances()
    elif functionality == 'f21':
        examples = get_counter_ref_instances()
    elif functionality == 'f18':
        examples = get_ident_neutral_instances()
    elif functionality == 'f19':
        examples = get_ident_pos_instances()
    elif functionality == 'f23':
        examples = get_target_indiv_nh_instances()
    else:
        examples = get_functionality_instances(functionality)
        # print('The functionality is not supported')
    predictions = predictor([examples])
    labels = np.zeros((len(examples),))
    tp = (np.array(predictions) == np.array(labels)).sum()
    tpr = tp / len(predictions)
    print('TPR on HateCheck {}: '.format(functionality), tpr)


def evaluate_on_hatecheck(predictor):
    # for functionality in ['f20', 'f21', 'f18', 'f19', 'f23']:
        # evaluate_on_hatecheck_functionality(predictor, functionality)
    for functionality in ["counter_quote_nh", "counter_ref_nh", "derog_dehum_h", "derog_impl_h", "derog_neg_attrib_h", "derog_neg_emote_h", "ident_neutral_nh", "ident_pos_nh", "negate_neg_nh", "negate_pos_h", "phrase_opinion_h", "phrase_question_h", "profanity_h", "profanity_nh", "ref_subs_clause_h", "ref_subs_sent_h", "slur_h", "slur_homonym_nh", "slur_reclaimed_nh", "spell_char_del_h", "spell_char_swap_h", "spell_leet_h", "spell_space_add_h", "spell_space_del_h", "target_group_nh", "target_indiv_nh", "target_obj_nh", "threat_dir_h", "threat_norm_h"]:
        evaluate_on_hatecheck_functionality(predictor, functionality)

