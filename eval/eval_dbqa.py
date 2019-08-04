import os
import sys
from collections import defaultdict
import pandas as pd

base_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.dirname(__file__))
sys.path.append(base_path)
from bleu_metric.bleu import Bleu
from rouge_metric.rouge import Rouge


def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
        "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['Rouge-L'] = rouge_score
    return scores


import json


def normalize(s):
    """
    Normalize strings to space joined chars.

    Args:
        s: a list of strings.

    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(' '.join(tokens))
    return normalized


temp_datas = {}
prediction_file = os.path.join(base_path, 'carbot_data/model/dbqa/predictions.json')
test_data_path = os.path.join(base_path, 'data/train/dbqa/test.json')

pred_answers = json.load(open(prediction_file, 'r'))
ref_answers = {}
test_data = json.load(open(test_data_path, 'r'))['data']
for data in test_data:
    context = data['paragraphs'][0]['context']
    d = data['paragraphs'][0]['qas'][0]
    id_ = str(d['id'])
    answer = d['answers'][0]['text']
    ref_answers[id_] = answer
    temp_datas[id_] = {
        'id': int(id_),
        'answer': answer,
        'doc': context,
        'question': d['question']
    }

pred_dict = {}
ref_dict = {}
datas = []
for key in pred_answers.keys():
    if key in ref_answers:
        pred_dict[key] = normalize([pred_answers[key]])
        ref_dict[key] = normalize([ref_answers[key]])
        temp_data = temp_datas[key]
        temp_data['pred'] = pred_answers[key]
        datas.append(temp_data)

result = compute_bleu_rouge(pred_dict, ref_dict)
result = [{'metrics': name, 'value': round(100 * score, 2)} for name, score in result.items()]

df = pd.DataFrame(datas, columns=['id', 'doc', 'question', 'answer', 'pred'])
df_metrics = pd.DataFrame(result, columns=['metrics', 'value']).sort_values('metrics')

writer = pd.ExcelWriter(os.path.join(base_path, 'report/dbqa.xlsx'))
df.to_excel(writer, sheet_name='result', index=False)
df_metrics.to_excel(writer, sheet_name='metrics', index=False)
writer.close()
