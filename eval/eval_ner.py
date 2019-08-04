import os
import sys
from collections import defaultdict
import pandas as pd

base_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(base_path)
from model.sequence_labeling import Predictor

predictor = Predictor(os.path.join(base_path, 'carbot_data/model/ner'))


def read_ner_data(input_file):
    datas = []
    text, tags = '', []
    with open(input_file, 'r') as r_f:
        for line in r_f:
            line = line.rstrip('\n')
            if line:
                char, tag = line.split('\t')
                text += char
                tags += tag
            else:
                if tags:
                    datas.append((text, tags))
                text, tags = '', []
    return datas


datas = read_ner_data(os.path.join(base_path, 'data/train/kbqa/ner/test.txt'))
all_true = set()
all_pred = set()
data_df = []
for index, (text, tags) in enumerate(datas):
    dd = {'text': text}
    result_true = predictor.concat_entity(text, tags)
    result_pred = predictor.predict_text(text)
    dd['true'] = []
    dd['pred'] = []
    for r in result_true:
        mark = '{}-{}-{}'.format(id, r['start'], r['end'])
        dd['true'].append('{}-{}-{}'.format(r['start'], r['end'], r['word']))
        all_true.add(mark)
    for r in result_pred:
        mark = '{}-{}-{}'.format(id, r['start'], r['end'])
        dd['pred'].append('{}-{}-{}'.format(r['start'], r['end'], r['word']))
        all_pred.add(mark)
    cuo = set(dd['true']) - set(dd['pred'])
    lou = set(dd['pred']) - set(dd['true'])
    dd['错的'] = cuo
    dd['漏的'] = lou
    data_df.append(dd)

true_pred = all_true & all_pred
precision = len(true_pred) / len(all_pred)
recall = len(true_pred) / len(all_true)
f1 = 2 * precision * recall / (precision + recall)
metrics = {
    'precision': precision,
    'recall': recall,
    'f1': f1,
    '识别出的正确实体数': len(true_pred),
    '识别出的实体数': len(all_pred),
    '样本的实体数': len(all_true),
    'tag': '汽车'
}
df = pd.DataFrame(data_df, columns=['text', 'true', 'pred', '错的', '漏的'])
df_metrics = pd.DataFrame([metrics], columns=['tag', '识别出的正确实体数', '识别出的实体数', '样本的实体数',
                                              'precision', 'recall', 'f1'])
df_metrics['precision'] = df_metrics['precision'].apply(lambda x: round(100 * x, 2))
df_metrics['recall'] = df_metrics['recall'].apply(lambda x: round(100 * x, 2))
df_metrics['f1'] = df_metrics['f1'].apply(lambda x: round(100 * x, 2))
writer = pd.ExcelWriter(os.path.join(base_path, 'report/ner.xlsx'))
df.to_excel(writer, sheet_name='result', index=False)
df_metrics.to_excel(writer, sheet_name='metrics', index=False)
writer.close()
