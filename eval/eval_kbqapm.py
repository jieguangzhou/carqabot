import os
import sys
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
base_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(base_path)
from model.text_classification import Predictor


predictor = Predictor(os.path.join(base_path, 'carbot_data/model/kbqapm'))


examples = predictor.processor.get_dev_examples(os.path.join(base_path, 'data/train/kbqa/predicate_match/'))
data_df = []
for example in tqdm(examples):
    pred, confidence = predictor.predict_text(example.text_a, example.text_b)
    text = example.text_a
    true = example.label
    data_df.append({
        'text':text,
        'true':true,
        'pred':pred,
        '是否正确': '是' if true == pred else '否'
    })


df = pd.DataFrame(data_df, columns=['text', 'true', 'pred', '是否正确'])
trues = df['true']
preds = df['pred']
labels = sorted(set(trues))
precision, recall, f1, num = metrics.precision_recall_fscore_support(y_true=trues, y_pred=preds, labels=labels, average=None)
precision_total, recall_total, f1_total, num_total = metrics.precision_recall_fscore_support(y_true=trues, y_pred=preds, labels=labels, average='micro')
precision = list(precision) + [precision_total]
recall = list(recall) + [recall_total]
f1 = list(f1) + [f1_total]
num = list(num) + [sum(num)]
labels.append('total')
metrics = {
    'precision':precision,
    'recall':recall,
    'f1':f1,
    '是否匹配':labels,
    '数量':num,
}
df_metrics = pd.DataFrame(metrics, columns=['是否匹配', '数量', 'precision', 'recall', 'f1'])
df_metrics['precision'] = df_metrics['precision'].apply(lambda x: round(100 * x, 2))
df_metrics['recall'] = df_metrics['recall'].apply(lambda x: round(100 * x, 2))
df_metrics['f1'] = df_metrics['f1'].apply(lambda x: round(100 * x, 2))
writer = pd.ExcelWriter(os.path.join(base_path, 'report/kbqapm.xlsx'))
df.to_excel(writer, sheet_name='result', index=False)
df_metrics.to_excel(writer, sheet_name='metrics', index=False)
writer.close()