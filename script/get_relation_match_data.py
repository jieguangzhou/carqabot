import pandas as pd
import os
from tqdm import tqdm
import argparse
import itertools
import sys

sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('--sample_qa_path',
                    type=str,
                    required=True)

parser.add_argument('--complex_qa_path',
                    type=str,
                    required=True)

parser.add_argument('--save_path',
                    type=str,
                    required=True)

args = parser.parse_args()



def deal(sample_qa_path, complex_qa_path, save_path):
    df_sample = pd.read_excel(sample_qa_path)
    df_sample['qa_type'] = df_sample['predicate']
    df_sample['type'] = 'simple'
    df_complex = pd.read_excel(complex_qa_path)
    df_complex['type'] = 'complex'
    df = pd.concat([df_sample, df_complex])
    datas = []
    print(set(df_complex['qa_type']))
    for qa_type, sub_df in tqdm(df.groupby('qa_type')):

        sub_df = [dict(data) for _, data in sub_df.iterrows()]
        data_type = sub_df[0]['type']
        if data_type == 'simple':
            sub_df = sub_df[:5]

        neg_df = df[df['qa_type'] != qa_type].sample(len(sub_df) * 3).copy()
        neg_df = [dict(data) for _, data in neg_df.iterrows()]

        all_df = sub_df + neg_df
        iter = itertools.permutations(all_df, 2)
        for i, j in iter:
            text_a = i['question']
            text_b = j['question']
            label = 'Yes' if i['qa_type'] == j['qa_type'] else 'No'
            datas.append({'text_a': text_a, 'text_b': text_b, 'label': label})



    new_df = pd.DataFrame(datas)
    df_train, df_test = spilt_dataset(new_df)

    df_train.to_excel(os.path.join(save_path, 'train.xlsx'),
                    index=False,
                    columns=['text_a', 'text_b', 'label'])

    df_test.to_excel(os.path.join(save_path, 'test.xlsx'),
                    index=False,
                    columns=['text_a', 'text_b', 'label'])


def spilt_dataset(df, test_ratio=0.1):
    df = df.sample(frac=1.0)
    cut_idx = max(int(round(test_ratio * df.shape[0])), 1)
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    return df_train, df_test

deal(
    sample_qa_path=args.sample_qa_path,
    complex_qa_path=args.complex_qa_path,
    save_path=args.save_path,
)
