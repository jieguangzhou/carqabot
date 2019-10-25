import requests
import pandas as pd
from tqdm import tqdm

api = 'http://localhost:10000/chat?&question={}'


data_path = '../report/测试数据.xlsx'

df = pd.read_excel(data_path)
datas = []
for _, row_data in tqdm(df.iterrows()):
   response_data = requests.get(api.format(row_data['text'])).json()
   result = response_data['result']
   row_data['result'] = result
   datas.append(row_data)

result_df = pd.DataFrame(datas, columns=['id', 'text', 'result'])
result_df.to_excel(data_path.replace('.xlsx', '_result.xlsx'), index=False)

