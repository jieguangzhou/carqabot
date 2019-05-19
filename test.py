from nlu.qa import QA

ner_model_path = 'out/ner'
relation_classifier_model_path = 'out/kbqapc'
qa = QA(ner_model_path, relation_classifier_model_path)
sentence = '08年车东风本田思域压缩比是多少'
result = qa.predict(sentence)
print(result)
