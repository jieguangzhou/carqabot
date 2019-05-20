def test_qa():
    from nlu.qa import QA

    ner_model_path = 'out/ner'
    relation_classifier_model_path = 'out/kbqapc'
    qa = QA(ner_model_path, relation_classifier_model_path)
    sentence = '思域和奔驰S级别哪个好'
    result = qa.predict(sentence)
    print(result)


# test_qa()

from kbqa.common.dictionary import Dictionary

path = 'data/dictionary/Train.txt'

matcher = Dictionary(path)
sentence = '思域和奔驰S级别哪个好'
print(matcher.match('思域'))
print(matcher.match('奔驰S级'))
