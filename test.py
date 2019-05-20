def test_qa():
    from nlu.qa import QA
    dictionary_dir = 'data/dictionary'
    kg_path = 'data/kg/car.ttl'
    ner_model_path = 'out/ner'
    relation_classifier_model_path = 'out/kbqapc'
    qa = QA(dictionary_dir, kg_path, ner_model_path, relation_classifier_model_path)
    sentence = '奔驰GLA(进口)什么时候上市啊'
    result = qa.predict(sentence)
    print(result)


test_qa()

# from kbqa.common.dictionary import Dictionary
#
# path = 'data/dictionary/Train.txt'
#
# matcher = Dictionary(path)
# sentence = '思域和奔驰S级别哪个好'
# print(matcher.match('思域'))
# print(matcher.match('奔驰S级'))
