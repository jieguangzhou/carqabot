import os


class Path:
    work = os.path.dirname(__file__)
    data_path = os.path.join(work, 'carbot_data')

    dictionary = os.path.join(data_path, 'dictionary')
    kg = os.path.join(data_path, 'kg', 'car.ttl')
    ner_model = os.path.join(data_path, 'model', 'ner')
    relation_classifier_model = os.path.join(data_path, 'model', 'kbqapc')
    relation_match_model = os.path.join(data_path, 'model', 'kbqapm')

    sample_question = os.path.join(data_path, 'sample_question')
