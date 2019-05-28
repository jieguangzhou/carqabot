import os
from logging import getLogger
import pickle
import numpy as np
import torch

from pytorch_pretrained_bert import BertForTokenClassification, BertTokenizer

debug_message = False
logger = getLogger('bert_tc')
PROCESSOR_NAME = 'Processor.pkl'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class NerProcessor:
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, labels):
        self.labels = labels

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._read_ner_data(os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._read_ner_data(os.path.join(data_dir, "test.txt"))

    def _read_ner_data(self, input_file):
        examples = []
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
                        examples.append(InputExample(guid=len(examples), text=text, labels=tags))
                    text, tags = '', []
        return examples

    def convert_examples_to_features(self, examples, max_seq_length,
                                     tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label: i for i, label in enumerate(self.labels)}

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0 and debug_message:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))


            tokens = []
            for char in example.text:
                token = tokenizer.tokenize(char)
                tokens.append(token[0] if token else '')
            label_ids = example.labels or []
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
                label_ids = label_ids[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            label_ids = ['O'] + label_ids + ['O']

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            label_ids += ['O'] * (max_seq_length - len(label_ids))

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            label_ids = [label_map.get(tag) for tag in label_ids]

            if ex_index < 5 and debug_message:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %s)" % (example.labels, label_ids))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
        return features


    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, 'rb'))


class Predictor:
    def __init__(self, dir_path, max_seq_length=30):
        self.max_seq_length = max_seq_length
        self.processor = NerProcessor.load(os.path.join(dir_path, PROCESSOR_NAME))
        self.tokenizer = BertTokenizer.from_pretrained(dir_path)
        self.classifier = BertForTokenClassification.from_pretrained(dir_path, len(self.processor.labels))
        self.classifier.eval()
        self.id2label = {i: label for i, label in enumerate(self.processor.labels)}

    def predict(self, InputExample):
        max_seq_length = len(InputExample.text) + 2
        features = self.processor.convert_examples_to_features([InputExample], max_seq_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        with torch.no_grad():
            logits = self.classifier(input_ids, segment_ids, input_mask, labels=None)
            preds = torch.softmax(logits[0], dim=1).detach().cpu().numpy()[1:-1]
        label_ids = np.argmax(preds, axis=1)
        tags = [self.id2label[label_id] for label_id in label_ids]
        confidences = [pred[label_id]  for pred, label_id in zip(preds, label_ids)]
        return tags, confidences


    def predict_text(self, text):
        example = InputExample(guid=None, text=text)
        tags, confidences = self.predict(example)
        entities = []
        entity_indexs = []
        for index, (tag, pro) in enumerate(zip(tags, confidences)):
            if tag == 'E' or tag == 'S':
                entity_indexs.append(index)
                if entity_indexs:
                    start, end = entity_indexs[0], entity_indexs[-1]
                    pros = confidences[start:end+1]
                    entities.append({'start':start,
                                     'end':end,
                                     'word':text[start:end+1],
                                     'confidence':round(sum(pros) / len(pros), 2)
                                     })
                entity_indexs = []
            elif tag == 'I':
                entity_indexs.append(index)
            elif tag == 'B':
                entity_indexs = [index]
            else:
                entity_indexs = []

        return entities