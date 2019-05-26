import os
from logging import getLogger
import pickle
import numpy as np
import torch

from pytorch_pretrained_bert import BertConfig, BertForSequenceClassification, BertTokenizer

is_main = __name__ == '__main__'
logger = getLogger('bert_tc')
PROCESSOR_NAME = 'Processor.pkl'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class PredicateClassificationProcessor:
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, labels, id_key='id', text_a_key='text_a', text_b_key='text_b', label_key='label'):
        self.labels = labels
        self.id_key = id_key
        self.text_a_key = text_a_key
        self.text_b_key = text_b_key
        self.label_key = label_key

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._read_excel(os.path.join(data_dir, "train.xlsx"))

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._read_excel(os.path.join(data_dir, "test.xlsx"))

    def _read_excel(self, input_file):
        from pandas import read_excel
        """Reads a tab separated value file."""
        df = read_excel(input_file)
        examples = []
        for n, row_data in df.iterrows():
            row_data = dict(row_data)
            guid = row_data.get(self.id_key, n)
            text_a = row_data.get(self.text_a_key)
            text_b = row_data.get(self.text_b_key)
            label = row_data.get(self.label_key)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    def convert_examples_to_features(self, examples, max_seq_length,
                                     tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label: i for i, label in enumerate(self.labels)}

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0 and is_main:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

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
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map.get(example.label)

            if ex_index < 5 and is_main:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %s)" % (example.label, label_id))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
        return features

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, 'rb'))


class Predictor:
    def __init__(self, dir_path, max_seq_length=30):
        self.max_seq_length = max_seq_length
        self.processor = PredicateClassificationProcessor.load(os.path.join(dir_path, PROCESSOR_NAME))
        self.tokenizer = BertTokenizer.from_pretrained(dir_path)
        self.classifier = BertForSequenceClassification.from_pretrained(dir_path, len(self.processor.labels))
        self.classifier.eval()
        self.id2label = {i: label for i, label in enumerate(self.processor.labels)}

    def predict(self, InputExample):
        features = self.processor.convert_examples_to_features([InputExample], self.max_seq_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        with torch.no_grad():
            logits = self.classifier(input_ids, segment_ids, input_mask, labels=None)
            preds = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        label_id = np.argmax(preds)
        confidence = preds[label_id]
        label = self.id2label.get(label_id)
        return label, confidence


    def predict_text(self, text):
        example = InputExample(guid=None, text_a=text)
        return self.predict(example)