import json
import argparse

from bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=1)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"

def run(bert_predictions_path, bert_input_path):
    with open(bert_predictions_path) as f:
        preds = json.load(f)

    with open(bert_input_path) as f:
        devset = json.load(f)

    for key in preds.keys():
        if type(preds[key]) is not list:
            preds[key] = [preds[key]]

    answers = {}
    for i in range(len(devset['data'])):
        text = devset['data'][i]['paragraphs'][0]['qas'][0]['answers'][0]['text']
        id_num = devset['data'][i]['paragraphs'][0]['qas'][0]['id']
        answers[str(id_num)] = [text]

    def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
        """
        Compute bleu and rouge scores.
        """
        assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(
                set(ref_dict.keys()) - set(pred_dict.keys()))
        scores = {}
        bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
        for i, bleu_score in enumerate(bleu_scores):
            bleu_score *= 100
            scores['Bleu-%d' % (i + 1)] = bleu_score
        return scores

    scores = compute_bleu_rouge(preds, answers)

    for score in scores.keys():
        print('{}: {}'.format(score, scores[score]))


parser = argparse.ArgumentParser(
    description='Run BLEU scoring on the BERT outputs')

parser.add_argument('answers_json', action="store")
parser.add_argument('prediction_json', action="store")

args = parser.parse_args()

run(args.prediction_json, args.answers_json)