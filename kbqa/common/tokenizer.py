from jieba import Tokenizer as BaseTokenizer, string_types, strdecode
import re

RE_NUM_POINT = re.compile('[\.\da-zA-Z\-]+')



class Tokenizer(BaseTokenizer):
    def suggest_freq(self, segment, tune=False):
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        """
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1
        if isinstance(segment, string_types):
            word = segment
            for seg in self.cut(word, HMM=False):
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        else:
            segment = tuple(map(strdecode, segment))
            word = ''.join(segment)
            for seg in segment:
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = min(int(freq * self.total), self.FREQ.get(word, 0))
        if tune:
            self.add_word(word, freq)
        return freq


    def lcut(self, sentence):
        special_words = RE_NUM_POINT.findall(sentence)
        delete_words = []
        for word in special_words:
            if word not in self.FREQ:
                self.suggest_freq(word, tune=True)
                delete_words.append(word)

        result = super(Tokenizer, self).lcut(sentence, HMM=False)
        return result



if __name__ == '__main__':
    tonkizer = Tokenizer()
    print(tonkizer.lcut('拓陆者 2018款 2.4L E3汽油两驱精英型国V 4G69S4M'))