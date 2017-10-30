import os
import re

from nltk.tokenize import WordPunctTokenizer


class SentenceReader:
    def __init__(self, dir_name, tokenizer=None, encoding='utf-8'):
        self.dir_name = dir_name
        self.encoding = encoding
        if not tokenizer:
            tokenizer = WordPunctTokenizer()
        self.tokenizer = tokenizer

    def __iter__(self):
        for file_name in os.listdir(self.dir_name):
            print('reading file {}'.format(file_name))
            if file_name.startswith('.') or os.path.isdir(file_name):
                continue
            with open(os.path.join(self.dir_name, file_name), encoding=self.encoding) as fp:
                for line in fp:
                    yield [s.strip() for s in self.tokenizer.tokenize(line)] + ['\n']


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
