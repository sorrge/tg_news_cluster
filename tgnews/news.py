#    Copyright 2020 sorrge
#
#    This file is part of tg_news_cluster.
#
#    tg_news_cluster is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    tg_news_cluster is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with tg_news_cluster.  If not, see <https://www.gnu.org/licenses/>.
import math
import os
import re
from collections import Counter
import glob

from language import detect_languages, read_file_info
import libs.cpp_stuff as cpp

dictionary_ru = "data/dictionary_ru.tsv"
categories_ru = "data/categories_ru"
category_model_ru = "data/category_model_ru.json"
dictionary_en = "data/dictionary_en.tsv"
categories_en = "data/categories_en"
category_model_en = "data/category_model_en.json"


def run_nn(file_info, language, model, dictionary):
    cpp.load_nn_model(model)
    normalizer = TextNormalizer(language)
    normalizer.load(dictionary)

    word_idx_in_texts = []
    for normalized_text in normalizer.normalize_texts(file_info):
        word_idx_in_texts.append([normalizer.word_idx[t]
                                  for t in normalized_text.split(" ") if t in normalizer.word_idx])

    coefs = cpp.run_nn(word_idx_in_texts, model)
    return coefs


def classify_ru(file_info):
    with open(categories_ru) as f:
        categories = []
        for line in f:
            categories.append(line.strip())

    labels = []
    for coefs in run_nn(file_info, "ru", category_model_ru, dictionary_ru):
        max_coef = max(enumerate(coefs), key=lambda x: x[1])[0]
        labels.append(categories[max_coef])

    for i, fi in enumerate(file_info):
        if len(fi.text) < 250:
            labels[i] = "junk"

    return labels


def classify_en(file_info):
    with open(categories_en) as f:
        categories = []
        for line in f:
            categories.append(line.strip())

    labels = []
    for coefs in run_nn(file_info, "en", category_model_en, dictionary_en):
        max_coef = max(enumerate(coefs), key=lambda x: x[1])[0]
        labels.append(categories[max_coef])

    for i, fi in enumerate(file_info):
        if len(fi.text) < 250:
            labels[i] = "junk"

    return labels


def classify_news_in_folder(folder):
    files = glob.glob(os.path.join(folder, "*.html"))
    file_info = read_file_info(files)
    res = classify_news(file_info)
    for ld in res:
        ld["articles"] = [os.path.basename(file) for file in ld["articles"]]

    return res


def classify_news(file_info):
    lang_data = detect_languages(file_info)
    articles_by_category = {}
    for ld in lang_data:
        if ld["lang_code"] == "ru":
            ru_files = frozenset(ld["articles"])
            ru_file_info = [fi for fi in file_info if fi.file in ru_files]
            categories = classify_ru(ru_file_info)
            for cat, fi in zip(categories, ru_file_info):
                if cat not in articles_by_category:
                    articles_by_category[cat] = []

                articles_by_category[cat].append(fi.file)
        elif ld["lang_code"] == "en":
            ru_files = frozenset(ld["articles"])
            ru_file_info = [fi for fi in file_info if fi.file in ru_files]
            categories = classify_en(ru_file_info)
            for cat, fi in zip(categories, ru_file_info):
                if cat not in articles_by_category:
                    articles_by_category[cat] = []

                articles_by_category[cat].append(fi.file)

    res = [{"category": cat, "articles": arts} for cat, arts in articles_by_category.items()]
    return res


word_pattern = re.compile(r"!|\?|[a-zа-я]+")
word_number_pattern = re.compile(r"!|\?|[a-zа-я0-9]+")
not_word_pattern = re.compile(r"\W")


def simple_tokenize(text):
    return word_pattern.findall(text.lower())


def simple_tokenize_with_numbers(text):
    return word_number_pattern.findall(text.lower())


class TextNormalizer:
    def __init__(self, language):
        self.idf = {}
        self.word_idx = {}
        self.language = language

    def load(self, file):
        with open(file) as f:
            for line in f:
                word, idx, idf = line.strip().split("\t")
                self.idf[word] = float(idf)
                self.word_idx[word] = int(idx)

    def train(self, file_info):
        word_stems = collect_stems(file_info, self.language)
        all_words = Counter()
        for fi in file_info:
            tokens = frozenset([word_stems[token] for token in simple_tokenize(fi.text)]) | \
                     frozenset(simple_tokenize(fi.url))

            all_words.update(tokens)

        max_count = len(file_info) // 5
        min_count = 10#len(file_info) * 0.0001
        for word in list(all_words):
            if all_words[word] <= min_count or all_words[word] >= max_count:
                del all_words[word]

        print(f"{len(all_words)} word stems")

        log_n = math.log(len(file_info))
        self.idf = {word: (log_n - math.log(1 + count)) for word, count in all_words.items()}

    def norm_for_fasttext(self, text):
        return [t for t in simple_tokenize(text) if t in self.idf]

    def tfidf_binary(self, text_words):
        res = [(word, self.idf[word]) for word in text_words if word in self.idf]
        res.sort(key=lambda x: x[1], reverse=True)
        return res

    def normalize_text(self, text, site, url, word_stems):
        words = [word_stems[token] for token in simple_tokenize(text)]
        words_sorted = self.tfidf_binary(frozenset(words))
        site_single_word = not_word_pattern.sub("", site.lower())
        tokens = [w for w in words if w in self.idf][:30] + [w for w, _ in words_sorted] + \
                 [site_single_word] + self.norm_for_fasttext(url)

        return tokens

    def normalize_texts(self, file_info):
        word_stems = collect_stems(file_info, self.language)
        train_texts = []
        for fi in file_info:
            train_texts.append(" ".join(self.normalize_text(fi.text, fi.site, fi.url, word_stems)))

        return train_texts


def stem(words, language):
    return cpp.stem_words(list(words), language == "en")


def collect_stems(file_info, language):
    all_words = set()
    for fi in file_info:
        all_words.update(simple_tokenize(fi.text))

    if language is None:
        return dict(zip(all_words, all_words))

    word_stems = dict(zip(all_words, stem(all_words, language)))
    return word_stems
