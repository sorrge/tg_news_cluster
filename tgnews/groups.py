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
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
import glob
import os

import news
from language import FileInfo, detect_languages, read_file_info

import libs.cpp_stuff as cpp


counts_for_grouping_ru = "data/chunk_counts_ru.bin"
counts_for_grouping_en = "data/chunk_counts_en.bin"


def calc_groups(texts: List[Tuple[List[str], str, str, float]], counts_file, similarity_cutoff):
    cpp.load_idf(counts_file)
    groups = cpp.group_texts(texts, counts_file, similarity_cutoff)
    return groups


def calc_groups_within_categories(texts: List[Tuple[List[str], str, str, float]], counts_file, similarity_cutoff, cats):
    all_cats = frozenset(cats) - frozenset(["junk"])
    cat_idx = [[i for i, c in enumerate(cats) if c == cat] for cat in all_cats]
    groups = []
    for idx in cat_idx:
        cat_texts = [texts[i] for i in idx]
        cat_groups = calc_groups(cat_texts, counts_file, similarity_cutoff)
        groups.extend([[idx[i] for i in group] for group in cat_groups])

    groups.sort(key=lambda x: len(x), reverse=True)
    return groups


def parse_datetime(datetime_string):
    try:
        return datetime.fromisoformat(datetime_string).astimezone(timezone.utc)
    except ValueError:
        return (datetime.utcnow() - timedelta(days=365*100)).astimezone(timezone.utc)


def extract_text_for_grouping(fi: FileInfo) -> List[str]:
    return (news.simple_tokenize_with_numbers(fi.title.lower()) + [""]) * 2 + \
           news.simple_tokenize_with_numbers(fi.text.lower())[:500]


def extract_texts_for_grouping(file_info) -> List[List[str]]:
    return [extract_text_for_grouping(fi) for fi in file_info]


def extract_infos_for_grouping(file_info):
    texts = extract_texts_for_grouping(file_info)
    return [(text, fi.site, fi.file, parse_datetime(fi.time).timestamp()) for text, fi in zip(texts, file_info)]


def group_news_ru(file_info: List[FileInfo], similarity_cutoff):
    info_for_grouping = extract_infos_for_grouping(file_info)
    categories = news.classify_ru(file_info)
    groups = calc_groups_within_categories(info_for_grouping, counts_for_grouping_ru, similarity_cutoff, categories)
    res = []
    aggregated_idx = []
    for group in groups:
        group.sort(key=lambda idx: parse_datetime(file_info[idx].time), reverse=True)
        group_title = file_info[group[0]].title
        res.append((group_title, [file_info[idx].file for idx in group]))
        aggregated_idx.extend(group)

    aggregated_idx = frozenset(aggregated_idx)
    for i, fi in enumerate(file_info):
        if i not in aggregated_idx and categories[i] != "junk":
            res.append((fi.title, [fi.file]))

    return res


def group_news_en(file_info: List[FileInfo], similarity_cutoff):
    info_for_grouping = extract_infos_for_grouping(file_info)
    categories = news.classify_en(file_info)
    groups = calc_groups_within_categories(info_for_grouping, counts_for_grouping_en, similarity_cutoff, categories)
    res = []
    aggregated_idx = []
    for group in groups:
        group.sort(key=lambda idx: parse_datetime(file_info[idx].time), reverse=True)
        group_title = file_info[group[0]].title
        res.append((group_title, [file_info[idx].file for idx in group]))
        aggregated_idx.extend(group)

    aggregated_idx = frozenset(aggregated_idx)
    for i, fi in enumerate(file_info):
        if i not in aggregated_idx and categories[i] != "junk":
            res.append((fi.title, [fi.file]))

    return res


def group_news(file_info, similarity_cutoff):
    lang_data = detect_languages(file_info)
    article_groups = []
    for ld in lang_data:
        if ld["lang_code"] == "ru":
            ru_files = frozenset(ld["articles"])
            ru_file_info = [fi for fi in file_info if fi.file in ru_files]
            groups = group_news_ru(ru_file_info, similarity_cutoff)
            for group_title, files in groups:
                article_groups.append({"title": group_title, "articles": files})
        elif ld["lang_code"] == "en":
            en_files = frozenset(ld["articles"])
            en_file_info = [fi for fi in file_info if fi.file in en_files]
            groups = group_news_en(en_file_info, similarity_cutoff)
            for group_title, files in groups:
                article_groups.append({"title": group_title, "articles": files})

    return article_groups


def group_news_in_folder(folder, similarity_cutoff):
    files = glob.glob(os.path.join(folder, "*.html"))
    file_info = read_file_info(files)
    res = group_news(file_info, similarity_cutoff)
    for ld in res:
        ld["articles"] = [os.path.basename(file) for file in ld["articles"]]

    return res
