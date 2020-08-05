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
import os
import glob
from datetime import datetime, timezone


from language import detect_languages, read_file_info
from groups import calc_groups_within_categories, counts_for_grouping_ru, extract_texts_for_grouping, parse_datetime, \
    counts_for_grouping_en

from news import classify_ru, run_nn, classify_en

ranking_dictionary_ru = "data/ranking_dictionary_ru.tsv"
ranking_model_ru = "data/ranking_model_ru.json"

ranking_dictionary_en = "data/ranking_dictionary_en.tsv"
ranking_model_en = "data/ranking_model_en.json"


def calculate_thread_priority(articles_idx, predicted_popularity, times):
    average_popularity = sum(predicted_popularity[i] for i in articles_idx) / len(articles_idx)
    last_update = max(times[i] for i in articles_idx)
    age_days = (datetime.now(timezone.utc) - last_update).days
    return average_popularity + min(4.0, len(articles_idx) * 0.3) - age_days * 0.5


def rank_ru(file_info, similarity_cutoff):
    popularity_predictions_obj = run_nn(file_info, "ru", ranking_model_ru, ranking_dictionary_ru)
    texts = extract_texts_for_grouping(file_info)
    categories = classify_ru(file_info)
    groups = calc_groups_within_categories(texts, counts_for_grouping_ru, similarity_cutoff, categories)
    times = [parse_datetime(fi.time) for fi in file_info]
    popularity_predictions = [p[0] for p in popularity_predictions_obj]

    threads = []
    aggregated_idx = []
    for group in groups:
        group.sort(key=lambda idx: times[idx], reverse=True)
        thread_title = file_info[group[0]].title
        threads.append((thread_title, categories[group[0]], group,
                        calculate_thread_priority(group, popularity_predictions, times)))

        aggregated_idx.extend(group)

    aggregated_idx = frozenset(aggregated_idx)
    for i, fi in enumerate(file_info):
        if i not in aggregated_idx and categories[i] != "junk":
            threads.append((fi.title, categories[i], [i], calculate_thread_priority([i], popularity_predictions,
                                                                                    times)))

    threads.sort(key=lambda t: t[3], reverse=True)
    return threads


def rank_en(file_info, similarity_cutoff):
    popularity_predictions_obj = run_nn(file_info, "en", ranking_model_en, ranking_dictionary_en)

    texts = extract_texts_for_grouping(file_info)
    categories = classify_en(file_info)
    groups = calc_groups_within_categories(texts, counts_for_grouping_en, similarity_cutoff, categories)
    times = [parse_datetime(fi.time) for fi in file_info]
    popularity_predictions = [p[0] for p in popularity_predictions_obj]

    threads = []
    aggregated_idx = []
    for group in groups:
        group.sort(key=lambda idx: times[idx], reverse=True)
        thread_title = file_info[group[0]].title
        threads.append((thread_title, categories[group[0]], group,
                        calculate_thread_priority(group, popularity_predictions, times)))

        aggregated_idx.extend(group)

    aggregated_idx = frozenset(aggregated_idx)
    for i, fi in enumerate(file_info):
        if i not in aggregated_idx and categories[i] != "junk":
            threads.append((fi.title, categories[i], [i], calculate_thread_priority([i], popularity_predictions,
                                                                                    times)))

    threads.sort(key=lambda t: t[3], reverse=True)
    return threads


def rank_news(file_info, similarity_cutoff):
    lang_data = detect_languages(file_info)
    all_ranked_threads = []
    for ld in lang_data:
        if ld["lang_code"] == "ru":
            ru_files = frozenset(ld["articles"])
            ru_file_info = [fi for fi in file_info if fi.file in ru_files]
            ranked_threads = rank_ru(ru_file_info, similarity_cutoff)
            for (_, _, articles_idx, _) in ranked_threads:
                for i, ai in enumerate(articles_idx):
                    articles_idx[i] = ru_file_info[ai].file

            all_ranked_threads.extend(ranked_threads)
        elif ld["lang_code"] == "en":
            en_files = frozenset(ld["articles"])
            en_file_info = [fi for fi in file_info if fi.file in en_files]
            ranked_threads = rank_en(en_file_info, similarity_cutoff)
            for (_, _, articles_idx, _) in ranked_threads:
                for i, ai in enumerate(articles_idx):
                    articles_idx[i] = en_file_info[ai].file

            all_ranked_threads.extend(ranked_threads)

    all_ranked_threads.sort(key=lambda t: t[3], reverse=True)
    threads_res = {"all": []}
    for (thread_title, category, article_files, priority) in all_ranked_threads:
        if category not in threads_res:
            threads_res[category] = []

        threads_res[category].append({"title": thread_title, "articles": article_files})
        threads_res["all"].append({"title": thread_title, "category": category, "articles": article_files})

    threads_res = [{"category": cat, "threads": threads} for cat, threads in threads_res.items()]
    return threads_res


def rank_news_in_folder(folder, similarity_cutoff):
    files = glob.glob(os.path.join(folder, "*.html"))
    file_info = read_file_info(files)
    res = rank_news(file_info, similarity_cutoff)
    for cd in res:
        for td in cd["threads"]:
            td["articles"] = [os.path.basename(file) for file in td["articles"]]

    return res
