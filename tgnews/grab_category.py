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
from collections import Counter
import math
import re
import json
from typing import List

import requests

import pandas as pd
import umap
from tqdm.autonotebook import tqdm

import language
from language import FileInfo


def grab_page(link):
    response = requests.get(link)
    assert response.status_code == 200
    page = response.content.decode("utf-8")
    return page


def grab_lenta_ru_categories(file_info: List[language.FileInfo]):
    lenta_ru_files = frozenset(fi.file for fi in file_info if fi.site == "lenta.ru")
    print(f"{len(lenta_ru_files)} lenta.ru files")

    lr_cats = ["Главное", "Россия", "Мир", "Бывший СССР", "Экономика", "Силовые структуры", "Наука и техника",
               "Культура", "Спорт", "Интернет и СМИ", "Ценности",
               "Путешествия", "Из жизни", "Дом"]

    file_lenta_ru_categories = {}
    for fi in tqdm(file_info):
        if fi.file not in lenta_ru_files or fi.file in file_lenta_ru_categories:
            continue

        page = grab_page(fi.url.replace("https://lenta.ru/", "https://m.lenta.ru/"))
        for cat in lr_cats:
            if f"title: '{cat}'," in page:
                file_lenta_ru_categories[fi.file] = cat
                break

    lr_to_tg_cats = {"Россия": "society", "Мир": "society", "Бывший СССР": "society", "Экономика": "economy",
                     "Силовые структуры": "society",
                     "Наука и техника": "science", "Культура": "entertainment", "Спорт": "sports",
                     "Интернет и СМИ": "other", "Ценности": "other",
                     "Путешествия": "other", "Из жизни": "other", "Дом": "other"}

    lr_file_cats = {}
    for file, lr_cat in file_lenta_ru_categories.items():
        lr_file_cats[file] = lr_to_tg_cats[lr_cat]

    return lr_file_cats


def grab_nv_categories(file_info):
    cat_kw = {
        "junk": ["style", "experts", "health/nutrition", "blogs", "radio", "opinion", "lifehacks", "sex", "medicines",
                 "top-", "fitness"],
        "society": ["politics", "kyiv", "events", "countries"],
        "economy": ["market", "biz/tech", "economics", "finance"],
        "technology": ["gadgets", "it-industry", "auto"],
        "other": [],
        "science": ["health/happiness", "science", "medicine"],
        "sports": ["sport"],
        }

    nv_file_cats = {}
    for fi in file_info:
        if fi.site == "НВ":
            m = re.match(r"https://(.*)/([^/]*)\.html", fi.url)
            url_part = m.group(1)
            url_title = m.group(2)

            cat = ""
            if re.match(r"^(kak|kuda)", url_title):
                cat = "junk"
            else:
                for cc, kws in cat_kw.items():
                    if any(kw in url_part for kw in kws):
                        cat = cc
                        break

            if cat != "":
                nv_file_cats[fi.file] = cat

    return nv_file_cats


def grab_federalpress_categories(file_info):
    cat_kw = {
        "junk": ["gossip"],
        "society": ["society", "incidents", "policy"],
        "economy": ["economy", "industry", "finance", "company-news", "energetics", "business"],
        "technology": [],
        "other": ["ecology", "projects", "realty"],
        "science": [],
        "sports": [],
        }

    file_cats = {}
    for fi in file_info:
        if fi.site == "ФедералПресс":
            m = re.match(r"^https://fedpress.ru/news/\d+/([^/]+)/.*", fi.url)
            if not m:
                cat = "junk"
            else:
                url_part = m.group(1)

                cat = ""
                for cc, kws in cat_kw.items():
                    if any(kw in url_part for kw in kws):
                        cat = cc
                        break

            if cat != "":
                file_cats[fi.file] = cat

    return file_cats


def grab_korrnet_categories(file_info):
    cat_kw = {
        "junk": [],
        "society": ["politics", "events"],
        "economy": ["business"],
        "technology": ["motors", "gadgets"],
        "other": [],
        "entertainment": ["showbiz"],
        "science": ["tech"],
        "sports": ["sport"],
        }

    file_cats = {}
    for fi in file_info:
        if fi.site == "Корреспондент.net":
            m = re.match(r"^https://korrespondent.net/(.*)/[^/]*", fi.url)
            if not m:
                continue

            url_part = m.group(1)

            cat = ""
            for cc, kws in cat_kw.items():
                if any(kw in url_part for kw in kws):
                    cat = cc
                    break

            if cat != "":
                file_cats[fi.file] = cat

    return file_cats


def grab_allhockey_categories(file_info):
    file_cats = {}
    for fi in file_info:
        if fi.site == "Allhockey.ru":
            if "/article/" in fi.url:
                file_cats[fi.file] = "junk"
            else:
                file_cats[fi.file] = "sports"

    return file_cats


junk_url_pattern_ru = re.compile(r"(/|\d-)kak-|gramota|blog|-ili-|-chto-delat-|-stoit-li-|(/|\d-)pochemu|obzor")
junk_url_pattern_en = re.compile(r"ничего")


def junk_by_url(file_info, junk_url_pattern):
    file_cats = {}
    for fi in file_info:
        if junk_url_pattern.search(fi.url):
            file_cats[fi.file] = "junk"

    return file_cats


def grab_reuters_categories(file_info):
    cat_kw = {
        "junk": ['insight', 'special reports', 'special', 'reports'],
        "society": ['government / politics', 'crime', 'disasters / accidents', 'insurgency',
                    'diplomacy / foreign policy', 'society / social issues', 'international / national security',
                    'military conflicts', 'lawmaking', 'civil unrest', 'ground accidents / collisions',
                    'regulation', 'corruption', 'elections / voting', 'taxation',
                    'freedom of speech / censorship', 'human rights / civil rights', 'whistleblower',
                    'judicial process / court cases / court decisions', 'labour / personnel', 'people',
                    'fundamental rights / civil liberties', 'us government news', 'children / youth issues',
                    'attack', 'computer crime / hacking / cybercrime', 'conflicts / war / peace',
                    'asylum / immigration / refugees', 'aid relief / humanitarian agencies', 'military procurement',
                    'blast', 'fire', 'politics', 'corruption / bribery / embezzlement', 'violence',
                    'government finances', 'race relations / ethnic issues', 'protests',
                    'international agencies / treaty groups', 'collapse', 'drug trafficking / narcotics'],
        "economy": ['company news', 'trade', 'agricultural markets', 'energy (trbc)', 'mineral resources (trbc)',
                    'technology (trbc)', 'markets', 'stocks', 'oil and gas (trbc)', 'economic indicators',
                    'energy markets', 'healthcare (trbc)', 'airlines (trbc)', 'auto and truck manufacturers (trbc)',
                    'department stores (trbc)', 'financial fraud / securities fraud', 'financials (trbc)',
                    'crime / law / justice', 'loans', 'oil refineries', 'all retail',
                    'monetary / fiscal policy / policy makers', 'funds', 'monopolies / antitrust issues', 'banks',
                    'aerospace and defense (trbc)', 'tariffs', 'computer and electronics retailers (trbc)',
                    'insurance', 'bankruptcy / insolvency', 'fishing and farming (trbc)',
                    'telecommunications services (trbc)', 'crude oil', 'economic news (3rd party)',
                    'technology / media / telecoms', 'plantings', 'economy', 'transportation (trbc)',
                    'market reports', 'deals', 'equities markets', 'international trade', 'bank',
                    'central banks / central bank events', 'technology equipment (trbc)',
                    'mergers / acquisitions / takeovers', 'layoffs', 'initial public offerings'],
        "technology": [],
        "other": ['environment', 'general news', 'education', 'climate politics', 'pollution', 'poll',
                  'wildfires / forest fires', 'school', 'weather', 'religion / belief', 'firefighters', 'quake',
                  'earthquakes', 'precipitation', 'living / lifestyle', 'celebrities', 'christianity',
                  'personalities / people', 'weather markets / weather', 'human interest / brights / odd news',
                  'anniversary', 'nature / wildlife'],
        "entertainment": ['media and publishing (trbc)', 'television', 'arts / culture / entertainment', 'music',
                          'video games', 'entertainment production (trbc)'],
        "science": ['science', 'health', 'health / medicine', 'pharmaceuticals and medical research (trbc)',
                    'biotechnology and medical research (trbc)', 'dialysis', 'vaccines', 'clinical medicine',
                    'humanskin', 'space', 'space exploration'],
        "sports": ["sport", 'olympics', 'soccer', 'basketball', 'motor racing', 'football', 'icehockey', 'baseball',
                   'horse racing', 'athletics', 'cycling', 'races', 'cricket', 'tennis', 'golf'],
        }

    with open("data/reuters_kw.json") as f:
        reuters_kw = json.load(f)

    file_title = {}
    for fi in file_info:
        if fi.file in reuters_kw:
            file_title[fi.file] = fi.title

    file_categories = {}
    for file, keywords in reuters_kw.items():
        keywords = frozenset(keywords)
        cats = Counter()
        for cat, kws in cat_kw.items():
            for kw in kws:
                if kw in keywords:
                    cats[cat] += 1

        if len(cats) == 0:
            pass
        else:
            file_categories[file] = cats.most_common(1)[0][0]

    return file_categories


def grab_theguardian_categories(file_info: List[FileInfo]):
    cat_kw = {
        "junk": ["lifeandstyle", "fashion", "travel", "commentisfree", "cities", "world", "food", "money",
                 "crosswords", "global", "inequality", "theobserver", "membership", "where-you-shop-matters",
                 "live-victoriously"],
        "society": ["society", "politics", "law"],
        "economy": ["business"],
        "technology": ["technology"],
        "other": ["environment", "australia-news", "global-development",
                  "education", "weather"],
        "entertainment": ["film", "music", "artanddesign", "games", "tv-and-radio", "media", "books", "stage",
                          "culture"],
        "science": ["science"],
        "sports": ["sport", "football"],
        }

    file_cats = {}
    for fi in file_info:
        if fi.site == "the Guardian":
            m = re.match(r"^https://www.theguardian.com/([^/]+)/.*", fi.url)
            if not m:
                print("No url match:", fi.url)
                continue

            url_part = m.group(1)

            cat = ""
            for cc, kws in cat_kw.items():
                if any(kw in url_part for kw in kws):
                    cat = cc
                    break

            if cat != "":
                file_cats[fi.file] = cat
            else:
                #print("No KW match:", url_part)
                #print("\t", fi.url)
                pass

    return file_cats


def grab_mirror_categories(file_info: List[FileInfo]):
    cat_kw = {
        "junk": ["lifeandstyle", "fashion", "travel", "commentisfree", "cities", "world", "food", "money",
                 "crosswords", "global", "inequality", "theobserver", "membership", "where-you-shop-matters",
                 "live-victoriously", 'lifestyle', 'play', "3am", 'real-life-stories'],
        "society": ["society", "politics", "law"],
        "economy": ["business"],
        "technology": ["technology", 'tech'],
        "other": ["environment", "global-development", 'world-news',
                  "education", "weather", 'weird-news'],
        "entertainment": ["film", "music", "artanddesign", "games", "tv-and-radio", "media", "books", "stage",
                          "culture", 'tv-news', 'celebrity-news', 'tv'],
        "science": ["science"],
        "sports": ["sport", "football"],
        }

    file_cats = {}
    for fi in file_info:
        if fi.site == "mirror":
            m = re.match(r"^https://www.mirror.co.uk/(.+)/[^/]*", fi.url)
            if not m:
                print("No url match:", fi.url)
                continue

            url_part = m.group(1)
            kws_url = url_part.split("/")

            cat = ""
            for cc, kws in cat_kw.items():
                if any(kw in kws_url for kw in kws):
                    cat = cc
                    break

            if kws_url == ["news"]:
                cat = "other"

            if cat != "":
                file_cats[fi.file] = cat
            else:
                #print("No KW match:", kws_url)
                #print("\t", fi.url)
                pass

    return file_cats


def load_gt(folder):
    ground_truth = {}
    if os.path.exists(os.path.join(folder, "ground_truth")):
        with open(os.path.join(folder, "ground_truth")) as f:
            for line in f:
                file, label = line.strip().split("\t")
                assert label in ["society", "sports", "entertainment", "technology", "junk", "other", "economy",
                                 "science"]

                ground_truth[file] = label

    print(f"{len(ground_truth)} GT labels loaded from {folder}")
    return ground_truth


def save_gt(ground_truth, folder):
    with open(os.path.join(folder, "ground_truth"), "w") as f:
        for file, label in ground_truth.items():
            f.write(f"{file}\t{label}\n")

    print(f"{len(ground_truth)} GT labels saved to {folder}")


def join_gt(gts):
    ground_truth = {}
    for gt in gts:
        for file, label in gt.items():
            ground_truth[file] = label

    return ground_truth


def gt_to_linear(linear_categories, gt, file_info: List[FileInfo]):
    for i, fi in enumerate(file_info):
        if fi.file in gt:
            linear_categories[i] = gt[fi.file]
