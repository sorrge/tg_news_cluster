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
from collections import Counter
from typing import List
import re

from tqdm.autonotebook import tqdm

import numpy as np
from sklearn.model_selection import KFold

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from tensorflow import keras

import news
from language import FileInfo

layers = keras.layers
models = keras.models
from tensorflow.keras.optimizers import Adam

# This code was tested with TensorFlow v1.8
print("You have TensorFlow version", tf.__version__)


def fasttext_predict_category(train_texts, train_categories, predict_texts):
    with open("data/category_train.txt", "w") as f_tr:
        for text, cat in zip(train_texts, train_categories):
            f_tr.write(f"__label__{cat} {text}\n")

    with open("data/category_test.txt", "w") as f_ts:
        for text in predict_texts:
            f_ts.write(f"{text}\n")

    res = os.system("data/fasttext supervised -input data/category_train.txt -output data/category_model -epoch 20 "
                    "-thread 15 -minCount 1 -wordNgrams 1 -dim 20")

    assert res == 0
    res = os.system("data/fasttext predict data/category_model.bin data/category_test.txt > data/test_labels")
    assert res == 0

    with open("data/test_labels") as f:
        test_categories = [l.split("__")[-1][:-1] for l in f.readlines()]

    return test_categories


category_words_rus = {
    "society": ["подделок", "подделк", "поддельн", "раскрыти", "преступн", "уголовн", "подложны", "налог", "тарифа",
                "скидк", "госсовет", "льгот", "заседан",
                "капремонт", "травм", "наезд", "проезжую", "несовершеннолетн", "кювет", "дтп", "спасательн", "мчс",
                "наводнени", "затаплива",
                "чрезвычайн", "государств", "мошенничеств", "растрат", " хищени", "фигурант", "приговор", "воруют",
                "оперативн", "пострада", "ровд", "дежурн",
                "обвиня", "закон", "расслед", "кримин", "судья", "загорелся", "эваку", "тушени", "локализован",
                "возгор", "пожар", "арест", "подозреваем", "террор",
                "крушен", "катастроф", "причастн", "группировк", "коррупц", "задержа", "детектив", "злоупотреб",
                "сепаратист", "перевернул", "опрокинулся",
                "ушиб", "обочин", "перелом", "экстренн", "госпитализ", "оккупированны", "выявлен", "нецелев", "патрул",
                "инспек", "следствен",
                "виновны", "труп", "убий", "смерт", "экспертиз", "парти", "парторг", "бюллет", "возгла", "делегат",
                "полит", "голосов",
                "кандидат", "выбран", "председ", "горсовет", "регионал", "подписан", "саммит", "урегулирован",
                "перегово", "пропал", "поиски", "волонтёр",
                " танк", " нато", "страны", "стрельб", "погиб", "огнестрел", "конфликт", "происшеств", "правоохр",
                "инцидент", "полиц", "военно", "тариф",
                "посольств", "дипломат", "подпольн", "чиновни", "администрация", "фонд", "аварий", "пособия", "выплат",
                "президент", "выборах", "губернатор", "международн"],
    "economy": ["доход", "нефтегаз", "баррел", "среднеконтрактных", "прибыток", "рублев", "бюджет", "профицит",
                "пошлин", "поставщик", "подрядчик",
                "аукцион", "поставк", "подорожа", "производител", "топлив", "стоимост", "холдинг", "сделк", " акци",
                "аналитик", "оценк", "компани", "котировк",
                "рынок", "долей", "миноритар", "инвест", "рынке", "продукци", "предпринимател", "бирж", "фондов"],
    "technology": ["паркетник", "кроссовер", "дизайн", "рестайлинг", "двигатель", "тачскрин", "легков", "рендеринг",
                   "гаджет", "бампер", "интерфейс", "автомобил",
                   "мотор", "смартфон"],
    "sports": ["чемпион", "спарринг", "поединк", "турнир", "финал", "матч", "сетах", "побед", "клуб", "премьер-лиг",
               "команд", "тренер", "игра", "арен",
               "спорт", "трибун", "футбол", " лиги ", "обыграл", " гран ", "триумф", "гонк", "гонщи", "регби"],
    "entertainment": [" шоу", "поклонник", "исполнительниц", "артист", "премьер", "анонс", "мульт", "сериал", "сюжет",
                      "студи", "худож", "рисован",
                      "телеведущ", "съёмк", "съёмоч", "фильм", "ролям", "снимал", "постановщик", "прокат", "боевик",
                      "режисс", "сиквел", "франшиз", "релиз",
                      "трейлер", "ролик", "комикс", "картин", "солист", "продюсер", "балет", "видеоиг", "фанат", "певц",
                      "слав", "популярн", "шоу-биз", "интервью",
                      "звездн", "актрис", "голливуд", "скриншот", "консол"],
    "science": ["патологи", "медик", "терапевт", "хроническ", "препарат", "клинически", "испытания", "учены", "опухол",
                "разрабаты", "онкол", "вакцин",
                "заболевани", "биолог", "вирус", "исследова", "учёные", "заболева", "фактор", "планет", "научн", "наук",
                "пересадк", "археолог", "трансплант"],
    "other": ["православн", "священн", "евангел", "прихожан", "засеян", "памятник", "реконструк", "пресс-служб",
              "неподтвержденн", "работы по",
              "туропер", "турагент", "турист", "туризм", "фестивал", "театрализованн", "снялась", "красотка",
              "состоится", "построен", "сообщил", "культур",
              "церк", "монах", "сообща", "праздни", "новост", "по данным", "по информации"],
    "junk": ["гид", "рекомендуют", "стоит ли покупать", "потепле", "градус", "похолода", "ru_auto", " ну ", "приходите",
             "хайп", " я ", " вот ", "подборка",
             "инструкция", " нам ", "гороскоп", "приметы", "обзор", " осадк", "облачност", "°С", "Атмосферное давление",
             "Голосуйте за"]
}


category_words_en = {
    "society": [" law ", "illegal", "human traffick", "whistleblower", "victim", "protest", "minister", "death toll",
                "government", "Impeach", "licence", "legal", "governor", "gunmen", "candidate", "corruption",
                "investigation", "crash", "laundering", "illicit", "detained", "allegations", "public", "troops",
                "terror", "Court", "precedent", "attorney", "Stolen", "Police", "State", "Directorate", "policy",
                "Ambassador", "Congress", "vote", "politic", "Parliament", "hostage", "insurgent", "President",
                "slave", "forced", "abuse", "Opposition", "election", "royal", "children", "policies", "Senator",
                "tortur", "kill", "attack", "lawsuit", "federal", "Supreme", "prince", "nationali", "suicid",
                "incident", " traged", "arrest", "criminal", "firearm", "assault", "officer"],
    "economy": ["retailer", "startup", "compan", "financ", "futures", "oil", "billion", "contract", "Bank", "market",
                "revenue", "sales", "trading", "business", "purchases", "invest", "energy", "export", "STRIKE",
                "trade union", "Trust", "merger"],
    "technology": ["streaming", "Android", "device", " app ", "connection", "Build", "issues", "update", "download",
                   "feature", "permission", "Artificial Intelligence", "technology", "algorithm", "laptop", "tablet",
                   "starter", "Computer", "fighter jet", "aircraft", "software", "processor", " specs ", "performance",
                   "application", " Auto ", "vehicle", " SUV ", "display", "memory", "GB", " apps", "setup",
                   "security", "crypto", "gadget", "database", "vulnerabilit", " car ", " cars ", "plug", "battery",
                   "motor"],
    "sports": ["cornerback", "golf", "Basketball", "football", "semifinal", "club", "game against",
               "games", "champion", "winning", "tournament", " won ", "Racing", "finish line", "League", "player",
               "Cricket", "match", "qualifiers", "victory", "beats", "game-time", "opening game", "innings", "winners",
               "sport", "Marathon", "Athletics", "runner", "finishers", "touchdown", "opening", "score", "coach",
               "national", "standings", " Cup ", " racer"],
    "entertainment": ["console", "episode", "entertainment", "platformer", "release", "movie", "role", "singer",
                      "songs", "Music", "popularity", "expansion", "version", "the game", "online", "production",
                      "actor", "memoir", "book", "will come out", "Trailer", "stars", "actress", "fans",
                      "viewers", "instalment", "director", "film", "gala", " art", "show", "author", " RPG ", "PC",
                      "upcoming", "character", "television", "appearance", "architect", "announce", "multiplayer",
                      "maps", "video game", "CONTROLLER", "gaming", "virtual", "Cheating", "spawn", " quest "],
    "science": ["health", "research", "University", "scientist", "science", "evolution", "study", "medical",
                "are defined as"],
    "other": ["deer", "flooding", "sets record", "Supermodel", "report", "cathedral", "architect", "hits new heights",
              "worrying growth", "environment", "climate", "prepare for", "freez", "weather", "museum", "recover",
              "Animal", "census", "according to", "bite", "Unusual", "local news", "bird", "pollution", "news release",
              "traffic", "repairs", "touris", "Design"],
    "junk": ["you're", re.compile(r"top \w* headlines", re.IGNORECASE),
             re.compile(r"(best|grab) \w* deals", re.IGNORECASE), "on sale",
             "You Need To Know", " I ", "our pick", "links of the day", "recipe",
             "You’ll", "Spoilers", "sorry", " we ", "Your", "promo code", "webinar", "you will", "Best time to",
             " top ", "aesthetic", "meditation", "making money", "?", "opinion", "comment", "We bring you",
             "highlights", "rumors", "in your face", "review", "we’ve", "Indeed", "Here’s why", "all sorts of",
             "strange things", "As in,", "will be", "obituar", "Date of death", "and more", "news this week",
             "The secret", "how to", "Do you", "Are you", "your best", "If you", "passed away", "funeral"]
}


for words in category_words_rus.values():
    for i in range(len(words)):
        words[i] = words[i].lower()

for words in category_words_en.values():
    for i in range(len(words)):
        if isinstance(words[i], str):
            words[i] = words[i].lower()


def cat_counter(text, category_words):
    text = text.lower()
    ctr = Counter()
    for cat, words in category_words.items():
        for word in words:
            if isinstance(word, str):
                ctr[cat] += text.count(word)
            else:
                ctr[cat] += len(word.findall(text))

    return ctr


def kw_cat(text, category_words):
    if len(text) < 250:
        return "junk"

    ctr = cat_counter(text, category_words)
    if len(ctr) == 0:
        return ""

    cat, weight = ctr.most_common(1)[0]
    if weight == 0:
        return ""

    return cat


def keyword_categories(file_info: List[FileInfo], category_words):
    return [kw_cat(fi.text + " " + " ".join(news.simple_tokenize(fi.url)), category_words)
            for fi in tqdm(file_info, desc="assigning keyword-based labels")]


def reassign_labels_one_dataset_ft(train_texts, train_categories):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    ft_labels = [""] * len(train_texts)
    categories_assigned = [i for i, cat in enumerate(train_categories) if cat != ""]
    for train_index, test_index in tqdm(kf.split(train_texts), total=n_splits, desc="reassigning labels [ft]"):
        train_index = train_index[np.in1d(train_index, categories_assigned)]
        train_x = [train_texts[i] for i in train_index]
        train_y = [train_categories[i] for i in train_index]
        test_x = [train_texts[i] for i in test_index]
        ft_test_labels = fasttext_predict_category(train_x, train_y, test_x)
        for i, j in enumerate(test_index):
            ft_labels[j] = ft_test_labels[i]

    return ft_labels
