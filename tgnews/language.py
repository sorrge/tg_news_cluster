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
import re
import subprocess
import json
from collections import Counter, namedtuple
from typing import List

import libs.cpp_stuff as cpp

site_languages_file = "data/site_languages.json"

extract_patterns = {"site": re.compile(r'^<meta property="og:site_name" content="(.*)"/>$'),
                    "url": re.compile(r'^<meta property="og:url" content="(.*)"/>$'),
                    "title": re.compile(r'^<meta property="og:title" content="(.*)"/>$'),
                    "description": re.compile(r'^<meta property="og:description" content="(.*)"/>$'),
                    "time": re.compile(r'^<meta property="article:published_time" content="(.*)"/>$'),
                    "address": re.compile(r'^<address><time datetime.*</time> by <a rel="author">(.*)</a></address>$')}

text_pattern = re.compile(r"^<(h\d|p|b|i|blockquote|sub)>(.*)</.*>$")
tag = re.compile(r"<[^>]+>")
drop_pattern = re.compile(r'Use the code "', re.IGNORECASE)
stop_pattern = re.compile(r"^Latest Headlines$", re.IGNORECASE)


def cleanup_text(t):
    t = t.replace("&amp;", '&')
    t = t.replace("&nbsp;", ' ')
    t = t.replace("&lt;", '<')
    t = t.replace("&gt;", '>')
    t = t.replace("&quot;", '"')
    t = t.replace("<br/>", " ")
    return tag.sub("", t).replace(u'\xa0', u' ').strip()


def parse_file(lines, ignore_paragraph):
    info = {"title": "", "site": "", "url": "", "time":""}
    paragraphs = []
    for line in lines:
        line = line.strip()
        matched = False
        for name, pattern in extract_patterns.items():
            match = pattern.match(line)
            if match:
                info[name] = cleanup_text(match.group(1))
                matched = True
                break

        if not matched:
            match = text_pattern.match(line)
            if match:
                paragraph = cleanup_text(match.group(2))
                if stop_pattern.search(paragraph):
                    break

                if paragraph != "" and paragraph not in ignore_paragraph and not drop_pattern.search(paragraph):
                    paragraphs.append(paragraph)

    info["paragraphs"] = paragraphs
    return info


def read_file(html, ignore_paragraph):
    with open(html) as f:
        lines = list(f)

    return parse_file(lines, ignore_paragraph)


def text_similar(t1: str, t2: str):
    t1 = t1[:-5]
    return t2.startswith(t1)


def file_text(info: dict):
    res = info["title"] + " "

    if "description" in info and not any(text_similar(info["description"], pp) for pp in info["paragraphs"]):
        res += info["description"] + " "

    for p in info["paragraphs"]:
        if p != info["title"]:
            res += p + " "

    return res.strip()


def load_ignores():
    with open("data/ignore_paragraph") as f:
        return frozenset(l.strip() for l in f.readlines())


def detect_languages_in_folder(folder):
    files = glob.glob(os.path.join(folder, "*.html"))
    res = detect_languages(read_file_info(files))
    #for ld in res:
    #    ld["articles"] = [os.path.basename(file) for file in ld["articles"]]

    return res


def run_process(to_run, process_input):
    if not isinstance(process_input, str):
        process_input = "\n".join(process_input) + "\n"

    p = subprocess.Popen(to_run,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         encoding="utf8", text=True)

    out, _ = p.communicate(process_input)
    return out


def detect_languages(file_info: List["FileInfo"]):
    texts = [fi.text for fi in file_info]
    sites = [fi.site for fi in file_info]
    languages = cpp.detect_languages(texts, sites)
    articles_by_code = {}
    for lang_code, fi in zip(languages, file_info):
        if lang_code not in articles_by_code:
            articles_by_code[lang_code] = []

        articles_by_code[lang_code].append(fi.file)

    res = [{"lang_code": code, "articles": articles_by_code[code]} for code in articles_by_code]
    return res


def merge_folder(sample_folder):
    all_file = os.path.join(sample_folder, "all")
    if os.path.exists(all_file):
        return all_file

    for day_folder in glob.glob(os.path.join(sample_folder, "????????")):
        for batch_folder in glob.glob(os.path.join(day_folder, "??")):
            dump_texts(glob.glob(os.path.join(batch_folder, "*.html")), all_file)

    return all_file


def collect_site_languages(train_folder, site_languages):
    file_infos = read_dump(merge_folder(train_folder))
    langs = detect_languages(file_infos)
    file_to_fi = dict(zip([fi.file for fi in file_infos], file_infos))
    for la in langs:
        for file in la["articles"]:
            site = file_to_fi[file].site
            if site not in site_languages:
                site_languages[site] = Counter()

            site_languages[site][la["lang_code"]] += 1


FileInfo = namedtuple("FileInfo", ["text", "title", "site", "url", "file", "time"])


def make_file_info(file, file_data):
    return FileInfo(text=file_text(file_data), site=file_data["site"], file=file, url=file_data["url"],
                    title=file_data["title"], time=file_data["time"])


def read_file_info(files):
    ignore_paragraph = load_ignores()
    return [make_file_info(file, read_file(file, ignore_paragraph)) for file in files]


def dump_texts(files, out_name):
    with open(out_name, "a") as f:
        for file_info in read_file_info(files):
            f.write(f"{file_info.text}\t{file_info.site}\t{file_info.file}\t{file_info.url}\t{file_info.title}\t"
                    f"{file_info.time}\n")


def read_dump(dump_file):
    file_info = []
    with open(dump_file) as f:
        for line in f:
            fields = line[:-1].split("\t")
            file_info.append(FileInfo(text=fields[0], site=fields[1], file=fields[2], url=fields[3], title=fields[4],
                                      time=fields[5]))

    return file_info
