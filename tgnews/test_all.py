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
import subprocess
import glob
import json
import time

from tqdm import tqdm


def run_tgnews(stage, folder, files):
    assert len(files) > 100
    start = time.process_time()
    p = subprocess.Popen(f"python tgnews.py {stage} {folder}",
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                         encoding="utf8", text=True)

    out, err = p.communicate()
    seconds_taken = time.process_time() - start

    assert p.returncode == 0
    res = json.loads(out)
    assert err == ""
    assert seconds_taken < len(files) / 1000 * 60 * 0.2
    return res


def test_languages(folder, files):
    res = run_tgnews("languages", folder, files)
    assert sum(ld["lang_code"] == "ru" for ld in res) == 1
    assert sum(ld["lang_code"] == "en" for ld in res) == 1
    assert sum(len(ld["articles"]) for ld in res if ld["lang_code"] in ["ru", "en"]) > 0.1 * len(files)
    for ld in res:
        assert all(a in files for a in ld["articles"])


def test_news(folder, files):
    res = run_tgnews("news", folder, files)
    assert "articles" in res
    assert len(res["articles"]) > 0.1 * len(files)
    assert all(a in files for a in res["articles"])


def test_categories(folder, files):
    res = run_tgnews("categories", folder, files)
    for cd in res:
        assert cd["category"] in ["society", "economy", "sports", "science", "other", "technology", "entertainment"]

    assert sum(len(cd["articles"]) for cd in res) > 0.1 * len(files)
    for cd in res:
        assert all(a in files for a in cd["articles"])


def test_threads(folder, files):
    res = run_tgnews("threads", folder, files)
    assert sum(len(td["articles"]) for td in res) > 0.1 * len(files)
    for td in res:
        assert "title" in td
        assert all(a in files for a in td["articles"])


def test_top(folder, files):
    res = run_tgnews("top", folder, files)
    assert res[0]["category"] == "all"
    all_articles = []
    for td in res[0]["threads"]:
        all_articles.extend(td["articles"])

    assert len(all_articles) > 0.1 * len(files)
    all_by_category = set()
    for cd in res[1:]:
        assert cd["category"] in ["society", "economy", "sports", "science", "other", "technology", "entertainment"]
        for td in cd["threads"]:
            assert all_by_category.isdisjoint(td["articles"])
            all_by_category.update(td["articles"])

    assert all_by_category == set(all_articles)


def test_all(folder):
    try:
        files = glob.glob(os.path.join(folder, "*.html"))
        files = frozenset(os.path.basename(file) for file in files)
        test_languages(folder, files)
        test_news(folder, files)
        test_categories(folder, files)
        test_threads(folder, files)
        test_top(folder, files)
    except BaseException:
        print(f"Error on folder {folder}")
        raise


folders = ["sample", "sample2", "sample3", "sample4"]
all_batch_folders = []
for folder in folders:
    for day_folder in glob.glob(os.path.join("data", folder, "????????")):
        all_batch_folders.extend(glob.glob(os.path.join(day_folder, "??")))


for folder in tqdm(all_batch_folders):
    test_all(folder)
