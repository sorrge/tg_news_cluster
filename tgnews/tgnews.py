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

import argparse
import json
import os

import language
import news
import groups
import ranking
import server
from libs import cpp_stuff as cpp


def console_dump_block(block, folder, ignore_paragraph):
    for key, value in block.items():
        if key == "articles":
            print(f"{len(value)} articles:")
            for article in value:
                file_text = language.file_text(language.read_file(os.path.join(folder, article), ignore_paragraph))
                print("\t" + file_text)
                print()

            print("\n\n")
        else:
            print(f"      *** {key}: {value} ***")
            print()


def console_dump(data, folder, ignore_paragraph):
    for block in data:
        console_dump_block(block, folder, ignore_paragraph)


def main():
    parser = argparse.ArgumentParser(description="Data Clustering Contest: Round 1 app",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("stage", type=str, choices=["languages", "news", "categories", "threads", "top", "server"],
                        help="stage")

    parser.add_argument("source_dir", type=str, help="path to the folder with HTML-files containing article texts or "
                                                     "port for the server")

    parser.add_argument("--console", required=False, action="store_true",
                        help="dump article texts to console")

    parser.add_argument("--similarity-cutoff", type=float, required=False, default=0.25,
                        help="similarity cutoff for grouping articles into threads. Larger values produce more "
                             "narrow-themed threads")

    args = parser.parse_args()
    with open(language.site_languages_file) as f:
        site_languages = json.load(f)

    cpp.load_sources_languages(site_languages)

    if args.stage == "languages":
        lang_data = language.detect_languages_in_folder(args.source_dir)
        lang_data.sort(key=lambda ld: ld["lang_code"] not in ["en", "ru"])
        for lang_part in lang_data:
            lang_part["articles"] = [os.path.basename(a) for a in lang_part["articles"]]

        if args.console:
            console_dump(lang_data, args.source_dir, language.load_ignores())
        else:
            print(json.dumps(lang_data, indent=2))
    elif args.stage == "news":
        classification_data = news.classify_news_in_folder(args.source_dir)
        news_articles = []
        for category_data in classification_data:
            if category_data["category"] != "junk":
                news_articles.extend(category_data["articles"])

        res = {"articles": news_articles}
        if args.console:
            console_dump_block(res, args.source_dir, language.load_ignores())
        else:
            print(json.dumps(res, indent=2))
    elif args.stage == "categories":
        classification_data = news.classify_news_in_folder(args.source_dir)
        res = [cd for cd in classification_data if cd["category"] != "junk"]
        if args.console:
            console_dump(res, args.source_dir, language.load_ignores())
        else:
            print(json.dumps(res, indent=2))
    elif args.stage == "threads":
        groups_data = groups.group_news_in_folder(args.source_dir, args.similarity_cutoff)
        if args.console:
            console_dump(groups_data, args.source_dir, language.load_ignores())
        else:
            print(json.dumps(groups_data, indent=2))
    elif args.stage == "top":
        threads_data = ranking.rank_news_in_folder(args.source_dir, args.similarity_cutoff)
        if args.console:
            console_dump(threads_data[0]["threads"], args.source_dir, language.load_ignores())
        else:
            print(json.dumps(threads_data, indent=2))
    elif args.stage == "server":
        port = int(args.source_dir)
        server.start_server(port, args.similarity_cutoff, args.console)
    else:
        print(f"Unknown stage: {args.stage}")
        exit(1)

if __name__ == "__main__":
    main()
