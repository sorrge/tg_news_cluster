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
import asyncio

from sanic import Sanic, response

import language
import groups
import news
import libs.cpp_stuff as cpp


index_folder = "index"
app = Sanic()
ignore_paragraph = language.load_ignores()
dump_text = False


@app.route("/<file_name>", methods=["PUT"])
async def index(request, file_name):
    if not cpp.is_indexer_ready():
        return response.empty(status=503)

    body = bytes.decode(request.body, encoding="utf-8")
    expire_after = int(request.headers["Cache-Control"][8:])
    file_info = language.make_file_info(file_name, language.parse_file(body.split("\n"), ignore_paragraph))
    tokens = news.simple_tokenize(file_info.text)
    tokens_for_grouping = groups.extract_text_for_grouping(file_info)
    title_tokens_for_grouping = news.simple_tokenize_with_numbers(file_info.title.lower())
    site_single_word = news.not_word_pattern.sub("", file_info.site.lower())
    url_tokens = news.simple_tokenize(file_info.url)

    work_id = cpp.index_article(file_name, file_info.text, tokens, tokens_for_grouping, file_info.title,
                                title_tokens_for_grouping,
                                file_info.site, site_single_word,
                                file_info.url, url_tokens,
                                groups.parse_datetime(file_info.time).timestamp(), float(expire_after))

    while True:
        result = cpp.check_work(work_id)
        if result != 0:
            break

        await asyncio.sleep(0.01)

    return response.empty(status=204 if result == 1 else 201)


@app.route("/<file_name>", methods=["DELETE"])
async def delete(request, file_name):
    if not cpp.is_indexer_ready():
        return response.empty(status=503)

    # print("request delete:", file_name)
    res = cpp.delete_article(file_name)
    # print("delete: res", res)
    if res != 0:
        while cpp.check_work(res) == 0:
            await asyncio.sleep(0.01)

    return response.empty(status=204 if res != 0 else 404)


@app.route("/threads", methods=["GET"])
def threads(request):
    if not cpp.is_indexer_ready():
        return response.empty(status=503)

    period = float(request.args["period"][0])
    lang_code = request.args["lang_code"][0]
    category = request.args["category"][0]
    res = cpp.get_threads(lang_code, category, period, dump_text)
    if dump_text:
        return response.text("\n\n".join(f"{title} {category} pri:{priority:.2f}\n\t" + "\n\t".join(articles)
                                         for title, category, articles, priority in res), status=200)

    return response.json({"threads":
                              [{"title": title,
                                "category": category,
                                "articles": articles} for title, category, articles, _ in res]}, status=200)


def start_server(port, similarity_cutoff, need_dump_text):
    global dump_text
    dump_text = need_dump_text
    if not os.path.exists(index_folder):
        os.mkdir(index_folder)

    cpp.start_indexer(float(similarity_cutoff))
    app.run(host='0.0.0.0', port=port, access_log=False, debug=False)
