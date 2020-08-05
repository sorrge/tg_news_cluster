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
rm -rf package
mkdir package
shiv -e tgnews.main -o package/tgnews sanic -p "/usr/bin/env python3" --site-packages for_package --uncompressed
cp libwhatlang_wrapper.so package/
cp readme.txt package/

mkdir package/data
cp data/{categories_en,categories_ru,category_model_en.json,category_model_ru.json,dictionary_en.tsv,dictionary_ru.tsv,chunk_counts_en.bin,chunk_counts_ru.bin,lid.176.ftz,ranking_dictionary_en.tsv,ranking_dictionary_ru.tsv,ranking_model_en.json,ranking_model_ru.json,site_languages.json,site_popularity_en.tsv,site_popularity_ru.tsv,source_country.tsv,ignore_paragraph} package/data/

cp -rL src/ package/src

cd package
zip -r solution.zip *
