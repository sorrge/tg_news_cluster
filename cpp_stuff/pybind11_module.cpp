//    Copyright 2020 sorrge
//
//    This file is part of tg_news_cluster.
//
//    tg_news_cluster is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    tg_news_cluster is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with tg_news_cluster.  If not, see <https://www.gnu.org/licenses/>.
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.h"
#include "groups.h"

namespace py = pybind11;
using namespace std;


void fasttext_init();
vector<pair<float, string>> fasttext_detect_languages(const string& s, int num_preds);
void whatlang_wrapper_init();
pair<double, string> whatlang_wrapper_detect(const string& s);
void load_sources_languages(const unordered_map<string, unordered_map<string, double>>& sl);
string detect_language(const string& s, const string& source);
vector<string> detect_languages(const vector<string>& s, const vector<string>& sources);
vector<vector<int>> group_texts(const vector<tuple<vector<string>, string, string, double>>& article_data_text, const string& idf_file, float similarity_cutoff);
void chunks_idf(const vector<vector<string>>& texts, const string& out_file);
void die_when_parent_process_dies();
void start_indexer(float similarity_cutoff);
uint64_t index_article(const string& file_name, const string& text, const vector<string>& text_tokens, const vector<string>& tokens_for_grouping,
		const string& title, const vector<string>& title_tokens_for_grouping,
		const string& site, const string& site_single_word,
		const string& url, const vector<string>& url_tokens,
		double timestamp, double expire_after);

bool is_indexer_ready();
uint64_t delete_article(const string& file_name);
int check_work(uint64_t work_id);
vector<tuple<string, string, vector<string>, double>> get_threads(const string& language, const string& category, double period, bool dump_text);


PYBIND11_MODULE(cpp_stuff, m) {
	fasttext_init();
	whatlang_wrapper_init();
	libstemmer_init();

    m.doc() = "pybind11-based C++ library for tgnews";

    m.def("utf8_test",
        [](const std::string &s) {
            cout << "utf-8 is icing on the cake." << endl;
            cout << s << endl;
        }
    );

    m.def("fasttext_detect_languages", &fasttext_detect_languages, "Detect languages with FastText");
    m.def("whatlang_detect", &whatlang_wrapper_detect, "Detect languages with whatlang");
    m.def("load_sources_languages", &load_sources_languages, "Load source->languages map");
    m.def("detect_language", &detect_language, "Detect language in a string");
    m.def("detect_languages", &detect_languages, "Detect languages in a string array");
    m.def("load_nn_model", &load_nn_model, "Load Keras model for future use");
    m.def("run_nn", &process_samples, "Run the nn model");
    m.def("stem_words", &stem_words, "Stem English or Russian words");
    m.def("make_idf", &chunks_idf, "Make IDF file for grouping");
    m.def("load_idf", &load_idf, "Load IDF file for future use in grouping");
    m.def("group_texts", &group_texts, "Group similar texts");
    m.def("die_when_parent_process_dies", &die_when_parent_process_dies, "Makes current process die when the parent process dies");
    m.def("index_article", &index_article, "Index an article");
    m.def("delete_article", &delete_article, "Delete an article");
    m.def("start_indexer", &start_indexer, "Start the indexer");
    m.def("is_indexer_ready", &is_indexer_ready, "Check if indexer has loaded the index");
    m.def("check_work", &check_work, "Check the status of work item");
    m.def("get_threads", &get_threads, "Get threads");
    m.def("calc_popularities", &calc_popularities, "Calculate article popularities based on similarity");
    m.def("calc_similarities", &calc_similarities, "Calculate article similarities");
    m.def("get_chunks", &get_chunks, "Get chunks for text");
}
