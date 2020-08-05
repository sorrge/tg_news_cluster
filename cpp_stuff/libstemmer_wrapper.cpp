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
#include <vector>
#include <unordered_map>
#include <memory>
#include <omp.h>
#include <mutex>

#include <libstemmer.h>

using namespace std;


class stemmer {
	vector<sb_stemmer*> thread_stemmers;
	unordered_map<string, string> cache;
public:
	stemmer(const string& language) {
		thread_stemmers.resize(omp_get_max_threads());
		for(int i = 0; i < (int)thread_stemmers.size(); ++i)
			thread_stemmers[i] = sb_stemmer_new(language.c_str(), NULL);
	}

	pair<string, bool> stem_word(const string& word, int thread_idx) {
		auto it = cache.find(word);
		if(it == cache.end()) {
			const char *res = (const char *)sb_stemmer_stem(thread_stemmers[thread_idx], (const sb_symbol*)word.c_str(), word.size());
			int res_size = sb_stemmer_length(thread_stemmers[thread_idx]);
			return make_pair(string(res, res_size), false);
		}

		return make_pair(it->second, true);
	}

	void add_stem(const string& word, const string& stem) {
		cache.emplace(word, stem);
	}
};


unique_ptr<stemmer> english_stemmer, russian_stemmer;


void libstemmer_init() {
	english_stemmer.reset(new stemmer("english"));
	russian_stemmer.reset(new stemmer("russian"));
}


mutex stemmer_mutex;


vector<string> stem_words(const vector<string>& words, bool english) {
	lock_guard<mutex> guard(stemmer_mutex);
	stemmer& s = english ? *english_stemmer : *russian_stemmer;
	vector<string> stems(words.size());
	vector<int> in_cache(words.size());
#pragma omp parallel for
	for(int i = 0; i < (int)words.size(); ++i) {
		auto res = s.stem_word(words[i], omp_get_thread_num());
		stems[i] = res.first;
		in_cache[i] = (int)res.second;
	}

	for(int i = 0; i < (int)words.size(); ++i)
		if(in_cache[i] == 0)
			s.add_stem(words[i], stems[i]);

	return stems;
}


