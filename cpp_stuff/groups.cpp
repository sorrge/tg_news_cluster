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
#include <tuple>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <shared_mutex>

#include "common.h"
#include "groups.h"

using namespace std;


unordered_map<string, pair<unordered_map<hash_t, float>, float>*> loaded_idfs;


pair<unordered_map<hash_t, float>, float>* read_idf_file(const string& file_name) {
	//cout << "Loading IDF " << file_name << endl;
	ifstream file(file_name, ios::binary);
	int num_texts;
	file.read((char*)&num_texts, sizeof(num_texts));
	auto loaded_data = new pair<unordered_map<hash_t, float>, float>;
	float max_idf = logf(num_texts);
	loaded_data->second = max_idf;

	while(true) {
		int count;
		file.read((char*)&count, sizeof(count));
		if(!file)
			break;

		int cc;
		file.read((char*)&cc, sizeof(cc));
		float cur_idf = max_idf - logf(count);
		for(int i = 0; i < cc; ++i) {
			hash_t h;
			file.read((char*)&h, sizeof(h));
			if(!file)
				break;

			loaded_data->first[h] = cur_idf;
		}
	}

	//cout << "IDF " << file_name << " loaded" << endl;
	return loaded_data;
}


void load_idf(const string& file_name) {
	if(loaded_idfs.count(file_name) > 0)
		return;

	loaded_idfs[file_name] = read_idf_file(file_name);
}


unordered_set<hash_t> get_words(const vector<string>& text) {
	unordered_set<hash_t> words;
	hash_t prev_hash = string_hash("");
	for(auto& word : text) {
		hash_t current_hash = string_hash(word);
		words.insert(current_hash);
		words.insert(hash_uint32(current_hash + prev_hash));
		prev_hash = current_hash;
	}

	return words;
}


void chunks_idf(const vector<vector<string>>& texts, const string& out_file) {
	unordered_map<hash_t, int> base_counts;
	for(auto& s : texts)
		for(auto& chunk : get_words(s))
			++base_counts[chunk];

	ofstream file(out_file, ios::binary);
	int num_texts = (int)texts.size();
	file.write((char*)&num_texts, sizeof(num_texts));
	vector<pair<hash_t, int>> base_counts_sorted(base_counts.begin(), base_counts.end());
	sort(base_counts_sorted.begin(), base_counts_sorted.end(), [](pair<hash_t, int>& p1, pair<hash_t, int>& p2)
			{ return p1.second < p2.second || (p1.second == p2.second && p1.first < p2.first); });

	unordered_map<int, int> count_of_counts;
	for(auto& kvp : base_counts_sorted)
		++count_of_counts[kvp.second];


	int prev_count = -1;
	for(auto& kvp : base_counts_sorted) {
		if(kvp.second == 1)
			continue;

		if(kvp.second != prev_count) {
			prev_count = kvp.second;
			file.write((char*)&prev_count, sizeof(prev_count));
			int cc = count_of_counts[kvp.second];
			file.write((char*)&cc, sizeof(cc));
		}

		file.write((char*)&kvp.first, sizeof(kvp.first));
	}
}


void get_words_with_count_parallel(const vector<tuple<vector<string>, string, string, double>>& article_data_text, vector<article_data>& article_data_processed) {
	article_data_processed.resize(article_data_text.size());
#pragma omp parallel for
	for(int i = 0; i < (int)article_data_text.size(); ++i) {
		unordered_map<hash_t, int> chunks;
		get_words_with_count(get<0>(article_data_text[i]), chunks);
		article_data_processed[i].chunks.reserve(chunks.size());
		for(auto& kvp : chunks)
			article_data_processed[i].chunks.push_back(kvp);

		article_data_processed[i].site = get<1>(article_data_text[i]);
		article_data_processed[i].file_name = get<2>(article_data_text[i]);
		article_data_processed[i].timestamp = get<3>(article_data_text[i]);
	}
}


vector<float> calc_popularities(const vector<tuple<vector<string>, string, string, double>>& text_data, const string& idf) {
	vector<article_data> articles;
	get_words_with_count_parallel(text_data, articles);
	const auto& idf_data = *loaded_idfs.at(idf);
	grouper g(idf_data.first, idf_data.second, 1.0f);
	g.add_texts(articles);
	vector<float> popularities(text_data.size(), 0);
//	for(auto& sim_kvp : g.get_sims()) {
//		popularities[sim_kvp.first.first] += sim_kvp.second;
//		popularities[sim_kvp.first.second] += sim_kvp.second;
//	}

	return popularities;
}


void get_words_with_count(const vector<string>& text, unordered_map<hash_t, int>& words) {
	hash_t prev_hash = 0;
	for(auto& word : text) {
		hash_t current_hash = string_hash(word);
		++words[current_hash];
		if(prev_hash != 0)
			++words[hash_uint32(current_hash + prev_hash)];

		prev_hash = current_hash;
	}
}


vector<vector<int>> group_texts(const vector<tuple<vector<string>, string, string, double>>& article_data_text, const string& idf_file, float similarity_cutoff) {
	auto idf_data = loaded_idfs.at(idf_file);
	grouper g(idf_data->first, idf_data->second, similarity_cutoff);
	vector<article_data> article_data_processed;
	get_words_with_count_parallel(article_data_text, article_data_processed);
	g.add_texts(article_data_processed);
//	for(auto& t : text_to_chunks)
//		g.add_text(t);

//	for(int i = 0; i < texts.size(); ++i) {
//		cout << i << "\t";
//		for(string w : texts[i])
//			cout << w << " ";
//
//		cout << "\n";
//	}

	vector<vector<int>> res;
	res.reserve(g.get_groups().size());
	for(auto& a : g.get_groups())
		res.push_back(a.texts);

	return res;
}


vector<pair<pair<int, int>, float>> calc_similarities(const vector<tuple<vector<string>, string, string, double>>& article_data_text, const string& idf_file, float similarity_cutoff) {
	auto idf_data = loaded_idfs.at(idf_file);
	grouper g(idf_data->first, idf_data->second, similarity_cutoff);
	vector<article_data> article_data_processed;
	get_words_with_count_parallel(article_data_text, article_data_processed);
	g.add_texts(article_data_processed);
	vector<pair<pair<int, int>, float>> res;//(g.get_sims().begin(), g.get_sims().end());
	return res;
}


unordered_map<hash_t, pair<int, float>> get_chunks(const vector<string>& words, const string& idf_file) {
	auto idf_data = loaded_idfs.at(idf_file);
	auto& idf = idf_data->first;
	float max_idf = idf_data->second;
	unordered_map<hash_t, int> chunks;
	get_words_with_count(words, chunks);
	unordered_map<hash_t, pair<int, float>> res;
	for(auto& kvp : chunks) {
		auto it = idf.find(kvp.first);
		float c_idf = it == idf.end() ? max_idf : it->second;
		if (c_idf >= 1.f)
			res.emplace(kvp.first, make_pair(kvp.second, c_idf));
	}

	return res;
}
