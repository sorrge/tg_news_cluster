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
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <omp.h>
#include <iostream>
#include <cinttypes>
#include <fstream>
#include <atomic>
#include <cstdio>
#include <unordered_set>
#include <queue>
#include <shared_mutex>

#include "3rdparty/concurrentqueue.h"

#include "common.h"
#include "groups.h"

using namespace std;


const int max_deleted_articles = 10000;


struct article_input_data {
	string file_name, text;
	vector<string> text_tokens, tokens_for_grouping, title_tokens_for_grouping;
	string title, site, site_single_word, url;
	vector<string> url_tokens;
	double timestamp, expire_after;
	uint64_t work_id;
};


struct article_expiration {
	uint64_t id;
	string file_name;
	double expire_at;

	bool operator<(const article_expiration& d) const {
		return expire_at > d.expire_at;
	}
};


class text_normalizer {
	unordered_map<string, pair<int, double>> stem_idx_idf;
	string language;
public:
	text_normalizer(const string& dictionary_file, const string& language) : language(language) {
		ifstream dict(dictionary_file);
		string stem;
		int idx;
		double idf;
		while (dict >> stem >> idx >> idf)
			stem_idx_idf.emplace(stem, make_pair(idx, idf));
	}

	void normalize_article_for_categories_and_ranking(const article_input_data& ad, vector<int>& nn_input) const {
		nn_input.clear();
		auto stems = stem_words(ad.text_tokens, language == "en");
		unordered_set<string> unique_stems(stems.begin(), stems.end());
		vector<pair<string, double>> stems_sorted_by_idf;
		for(const string& s : unique_stems) {
			auto it = stem_idx_idf.find(s);
			if(it != stem_idx_idf.end())
				stems_sorted_by_idf.emplace_back(s, it->second.second);
		}

		sort(stems_sorted_by_idf.begin(), stems_sorted_by_idf.end(), [](const pair<string, double> &left, const pair<string, double> &right) {
		    return left.second < right.second;
		});

		vector<string> tokens;
		for(int i = 0; i < min(30, (int)stems.size()); ++i)
			if(stem_idx_idf.count(stems[i]) != 0)
				tokens.push_back(stems[i]);

		for(auto& p : stems_sorted_by_idf)
			tokens.push_back(p.first);

		tokens.push_back(ad.site_single_word);

		for(auto& t : ad.url_tokens)
			if(stem_idx_idf.count(t) != 0)
				tokens.push_back(t);

		for(auto& t : tokens) {
			auto it = stem_idx_idf.find(t);
			if(it != stem_idx_idf.end())
				nn_input.push_back(it->second.first);
		}
	}
};


class threads_index {
	grouper groups;
	vector<pair<string, uint64_t>> articles;
	mutable shared_timed_mutex mut;
public:
	threads_index(const unordered_map<hash_t, float>& idf, float max_idf, float similarity_cutoff) : groups(idf, max_idf, similarity_cutoff) {}

	void add1(const article_data& ad) {
		lock_guard<shared_timed_mutex> guard(mut);
		articles.emplace_back(ad.file_name, ad.id);
		groups.add_text(ad);
	}

	void add_many(const vector<pair<string, uint64_t>>& new_articles, const vector<article_data>& article_data) {
		lock_guard<shared_timed_mutex> guard(mut);
		articles.insert(articles.end(), new_articles.begin(), new_articles.end());
		groups.add_texts(article_data);
	}

	void get(vector<vector<int>>& threads, double min_time) const {
		shared_lock<shared_timed_mutex> guard(mut);
		unordered_set<int> articles_in_threads;
		threads.resize(groups.get_groups().size());
		for(int i = 0; i < (int)groups.get_groups().size(); ++i)
			if(groups.get_groups()[i].average_timestamp > min_time)
				for(auto& n : groups.get_groups()[i].texts) {
					articles_in_threads.insert(n);
					threads[i].push_back(n);
				}

		threads.reserve(threads.size() + articles.size() - articles_in_threads.size());
		for(int n = 0; n < (int)articles.size(); ++n)
			if(articles_in_threads.count(n) == 0 && groups.get_articles()[n].timestamp > min_time)
				threads.push_back({n});
	}

	void dump_sims() const {
		shared_lock<shared_timed_mutex> guard(mut);
		ofstream file("data/sims");
		//for(auto& e : groups.get_sims())
		//	file << articles[e.first.first].first << " " << articles[e.first.second].first << " " << e.second << "\n";
	}

	pair<string, uint64_t> operator[](int i) const {
		shared_lock<shared_timed_mutex> guard(mut);
		return articles[i];
	}

	float title_similarity(const vector<hash_t>& title_chunks, int thread) const {
		shared_lock<shared_timed_mutex> guard(mut);
		return groups.title_similarity(title_chunks, thread);
	}

	void sort_by_relevance(vector<int>& thread, int thread_id) const {
		shared_lock<shared_timed_mutex> guard(mut);
		vector<pair<int, float>> thread_similarities;
		thread_similarities.reserve(thread.size());
		for(int n : thread)
			thread_similarities.emplace_back(n, groups.article_group_similarity(n, thread_id));

		sort(thread_similarities.begin(), thread_similarities.end(), [](pair<int, float>& t1, pair<int, float>& t2) { return t1.second > t2.second; });
		for(int i = 0; i < (int)thread.size(); ++i)
			thread[i] = thread_similarities[i].first;
	}

	void remove_articles(const unordered_set<uint64_t>& to_remove) {
		lock_guard<shared_timed_mutex> guard(mut);
		unordered_set<int> aids_to_remove;
		vector<pair<string, uint64_t>> new_articles;
		new_articles.reserve(articles.size());
		for(int aid = 0; aid < (int)articles.size(); ++aid)
			if(to_remove.count(articles[aid].second) == 0)
				new_articles.push_back(articles[aid]);
			else
				aids_to_remove.insert(aid);

		new_articles.swap(articles);
		groups.remove_articles(aids_to_remove);
	}

	pair<int, int> get_num_articles_threads() const {
		shared_lock<shared_timed_mutex> guard(mut);
		return make_pair((int)articles.size(), (int)groups.get_groups().size());
	}
};


unordered_map<string, article_data> index_data;
priority_queue<article_expiration> expiration_data;
shared_timed_mutex index_data_mutex, work_finished_mutex;
unordered_map<string, threads_index*> threads_en, threads_ru;
unordered_map<uint64_t, int> work_finished;
moodycamel::ConcurrentQueue<article_input_data> indexing_work;
moodycamel::ConcurrentQueue<uint64_t> deleting_work;
vector<thread> indexers;
thread deleter, starter, index_compactor;
enum WorkerStatus { Starting, Running, GoingToSleep, Sleeping, WakingUp};
deque<atomic<WorkerStatus>> indexers_status;
atomic<WorkerStatus> deleter_status;
atomic<uint64_t> new_id {1};
atomic<size_t> num_deleted_on_disk {0};
double latest_time = 0;

string deleted_file_name = "index/deleted", deleted_file_backup = deleted_file_name + ".backup";
string category_model_en_file = "data/category_model_en.json", category_model_ru_file = "data/category_model_ru.json";
string ranking_model_en_file = "data/ranking_model_en.json", ranking_model_ru_file = "data/ranking_model_ru.json";
unique_ptr<text_normalizer> normalizer_cat_en, normalizer_cat_ru, normalizer_rank_en, normalizer_rank_ru;
vector<string> categories_en, categories_ru;
unordered_map<string, string> site_to_country;
unordered_map<string, float> site_popularity_ru, site_popularity_en;


article_data process_article(const article_input_data& data) {
	article_data res;
	res.file_name = data.file_name;
	res.timestamp = data.timestamp;
	res.expire_at = data.timestamp + data.expire_after;
	res.language = detect_language(data.text, data.site);
	if(res.language == "ru" || res.language == "en") {
		bool english = res.language == "en";
		auto& category_model = english ? category_model_en_file : category_model_ru_file;
		auto& normalizer = english ? *normalizer_cat_en : *normalizer_cat_ru;
		vector<int> nn_input;
		normalizer.normalize_article_for_categories_and_ranking(data, nn_input);
		auto nn_output = process_sample(nn_input, category_model);
		auto best_category_idx = distance(nn_output.begin(), max_element(nn_output.begin(), nn_output.end()));
		res.category = (english ? categories_en : categories_ru)[best_category_idx];
		if(res.category != "junk") {
			unordered_map<hash_t, int> chunks;
			get_words_with_count(data.tokens_for_grouping, chunks);
			res.chunks.insert(res.chunks.end(), chunks.begin(), chunks.end());
			res.title = data.title;
			auto title_chunks = get_words(data.title_tokens_for_grouping);
			res.title_chunks.insert(res.title_chunks.end(), title_chunks.begin(), title_chunks.end());
			auto& ranking_normalizer = english ? *normalizer_rank_en : *normalizer_rank_ru;
			auto& ranking_model = english ? ranking_model_en_file : ranking_model_ru_file;
			ranking_normalizer.normalize_article_for_categories_and_ranking(data, nn_input);
			res.popularity_prediction = process_sample(nn_input, ranking_model)[0];
			res.site = data.site;
		}
	}
	//cout << data.text << endl;

	return res;
}


int check_work(uint64_t work_id) {
	shared_lock<shared_timed_mutex> guard(work_finished_mutex);
	auto it = work_finished.find(work_id);
	if(it == work_finished.end())
		return 0;

	int res = it->second;
	work_finished.erase(it);
	return res;
}


void set_work_finished(uint64_t work_id, int status) {
	lock_guard<shared_timed_mutex> guard(work_finished_mutex);
	work_finished.emplace(work_id, status);
}


void deleter_worker() {
	ofstream deleted_file(deleted_file_name, ios::app | ios::binary);

	uint64_t to_delete;
	deleter_status = Running;
	while(true) {
		if(deleter_status == GoingToSleep) {
			deleted_file.close();
			deleter_status = Sleeping;
		}
		else if(deleter_status == WakingUp) {
			deleted_file.open(deleted_file_name, ios::app | ios::binary);
			deleter_status = Running;
		}
		else if(deleter_status == Running && deleting_work.try_dequeue(to_delete)) {
			serialize_value(deleted_file, to_delete);
			deleted_file.flush();
			++num_deleted_on_disk;
			set_work_finished(to_delete, 1);
		}
		else {
			this_thread::sleep_for(chrono::milliseconds(10));
		}
	}
}


void add_article_to_threads_index(const article_data& ad) {
	if(ad.is_in_threads())
		(ad.language == "en" ? threads_en : threads_ru)[ad.category]->add1(ad);
}


bool add_article_to_memory_index(const article_data& ad) {
	auto existing_it = index_data.find(ad.file_name);
	bool updating = false;
	if(existing_it != index_data.end()) {
		if(existing_it->second.id >= ad.id)
			return false;

		updating = true;
		deleting_work.enqueue(existing_it->second.id);
		index_data.erase(existing_it);
	}

	index_data.emplace(ad.file_name, ad);
	if(latest_time < ad.timestamp)
		latest_time = ad.timestamp;

	expiration_data.push(article_expiration {ad.id, ad.file_name, ad.expire_at});
	while(expiration_data.top().expire_at < latest_time) {
		auto it = index_data.find(expiration_data.top().file_name);
		if(it != index_data.end() && it->second.id == expiration_data.top().id) {
			index_data.erase(it);
			deleting_work.enqueue(expiration_data.top().id);
		}

		expiration_data.pop();
	}

	return updating;
}


void read_index(const string& index_file_name, const unordered_set<uint64_t>& deleted_ids, vector<article_data>& loaded) {
	ifstream index_file_in(index_file_name, ios::binary);
	for(; index_file_in;) {
		article_data ad;
		ad.serialize(index_file_in);
		if(!index_file_in)
			break;

		if(deleted_ids.count(ad.id) == 0)
			loaded.push_back(ad);
	}
}


void prepare_index(const string& index_file_name, const unordered_set<uint64_t>& deleted_ids) {
	vector<article_data> loaded;
	read_index(index_file_name, deleted_ids, loaded);

	unordered_map<string, pair<vector<pair<string, uint64_t>>, vector<article_data>>> threads_init_data_en, threads_init_data_ru;

	{
		lock_guard<shared_timed_mutex> guard(index_data_mutex);
		uint64_t max_id = 0;
		for(auto& ad : loaded) {
			max_id = max(max_id, ad.id);
			add_article_to_memory_index(ad);
			if(ad.is_in_threads()) {
				auto& data = (ad.language == "en" ? threads_init_data_en : threads_init_data_ru)[ad.category];
				data.first.emplace_back(ad.file_name, ad.id);
				data.second.push_back(ad);
			}
		}

		if(new_id <= max_id)
			new_id = max_id + 1;
	}

	for(auto& kvp : threads_init_data_en)
		threads_en[kvp.first]->add_many(kvp.second.first, kvp.second.second);

	for(auto& kvp : threads_init_data_ru)
		threads_ru[kvp.first]->add_many(kvp.second.first, kvp.second.second);
}


bool is_indexer_ready() {
	for(int i = 0; i < omp_get_max_threads(); ++i)
		if(indexers_status[i] == Starting)
			return false;

	return true;
}


void indexer_worker(int worker_idx, unordered_set<uint64_t> deleted_ids) {
	//cout << worker_idx << " started" << endl;
	string index_file_name = "index/idx_" + to_string(worker_idx);
	prepare_index(index_file_name, deleted_ids);

	ofstream index_file(index_file_name, ios::app | ios::binary);
	article_input_data data;
	article_data to_save;

	{
		lock_guard<shared_timed_mutex> guard(index_data_mutex);
		indexers_status[worker_idx] = Running;
		if(is_indexer_ready()) {
			//threads_en["society"]->dump_sims();
			cout << "Server ready. " << index_data.size() << " articles in index" << endl;
			cout << "ru: ";
			for(auto& kvp : threads_ru) {
				auto stats = kvp.second->get_num_articles_threads();
				cout << kvp.first << ": " << stats.first << "a " << stats.second << "t;    ";
			}

			cout << "\nen: ";
			for(auto& kvp : threads_en) {
				auto stats = kvp.second->get_num_articles_threads();
				cout << kvp.first << ": " << stats.first << "a " << stats.second << "t;    ";
			}

			cout << endl;
		}
	}

	while(true) {
		if(indexers_status[worker_idx] == GoingToSleep) {
			index_file.close();
			indexers_status[worker_idx] = Sleeping;
		}
		else if(indexers_status[worker_idx] == WakingUp) {
			index_file.open(index_file_name, ios::app | ios::binary);
			indexers_status[worker_idx] = Running;
		}
		else if(indexers_status[worker_idx] == Running && indexing_work.try_dequeue(data)) {
			to_save = process_article(data);

			bool updating = false;
			{
				lock_guard<shared_timed_mutex> guard(index_data_mutex);
				to_save.id = ++new_id;
				updating = add_article_to_memory_index(to_save);
				//cout << index_data.size() << " articles in index" << endl;
			}

			add_article_to_threads_index(to_save);

			to_save.serialize(index_file);
			index_file.flush();
			set_work_finished(data.work_id, updating ? 1 : 2);
		}
		else
			this_thread::sleep_for(chrono::milliseconds(10));
	}
}


void read_deleted(const string& file_name, unordered_set<uint64_t>& deleted_ids) {
	ifstream deleted_file(file_name, ios::binary);
	for(; deleted_file;) {
		uint64_t id;
		serialize_value(deleted_file, id);
		if(!deleted_file)
			break;

		deleted_ids.insert(id);
	}
}


void index_compactor_worker() {
	while(true) {
		if(num_deleted_on_disk > max_deleted_articles) {
			cout << "Compacting index" << endl;
			deleter_status = GoingToSleep;
			while(deleter_status != Sleeping)
				this_thread::sleep_for(chrono::milliseconds(10));

			rename(deleted_file_name.c_str(), deleted_file_backup.c_str());
			num_deleted_on_disk = 0;
			deleter_status = WakingUp;

			unordered_set<uint64_t> deleted_ids;
			read_deleted(deleted_file_backup, deleted_ids);
			vector<article_data> loaded;
			for(int indexer_id = 0; indexer_id < omp_get_max_threads(); ++indexer_id) {
				indexers_status[indexer_id] = GoingToSleep;
				while(indexers_status[indexer_id] != Sleeping)
					this_thread::sleep_for(chrono::milliseconds(10));

				loaded.clear();
				string index_file_name = "index/idx_" + to_string(indexer_id);
				string index_file_name_compacted = "index/idx_" + to_string(indexer_id) + ".compacted";
				read_index(index_file_name, deleted_ids, loaded);
				{
					ofstream index_file(index_file_name_compacted);
					for(auto& ad : loaded)
						ad.serialize(index_file);
				}

				rename(index_file_name_compacted.c_str(), index_file_name.c_str());
				indexers_status[indexer_id] = WakingUp;
			}

			remove(deleted_file_backup.c_str());

//			for(auto& kvp : threads_en)
//				kvp.second->remove_articles(deleted_ids);

//			for(auto& kvp : threads_ru)
//				kvp.second->remove_articles(deleted_ids);
		}
		else
			this_thread::sleep_for(chrono::seconds(1));
	}
}


vector<string> read_categories(const string& file_name) {
	ifstream file(file_name);
	string cat;
	vector<string> res;
	while (file >> cat)
		res.emplace_back(cat);

	return res;
}


void starter_worker(float similarity_cutoff) {
	unordered_set<uint64_t> deleted_ids;

#pragma omp parallel sections
	{
#pragma omp section
		{
			normalizer_cat_en.reset(new text_normalizer("data/dictionary_en.tsv", "en"));
			normalizer_cat_ru.reset(new text_normalizer("data/dictionary_ru.tsv", "ru"));

			categories_en = read_categories("data/categories_en");
			categories_ru = read_categories("data/categories_ru");

			normalizer_rank_en.reset(new text_normalizer("data/ranking_dictionary_en.tsv", "en"));
			normalizer_rank_ru.reset(new text_normalizer("data/ranking_dictionary_ru.tsv", "ru"));

			ifstream src_country_file("data/source_country.tsv");
			string line;
			while (getline(src_country_file, line)) {
				auto fields = split(line, '\t');
				site_to_country[fields[0]] = fields[1];
			}


			ifstream src_pop_en_file("data/site_popularity_en.tsv");
			while (getline(src_pop_en_file, line)) {
				auto fields = split(line, '\t');
				site_popularity_en[fields[0]] = stof(fields[1]);
			}

			ifstream src_pop_ru_file("data/site_popularity_ru.tsv");
			while (getline(src_pop_ru_file, line)) {
				auto fields = split(line, '\t');
				site_popularity_ru[fields[0]] = stof(fields[1]);
			}
		}

#pragma omp section
		load_nn_model(category_model_en_file);

#pragma omp section
		load_nn_model(category_model_ru_file);

#pragma omp section
		load_nn_model(ranking_model_en_file);

#pragma omp section
		load_nn_model(ranking_model_ru_file);

#pragma omp section
		{
			auto en_idf_data = read_idf_file("data/chunk_counts_en.bin");
			for(string cat : categories_en)
				if(cat != "junk")
					threads_en.emplace(cat, new threads_index(en_idf_data->first, en_idf_data->second, similarity_cutoff));
		}

#pragma omp section
		{
			auto ru_idf_data = read_idf_file("data/chunk_counts_ru.bin");
			for(string cat : categories_ru)
				if(cat != "junk")
					threads_ru.emplace(cat, new threads_index(ru_idf_data->first, ru_idf_data->second, similarity_cutoff));
		}

#pragma omp section
		{
			read_deleted(deleted_file_name, deleted_ids);
			if(file_exists(deleted_file_backup)) {
				read_deleted(deleted_file_backup, deleted_ids);
				vector<uint64_t> deleted_vector(deleted_ids.begin(), deleted_ids.end());
				{
					ofstream deleted_file(deleted_file_name, ios::binary);
					serialize_value(deleted_file, deleted_vector[0], deleted_vector.size());
				}

				remove(deleted_file_backup.c_str());
			}

			num_deleted_on_disk = deleted_ids.size();
			cout << num_deleted_on_disk << " deleted on disk" << endl;
		}
	}

	for(int i = 0; i < omp_get_max_threads(); ++i)
		indexers.push_back(thread(&indexer_worker, i, deleted_ids));

	deleter = thread(&deleter_worker);
	index_compactor = thread(&index_compactor_worker);
}


void start_indexer(float similarity_cutoff) {
	for(int i = 0; i < omp_get_max_threads(); ++i)
		indexers_status.emplace_back(Starting);

	starter = thread(&starter_worker, similarity_cutoff);
}


uint64_t index_article(const string& file_name, const string& text, const vector<string>& text_tokens, const vector<string>& tokens_for_grouping,
		const string& title, const vector<string>& title_tokens_for_grouping,
		const string& site, const string& site_single_word,
		const string& url, const vector<string>& url_tokens,
		double timestamp, double expire_after) {
	//cout << "indexing " << title << endl;
	article_input_data data {file_name, text, text_tokens, tokens_for_grouping, title_tokens_for_grouping, title, site, site_single_word, url, url_tokens, timestamp, expire_after, ++new_id};
	indexing_work.enqueue(data);
	return data.work_id;
}


uint64_t delete_article(const string& file_name) {
	lock_guard<shared_timed_mutex> guard(index_data_mutex);
	auto it = index_data.find(file_name);
	if(it == index_data.end())
		return 0;

	uint64_t id = it->second.id;
	deleting_work.enqueue(id);
	//cout << "deleting " << file_name << " id " << it->second.id << endl;
	index_data.erase(it);
	return id;
}


vector<tuple<string, string, vector<string>, double>> get_threads(const string& language, const string& category, double period, bool dump_text) {
	const auto& pop_data = language == "en" ? site_popularity_en : site_popularity_ru;

	double period_days = ceil(period / 86400.0);
	double min_time = latest_time - period_days * 86400;
	vector<string> categories;
	if(category == "any") {
		for(auto& cat : categories_en)
			if(cat != "junk")
				categories.push_back(cat);
	}
	else
		categories.push_back(category);

	vector<tuple<string, string, vector<string>, double>> res;
	for(auto& cat : categories) {
		const auto& thread_indexer = *(language == "en" ? threads_en : threads_ru)[cat];
		vector<vector<int>> threads;
		thread_indexer.get(threads, min_time);
		shared_lock<shared_timed_mutex> guard(index_data_mutex);
		for(int thread_id = 0; thread_id < (int)threads.size(); ++thread_id) {
			auto& t = threads[thread_id];
			vector<tuple<string, vector<hash_t>, float>> titles;
			vector<int> valid_files_idx;
			double popularity_sum = 0;
			double ages_sum = 0;
			unordered_set<string> countries;
			float max_site_popularity = -1000000.0f;
			for(int n : t) {
				auto a = thread_indexer[n];
				auto it = index_data.find(a.first);
				if(it != index_data.end() && it->second.id == a.second) {
					valid_files_idx.push_back(n);
					titles.emplace_back(it->second.title, it->second.title_chunks, 0);
					popularity_sum += it->second.popularity_prediction;
					ages_sum += latest_time - it->second.timestamp;
					auto c_it = site_to_country.find(it->second.site);
					if(c_it != site_to_country.end())
						countries.insert(c_it->second);

					auto p_it = pop_data.find(it->second.site);
					float pop = 0;
					if(p_it != pop_data.end())
						pop = p_it->second;

					max_site_popularity = max(max_site_popularity, pop);
				}
			}

			if(!titles.empty()) {
				if(titles.size() > 1) {
					for(auto& tt : titles)
						get<2>(tt) = thread_indexer.title_similarity(get<1>(tt), thread_id) / (1 + get<0>(tt).size());

					thread_indexer.sort_by_relevance(valid_files_idx, thread_id);
				}

				string title = get<0>(*max_element(titles.begin(), titles.end(),
						[](tuple<string, vector<hash_t>, float>& t1, tuple<string, vector<hash_t>, float>& t2) { return get<2>(t1) < get<2>(t2); }));

				vector<string> valid_files;
				valid_files.reserve(titles.size());
				for(int n : valid_files_idx) {
					auto a = thread_indexer[n];
					auto it = index_data.find(a.first);
					valid_files.push_back(dump_text ? it->second.title + " site:" + it->second.site: it->second.file_name);
				}

				double average_age_days = ages_sum / valid_files.size() / 86400.0;
				double diversity_factor = 0;
				if(language == "en" && countries.size() > 1)
					diversity_factor = min((int)countries.size(), 4) * 0.4;
				else
					diversity_factor = -0.5;

				float popularity_factor = max_site_popularity * 0.3;

				if(dump_text)
					title += " #" + to_string(valid_files.size()) + " age:" + to_string(average_age_days) + " countries:" + join(countries, ";") + " pop:" + to_string(popularity_factor);

				double priority = popularity_sum / valid_files.size() + min(4.0, valid_files.size() * 0.3) + diversity_factor + popularity_factor;
				res.emplace_back(title, cat, valid_files, priority);
			}
		}
	}

	sort(res.begin(), res.end(), [](tuple<string, string, vector<string>, double>& t1, tuple<string, string, vector<string>, double>& t2) { return get<3>(t1) > get<3>(t2); });
	return res;
}
