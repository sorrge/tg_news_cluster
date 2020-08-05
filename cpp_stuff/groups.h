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
#pragma once
#include <cmath>
#include <vector>

using namespace std;


extern unordered_map<string, pair<unordered_map<hash_t, float>, float>*> loaded_idfs;

const double max_seconds_gap_between_articles_in_thread = 1.5 * 24 * 60 * 60;
const int max_document_per_chunk = 500, max_results = 50;
const float similarity_penalty_for_same_site = 0.1f;


struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return (size_t)hash_uint64(hash_uint64(h1) + h2);
    }
};


inline pair<int, int> edge(int n1, int n2) {
	return make_pair(min(n1, n2), max(n1, n2));
}


class sparse_similarity {
	unordered_map<pair<int, int>, float, pair_hash> edge_similarity;
public:
	void update(int v1, int v2, float sim) {
		auto e = edge(v1, v2);
		auto it = edge_similarity.find(e);
		if(it == edge_similarity.end())
			edge_similarity.emplace(e, sim);
		else if(it->second < sim)
			it->second = sim;
	}

	float get(int v1, int v2) const {
		auto it = edge_similarity.find(edge(v1, v2));
		if(it == edge_similarity.end())
			return 0.0f;

		return it->second;
	}

	const unordered_map<pair<int, int>, float, pair_hash>& get_sims() const { return edge_similarity; }
};


struct article_data {
	uint64_t id;
	string file_name, language, category, title;
	double popularity_prediction, timestamp, expire_at;
	vector<pair<hash_t, int>> chunks;
	vector<hash_t> title_chunks;
	string site;

	template<class Stream>
	void serialize(Stream& where) {
		serialize_value(where, id);
		serialize_sized(where, file_name);
		serialize_sized(where, language);
		serialize_sized(where, category);
		serialize_sized(where, title);
		serialize_value(where, popularity_prediction);
		serialize_value(where, timestamp);
		serialize_value(where, expire_at);
		serialize_sized(where, chunks);
		serialize_sized(where, title_chunks);
		serialize_sized(where, site);
	}

	bool is_in_threads() const { return (language == "ru" || language == "en") && category != "junk"; }
};


struct chunk_ptr {
	hash_t chunk;
	int pos;
};


struct grouper_article_data {
	vector<chunk_ptr> chunks;
	hash_t site, file_name;
	double timestamp;
	int gid;

	float content_sum;
};


struct grouper_thread_data {
	unordered_map<hash_t, int> chunk_pos;
	unordered_set<hash_t> sites;
	vector<int> texts;
	double average_timestamp;

	float content_sum;
};


struct chunk_info {
	int doc_id, pos_in_article;
	float count;
};


class grouper {
	const unordered_map<hash_t, float>& idf;
	const float max_idf, similarity_cutoff;

	vector<grouper_article_data> articles;
	unordered_map<hash_t, vector<chunk_info>> chunk_to_docs;
	vector<grouper_thread_data> groups;
public:
	grouper(const unordered_map<hash_t, float>& idf, float max_idf, float similarity_cutoff) : idf(idf), max_idf(max_idf), similarity_cutoff(similarity_cutoff) {}

	void add_texts(const vector<article_data>& new_ads) {
		auto prev_size = articles.size();
		articles.resize(prev_size + new_ads.size());

		for(int i = (int)prev_size; i < (int)articles.size(); ++i) {
			add_article(new_ads[i - prev_size], i);
			vector<pair<int, float>> text_similarities;
			calc_similarities_for_article(i, text_similarities);

			if(!text_similarities.empty())
				update_groups(i, text_similarities);
		}
	}

	void add_text(const article_data& text) {
		//debug_check_all_doc_id_valid("add_text");

		articles.resize(articles.size() + 1);
		int new_aid = (int)articles.size() - 1;
		add_article(text, new_aid);

		vector<pair<int, float>> sims;
		calc_similarities_for_article(new_aid, sims);
		if(!sims.empty())
			update_groups(new_aid, sims);
	}

	float title_similarity(const vector<hash_t>& title_chunks, int gid) const {
		float sim = 0;
		const auto& g = groups[gid];
		for(hash_t chunk : title_chunks) {
			auto c_it = g.chunk_pos.find(chunk);
			if(c_it == g.chunk_pos.end())
				continue;

			auto it = idf.find(chunk);
			float c_idf = it == idf.end() ? max_idf : it->second;
			sim += c_idf * chunk_to_docs.at(chunk)[c_it->second].count;
		}

		return sim;
	}

	//float get_sim(int t1, int t2) const { return similarities.get(t1, t2); }
	const vector<grouper_thread_data>& get_groups() const { return groups; }
	const vector<grouper_article_data>& get_articles() const { return articles; }
	//const unordered_map<pair<int, int>, float, pair_hash>& get_sims() const { return similarities.get_sims(); }

	float article_group_similarity(int aid, int gid) const {
		const auto& g = groups.at(gid);
		const auto& a = articles.at(aid);
		if(fabs(g.average_timestamp - a.timestamp) > max_seconds_gap_between_articles_in_thread)
			return 0;

		float mult = 1.0f / max(g.content_sum, a.content_sum);
		float sim = 0;
		if(g.sites.count(a.site) != 0)
			sim -= similarity_penalty_for_same_site;

		for(const auto& cp : a.chunks) {
			auto it = g.chunk_pos.find(cp.chunk);
			if(it != g.chunk_pos.end()) {
				auto it2 = idf.find(cp.chunk);
				float c_idf = it2 == idf.end() ? max_idf : it2->second;
				const auto& v = chunk_to_docs.at(cp.chunk);
				sim += c_idf * min(v[cp.pos].count, v[it->second].count) * mult;
			}
		}

		return sim;
	}

	void remove_articles(const unordered_set<int>& to_remove) {
		vector<int> new_ids(articles.size(), -1);
		vector<grouper_article_data> new_articles(articles.size() - to_remove.size());
		for(int new_id = 0, old_id = 0; old_id < (int)articles.size(); ++old_id)
			if(to_remove.count(old_id) == 0) {
				new_articles[new_id] = articles[old_id];
				new_ids[old_id] = new_id++;
			}

		new_articles.swap(articles);

		vector<int> new_group_ids(groups.size(), -1);
		vector<grouper_thread_data> new_groups;
		new_groups.reserve(groups.size());
		for(int new_id = 0, old_id = 0; old_id < (int)groups.size(); ++old_id) {
			auto& g = groups[old_id];
			vector<int> new_texts;
			new_texts.reserve(g.texts.size());
			for(int a : g.texts)
				if(new_ids[a] != -1)
					new_texts.push_back(new_ids[a]);

			if(!new_texts.empty()) {
				g.texts.swap(new_texts);
				new_groups.push_back(g);
				new_group_ids[old_id] = new_id++;
			}
		}

		new_groups.swap(groups);

		for(auto& a : articles)
			if(a.gid != -1)
				a.gid = new_group_ids[a.gid];

		for(auto& kvp : chunk_to_docs) {
			vector<chunk_info> new_cis;
			new_cis.reserve(kvp.second.size());
			for(auto& ci : kvp.second)
				if((ci.doc_id >= 0 && new_ids[ci.doc_id] != -1) ||
						(ci.doc_id < 0 && new_group_ids[-1 - ci.doc_id] != -1)) {
					if(ci.doc_id >= 0)
						ci.doc_id = new_ids[ci.doc_id];
					else
						ci.doc_id = -1 - new_group_ids[-1 - ci.doc_id];

					new_cis.push_back(ci);
					chunk_pos(ci, kvp.first) = (int)new_cis.size() - 1;
				}

			kvp.second.swap(new_cis);
		}
	}
private:
	void add_article(const article_data& ad, int id) {
		auto& a = articles[id];
		a.file_name = string_hash(ad.file_name);
		a.site = string_hash(ad.site);
		a.timestamp = ad.timestamp;
		a.content_sum = 0.0f;
		a.gid = -1;
		for(auto& c : ad.chunks) {
			auto it = idf.find(c.first);
			float c_idf = it == idf.end() ? max_idf : it->second;
			if (c_idf >= 1.f) {
				a.content_sum += c_idf * c.second;
//				texts[text_id].content_sum += c_idf * i->second * i->second;
//				texts[text_id].content_sum += c_idf * (1 + logf(i->second));
//				texts[text_id].content_sum += c_idf;

				auto& v = chunk_to_docs[c.first];
				v.push_back({id, (int)a.chunks.size(), (float)c.second});
				a.chunks.push_back({c.first, (int)v.size() - 1});
			}
		}
	}

	void calc_similarities_for_article(int id, vector<pair<int, float>>& similarities) const {
		unordered_map<int, float> similarity;
		for(auto& cp : articles.at(id).chunks) {
			auto it2 = chunk_to_docs.find(cp.chunk);
			if(it2 == chunk_to_docs.end())
				continue;

			auto it = idf.find(cp.chunk);
			float c_idf = it == idf.end() ? max_idf : it->second;
			float this_count = it2->second[cp.pos].count;

			for(auto& ci : it2->second)
				if(ci.doc_id != id) {
					//debug_check_valid_doc_id(ci.doc_id, "calc_similarities_for_article");
//					similarity[other_text] += c_idf;
//					similarity[other_text] += c_idf * count_intersection * count_intersection;
					similarity[ci.doc_id] += c_idf * min(this_count, ci.count);
//					similarity[other_text] += c_idf * (1 + logf(min(kvp.second, texts[other_text].chunks.at(kvp.first))));
				}
		}

		for(auto it = similarity.begin(); it != similarity.end(); ) {
			int other_id = it->first;
			float& sim = it->second;
			if(other_id >= 0) {
				sim /= max(articles[id].content_sum, articles.at(other_id).content_sum);
				if((sim < similarity_cutoff + similarity_penalty_for_same_site && articles[id].site == articles[other_id].site) ||
						fabs(articles[id].timestamp - articles[other_id].timestamp) > max_seconds_gap_between_articles_in_thread)
					sim = 0;
			}
			else {
				sim /= max(articles[id].content_sum, groups.at(-1 - other_id).content_sum);
				if((sim < similarity_cutoff + similarity_penalty_for_same_site && groups[-1 - other_id].sites.count(articles[id].site) != 0) ||
						fabs(articles[id].timestamp - groups[-1 - other_id].average_timestamp) > max_seconds_gap_between_articles_in_thread)
					sim = 0;
			}

			if(sim >= similarity_cutoff)
				++it;
			else
				it = similarity.erase(it);
		}

		similarities.insert(similarities.begin(), similarity.begin(), similarity.end());
		sort(similarities.begin(), similarities.end(), [](pair<int, float>& p1, pair<int, float>& p2) { return p1.second > p2.second; });
		if((int)similarities.size() > max_results)
			similarities.resize(max_results);
	}

	int& chunk_pos(const chunk_info& ci, hash_t chunk) {
		if(ci.doc_id >= 0)
			return articles.at(ci.doc_id).chunks.at(ci.pos_in_article).pos;

		//if(groups[-1 - ci.doc_id].chunk_pos.count(chunk) == 0)
		//	throw runtime_error("g" + to_string(-1 - ci.doc_id) + " has no chunk" + to_string(chunk));

		return groups[-1 - ci.doc_id].chunk_pos.at(chunk);
	}

	void remove_chunk_info(vector<chunk_info>& cis, hash_t chunk, int pos) {
		if(pos != (int)cis.size() - 1) {
			auto& ci = cis.at(pos);
			ci = cis.back();
			chunk_pos(ci, chunk) = pos;
		}

		cis.pop_back();
	}

	void update_group_id(int gid) {
		for(auto& kvp : groups[gid].chunk_pos)
			chunk_to_docs[kvp.first][kvp.second].doc_id = -1 - gid;
	}

	int new_group(int aid1, int aid2) {
		int new_group_id = (int)groups.size();
		//cerr << "a" << aid1 << " + a" << aid2 << " = g" << new_group_id << endl;
		groups.resize(groups.size() + 1);
		auto& g = groups.back();
		auto& t1 = articles[aid1];
		for(auto& cp : t1.chunks)
			g.chunk_pos.emplace(cp.chunk, cp.pos);

		g.content_sum = t1.content_sum;
		g.texts.push_back(aid1);
		g.sites.insert(t1.site);
		g.average_timestamp = t1.timestamp;
		update_group_id(new_group_id);
		add_article_to_group(aid2, new_group_id);
		t1.gid = new_group_id;
		return new_group_id;
	}

	/*
	void debug_check_article_chunks_in_index(int aid) const {
		auto& a = articles[aid];
		if(a.gid != -1)
			throw runtime_error("a" + to_string(aid) + " in g" +  to_string(a.gid));

		for(int p = 0; p < (int)a.chunks.size(); ++p) {
			auto& cp = a.chunks[p];
			auto& cis = chunk_to_docs.at(cp.chunk);
			auto& ci = cis.at(cp.pos);
			if(ci.pos_in_article != p)
				throw runtime_error(to_string(aid) + ": " +  to_string(ci.pos_in_article) + " -> " + to_string(p));

			if(ci.doc_id != aid)
				throw runtime_error(to_string(ci.doc_id) + " != " + to_string(aid));
		}
	}*/

	/*
	void debug_check_valid_doc_id(int doc_id, const char *src) const {
		if(doc_id >= (int)articles.size())
			throw runtime_error("Wrong aid " + to_string(doc_id) + " -" + string(src));
		else if(doc_id < 0 && -1 - doc_id >= (int)groups.size())
			throw runtime_error("Wrong gid " + to_string(-1 - doc_id) + " -" + string(src));
	}

	void debug_check_group_erased(int gid, const char *src) const {
		auto& g = groups.at(gid);
		for(auto& kvp : g.chunk_pos)
			for(auto& ci : chunk_to_docs.at(kvp.first))
				if(ci.doc_id == -1 - gid)
					throw runtime_error("Gid " + to_string(gid) + "/" + to_string(groups.size()) + " not erased -" + string(src));
	}

	void debug_check_all_doc_id_valid(const char *src) const {
		for(auto& kvp : chunk_to_docs)
			for(auto& ci : kvp.second)
				debug_check_valid_doc_id(ci.doc_id, src);
	}*/

	void add_article_to_group(int aid, int gid) {
		//cerr << "a" << aid << " -> g" << gid << endl;
		//debug_check_article_chunks_in_index(aid);

		auto& g = groups.at(gid);
		auto& a = articles.at(aid);
		float mult = (float)g.texts.size() / (g.texts.size() + 1);
		for(auto& kvp : g.chunk_pos) {
			auto& ci = chunk_to_docs[kvp.first][kvp.second];
			ci.count *= mult;
		}

		mult = 1.0f / (g.texts.size() + 1);
		for(auto& cp : a.chunks) {
			auto it = g.chunk_pos.find(cp.chunk);
			if(it == g.chunk_pos.end()) {
				g.chunk_pos.emplace(cp.chunk, cp.pos);
				auto& ci = chunk_to_docs[cp.chunk][cp.pos];
				ci.count *= mult;
				ci.doc_id = -1 - gid;
				ci.pos_in_article = -1;
			}
			else {
				auto& cis = chunk_to_docs[cp.chunk];
				auto& ci = cis[it->second];
				auto& ci2 = cis[cp.pos];
				ci.count += ci2.count * mult;
				remove_chunk_info(cis, cp.chunk, cp.pos);
			}
		}

		g.content_sum = (g.content_sum * g.texts.size() + a.content_sum) * mult;
		g.average_timestamp = (g.average_timestamp * g.texts.size() + a.timestamp) * mult;
		g.texts.push_back(aid);
		g.sites.insert(a.site);
		a.gid = gid;
	}

	bool are_groups_similar(int gid1, int gid2) const {
		const auto& g1 = groups[gid1], &g2 = groups[gid2];
		if(fabs(g1.average_timestamp - g2.average_timestamp) > max_seconds_gap_between_articles_in_thread)
			return false;

		float min_sim = similarity_cutoff;
		for(auto site : g1.sites)
			if(g2.sites.count(site) != 0) {
				min_sim += similarity_penalty_for_same_site;
				break;
			}

		float mult = 1.0f / max(g1.content_sum, g2.content_sum);
		float sim = 0;
		for(const auto& kvp : g1.chunk_pos) {
			auto it = g2.chunk_pos.find(kvp.first);
			if(it != g2.chunk_pos.end()) {
				auto it2 = idf.find(kvp.first);
				float c_idf = it2 == idf.end() ? max_idf : it2->second;
				const auto& v = chunk_to_docs.at(kvp.first);
				sim += c_idf * min(v[kvp.second].count, v[it->second].count) * mult;
				if(sim >= min_sim)
					return true;
			}
		}

		return false;
	}

	void join_groups(int group_to_keep, int group_to_remove) {
		auto& g = groups.at(group_to_keep), &gr = groups.at(group_to_remove);
		float mult = (float)g.texts.size() / (g.texts.size() + gr.texts.size());
		for(auto& kvp : g.chunk_pos) {
			auto& ci = chunk_to_docs[kvp.first][kvp.second];
			ci.count *= mult;
		}

		mult = (float)gr.texts.size() / (g.texts.size() + gr.texts.size());
		for(auto& kvp : gr.chunk_pos) {
			auto it = g.chunk_pos.find(kvp.first);
			if(it == g.chunk_pos.end()) {
				g.chunk_pos.insert(kvp);
				auto& ci = chunk_to_docs[kvp.first][kvp.second];
				ci.count *= mult;
				ci.doc_id = -1 - group_to_keep;
			}
			else {
				auto& cis = chunk_to_docs[kvp.first];
				auto& ci = cis[it->second];
				auto& ci2 = cis[kvp.second];
				ci.count += ci2.count * mult;
				remove_chunk_info(cis, kvp.first, kvp.second);
			}
		}

		mult = 1.0f / (g.texts.size() + gr.texts.size());
		g.content_sum = (g.content_sum * g.texts.size() + gr.content_sum * gr.texts.size()) * mult;
		g.average_timestamp = (g.average_timestamp * g.texts.size() + gr.average_timestamp * gr.texts.size()) * mult;
		g.texts.insert(g.texts.end(), gr.texts.begin(), gr.texts.end());
		g.sites.insert(gr.sites.begin(), gr.sites.end());
		for(int aid : gr.texts)
			articles[aid].gid = group_to_keep;

		//debug_check_group_erased(group_to_remove, "join_groups");
	}

	void update_groups(int new_id, const vector<pair<int, float>>& new_edges) {
		//cerr << "updating groups for a" << new_id << endl;
		int main_group;
		int id = new_edges[0].first;
		if(id >= 0)
			main_group = new_group(new_id, id);
		else {
			main_group = -1 - id;
			add_article_to_group(new_id, main_group);
		}

		vector<int> erased_groups;
		for(int i = 1; i < (int)new_edges.size(); ++i) {
			int id = new_edges[i].first;
			if(id >= 0) {
				if(article_group_similarity(id, main_group) >= similarity_cutoff)
					add_article_to_group(id, main_group);
			}
			else {
				int other_group = -1 - id;
				if(are_groups_similar(main_group, other_group)) {
					join_groups(main_group, other_group);
					erased_groups.push_back(other_group);
				}
			}
		}

		sort(erased_groups.rbegin(), erased_groups.rend());

		//for(int g : erased_groups)
		//	debug_check_group_erased(g, "doublecheck");

		for(int g : erased_groups) {
			//debug_check_group_erased(g, "1");

			int last_group = (int)groups.size() - 1;
			if(g < last_group) {
				groups[g] = groups[last_group];
				update_group_id(g);
				//debug_check_group_erased(last_group, "2");
			}

			groups.pop_back();
		}

		//debug_check_all_doc_id_valid("end update_groups");
	}
};


pair<unordered_map<hash_t, float>, float>* read_idf_file(const string& file_name);
unordered_set<hash_t> get_words(const vector<string>& text);
vector<float> calc_popularities(const vector<tuple<vector<string>, string, string, double>>& text_data, const string& idf);
vector<pair<pair<int, int>, float>> calc_similarities(const vector<tuple<vector<string>, string, string, double>>& article_data_text, const string& idf_file, float similarity_cutoff);
unordered_map<hash_t, pair<int, float>> get_chunks(const vector<string>& words, const string& idf_file);
