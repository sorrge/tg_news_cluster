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
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>

using namespace std;


vector<pair<float, string>> fasttext_detect_languages(const string& s, int num_preds);
pair<double, string> whatlang_wrapper_detect(const string& s);


unordered_map<string, unordered_map<string, double>> sources_languages;


void load_sources_languages(const unordered_map<string, unordered_map<string, double>>& sl) {
	sources_languages = sl;
}


string detect_language(const string& s, const string& source) {
	auto wl_res = whatlang_wrapper_detect(s);
	auto ft_res = fasttext_detect_languages(s, 3);
	unordered_map<string, double> res;
	auto it = sources_languages.find(source);
	if(it != sources_languages.end())
		res = it->second;

	for(auto& p: ft_res)
		res[p.second] += p.first;

	res[wl_res.second] += sqrt(wl_res.first);
	string best_language = res.begin()->first;
	for(auto& p : res)
		if(p.second > res[best_language])
			best_language = p.first;

	return best_language;
}


vector<string> detect_languages(const vector<string>& s, const vector<string>& sources) {
	vector<string> res(s.size());
#pragma omp parallel for
	for(int i = 0; i < (int)s.size(); ++i)
		res[i] = detect_language(s[i], sources[i]);

	return res;
}

