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
#include <string>
#include <sstream>
#include <cinttypes>
#include <sys/stat.h>

using namespace std;


inline bool file_exists(const string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


template <typename Out>
void split(const string &s, char delim, Out result) {
    istringstream iss(s);
    string item;
    while (getline(iss, item, delim))
        *result++ = item;
}


inline vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, back_inserter(elems));
    return elems;
}


template<class S>
string join(const S& pieces, const string& connection) {
	string res;
	int idx = 0;
	for(string p : pieces) {
		if(idx > 0)
			res += connection;

		res += p;
		++idx;
	}

	return res;
}


template<class T>
void serialize_value(istream &file, T& value, int64_t count = 1) {
	file.read((char*) &value, sizeof(T) * count);
}


template<class T>
void serialize_value(ostream &file, const T& value, int64_t count = 1) {
	file.write((char*) &value, sizeof(T) * count);
}


template<class T>
void serialize_sized(ostream& destination, const T& v) {
	uint64_t size = v.size();
	serialize_value(destination, size);
	if (size > 0)
		serialize_value(destination, v[0], v.size());
}


template<class T>
void serialize_sized(istream& source, T& v) {
	uint64_t size;
	serialize_value(source, size);
	if(!source)
		return;

	v.resize((size_t)size);
	if (size > 0)
		serialize_value(source, v[0], v.size());
}


typedef uint32_t hash_t;

inline uint32_t hash_uint32(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}


inline uint64_t hash_uint64(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
}


inline hash_t string_hash(const string& s) {
	hash_t h = 1;
	for(char c : s)
		h = hash_uint32(((hash_t)c) + h);

	return h;
}


string detect_language(const string& s, const string& source);
void load_nn_model(const string& file_name);
vector<float> process_sample(const vector<int>& sample, const string& model_file);
vector<vector<float>> process_samples(const vector<vector<int>>& samples, const string& model_file);
void libstemmer_init();
vector<string> stem_words(const vector<string>& words, bool english);
void get_words_with_count(const vector<string>& text, unordered_map<hash_t, int>& words);
void load_idf(const string& file_name);
