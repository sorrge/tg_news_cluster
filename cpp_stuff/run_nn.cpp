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
#include <fstream>
#include <vector>
#include <unordered_map>
#include <mutex>

#include <fdeep/fdeep.hpp>

#include "common.h"

using namespace fdeep;
using namespace std;


const int input_size = 7000;


unordered_map<string, model> loaded_models;
mutex models_mutex;


inline void null_logger(const string& str) {}


void load_nn_model(const string& file_name) {
	{
		lock_guard<mutex> guard(models_mutex);
		if(loaded_models.count(file_name) != 0)
			return;
	}

	model m = fdeep::load_model(file_name, false, null_logger);

	{
		lock_guard<mutex> guard(models_mutex);
		loaded_models.emplace(file_name, m);
	}
}


void parse_text(const std::string& text, tensor5& res) {
	for(auto word : split(text, ' '))
		res.set(0, 0, 0, 0, stoi(word), 1.0f);
}


void set_vector(const vector<int>& idx, tensor5& onehot) {
	for(int i : idx)
		onehot.set(0, 0, 0, 0, i, 1.0f);
}


vector<float> process_sample(const vector<int>& sample, const string& model_file) {
	const model& keras_model = loaded_models.at(model_file);
	tensor5s input = {tensor5(shape5(1, 1, 1, 1, input_size), 0.0f)};
	set_vector(sample, input[0]);
    const auto result = keras_model.predict(input);
    return *result[0].as_vector();
}


vector<vector<float>> process_samples(const vector<vector<int>>& samples, const string& model_file) {
	const model& keras_model = loaded_models.at(model_file);
    int batch_size = 128;
	vector<tensor5s> batch;
	vector<vector<float>> all_results(samples.size());
    for (int i = 0; i < (int)samples.size(); i += batch_size) {
    	int s = std::min(batch_size, (int)samples.size() - i);
    	batch.resize(s);
    	for(int j = 0; j < s; ++j) {
    		batch[j] = {tensor5(shape5(1, 1, 1, 1, input_size), 0.0f)};
    		set_vector(samples[i + j], batch[j][0]);
    	}

        const auto result = keras_model.predict_multi(batch, true);
    	for(int j = 0; j < s; ++j)
    		all_results[i + j] = *result[j][0].as_vector();
    }

    return all_results;
}
