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
#include <memory>
#include <sstream>

#include "3rdparty/fasttext/src/fasttext.h"

using namespace fasttext;
using namespace std;


unique_ptr<FastText> language_detection_model;

void fasttext_init() {
	language_detection_model.reset(new FastText);
	language_detection_model->loadModel("data/lid.176.ftz");
	//cout << language_detection_model->getDimension() << endl;
	//cout << "Everything's fine 🎂" << endl;
}


vector<pair<real, string>> fasttext_detect_languages(const string& s, int num_preds) {
	vector<pair<real, string>> predictions;
	istringstream ss(s);
	language_detection_model->predictLine(ss, predictions, num_preds, 0);
	for(auto& p : predictions)
		p.second = p.second.substr(9);

	return predictions;
}
