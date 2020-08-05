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
#include <sstream>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;


extern "C" {
char *whatlang_detect(const char *text, double *confidence);
void lang_code_free(char *lang_code);
}


const string lang_code_table = "data/iso-639-3_Code_Tables_20190408/iso-639-3_20190408.tab";
unordered_map<string, string> code_map;


void whatlang_wrapper_init() {
	ifstream infile(lang_code_table);
	string line;
	while (getline(infile, line)) {
	    istringstream iss(line);
        string substr;
        vector<string> fields;
        while(getline(iss, substr, '\t'))
        	fields.push_back(substr);

	    code_map[fields[0]] = fields[3];
	}

    code_map["pes"] = code_map["fas"];
    code_map["arb"] = code_map["ara"];
    code_map["cmn"] = code_map["zho"];
    code_map["azj"] = code_map["aze"];
    code_map["ydd"] = code_map["yid"];
    code_map["bho"] = "bho";
    code_map["mai"] = "mai";
    code_map["ilo"] = "ilo";
    code_map["ceb"] = "ceb";
    code_map["skr"] = "skr";

    //for (auto p: code_map)
    //	cout << p.first << " -> " << p.second << endl;
}


pair<double, string> whatlang_wrapper_detect(const string& s) {
	double confidence;
	char *lang_code_wl = whatlang_detect(s.c_str(), &confidence);
	string lang_code = lang_code_wl;
	lang_code_free(lang_code_wl);
	if(code_map.count(lang_code) == 0)
		lang_code = "None";
	else
		lang_code = code_map[lang_code];

	auto res = make_pair(confidence, lang_code);
	return res;
}

