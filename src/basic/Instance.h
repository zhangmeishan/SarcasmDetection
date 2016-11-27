#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "N3L.h"
#include "Metric.h"

using namespace std;

class Instance {
public:
	Instance() {
	}
	~Instance() {
	}

	int seqsize() const {
		return words.size();
	}


	int wordnum() const{
		return words[seqsize() - 1].size();
	}

	void clear() {
		label = "";
		for (int i = 0; i < seqsize(); i++) {
			words[i].clear();
		}
		words.clear();
		confidence = -1.0;
	}

	void allocate(int seq_size) {
		clear();
		label = "";
		words.resize(seq_size);
		confidence = -1.0;
	}

	void copyValuesFrom(const Instance& anInstance) {
		allocate(anInstance.seqsize());
		for (int i = 0; i < anInstance.seqsize(); i++) {
			for (int j = 0; j < anInstance.words[i].size(); j++)
				words[i].push_back(anInstance.words[i][j]);
		}
		label = anInstance.label;
	}

	void assignLabel(const string& resulted_label) {
		label = resulted_label;
	}

	void assignLabel(const string& resulted_label, dtype resulted_confidence){
		label = resulted_label;
		confidence = resulted_confidence;
	}

	void Evaluate(const string& resulted_label, Metric& eval) const {
		if (resulted_label.compare(label) == 0)
			eval.correct_label_count++;
		eval.overall_label_count++;

	}


public:
	string label;
	vector<vector<string> > words;
	dtype confidence;
};

#endif

