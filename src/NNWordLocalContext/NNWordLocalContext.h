/*
 * SparseDetector.h
 *
 *  Created on: Oct 23, 2016
 *      Author: DaPan
 */

#ifndef SRC_SparseDetector_H_
#define SRC_SparseDetector_H_


#include "N3L.h"
#include "Driver.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"
#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Detector {


public:
	unordered_map<string, int> m_feat_stats;
	unordered_map<string, int> m_word_stats;
	int m_maxseq_size;

public:
	Options m_options;

	Pipe m_pipe;

	Driver m_driver;


public:
	Detector(size_t memsize);
	virtual ~Detector();

public:

	int createAlphabet(const vector<Instance>& vecTrainInsts);
	void addTestAlphabet(const vector<Instance>& vecInsts);

	void extractDenseFeatures(vector<Feature>& features, const Instance* pInstance);
	void extractLinearFeatures(vector<string>& features, const Instance* pInstance);

	void convert2Example(const Instance* pInstance, Example& exam);
	void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
	int predict(const vector<Feature>& features, string& outputs);
	void test(const string& testFile, const string& outputFile, const string& modelFile);

	void writeModelFile(const string& outputModelFile);
	void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_SparseDetector_H_ */
