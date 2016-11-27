/*
 * SparseDetector.cpp
 *
 *  Created on: Oct 23, 2016
 *      Author: DaPan
 */

#include "NNWordLocalContext.h"

#include "Argument_helper.h"

Detector::Detector(size_t memsize) : m_driver(memsize){
	// TODO Auto-generated constructor stub
	srand(0);
}

Detector::~Detector() {
	// TODO Auto-generated destructor stub
}

int Detector::createAlphabet(const vector<Instance>& vecInsts) {
	if (vecInsts.size() == 0) {
		std::cout << "training set empty" << std::endl;
		return -1;
	}
	cout << "Creating Alphabet..." << endl;

	m_maxseq_size = -1;
	int numInstance;

	m_driver._modelparams.labelAlpha.clear();
	// label alphabet and word statistics
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<vector<string> > &words = pInstance->words;
		const string &label = pInstance->label;

		int labelId = m_driver._modelparams.labelAlpha.from_string(label);

		int seq_size = pInstance->seqsize();
		if (seq_size > m_maxseq_size)
			m_maxseq_size = seq_size;
		for (int i = 0; i < seq_size; i++) {
			int wordLength = words[i].size();
			for (int j = 0; j < wordLength; ++j) {
				string curword = normalize_to_lowerwithdigit(words[i][j]);
				m_word_stats[curword]++;
			}

		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
	cout << "Label num: " << m_driver._modelparams.labelAlpha.size() << endl;


	m_driver._modelparams.labelAlpha.set_fixed_flag(true);

	if (m_options.linearfeatCat > 0) {
		cout << "Extracting linear features..." << endl;
		for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
			const Instance *pInstance = &vecInsts[numInstance];
			vector<string> linearfeat;
			extractLinearFeatures(linearfeat, pInstance);
			for (int i = 0; i < linearfeat.size(); i++)
				m_feat_stats[linearfeat[i]] ++;
		}
		m_feat_stats[unknownkey] = m_options.featCutOff + 1;
		cout << "Total feature num: " << m_feat_stats.size() << endl;
		m_driver._modelparams.featAlpha.initial(m_feat_stats, m_options.featCutOff);
		cout << "Remina feature num:" << m_driver._modelparams.featAlpha.size() << endl;
		m_driver._modelparams.featAlpha.set_fixed_flag(true);
	}
	return 0;
}

void Detector::addTestAlphabet(const vector<Instance>& vecInsts)
{
	cout << "Adding other word Alphabet..." << endl;
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<vector<string> > &words = pInstance->words;
		int seq_size = pInstance->seqsize();
		for (int i = 0; i < seq_size; ++i) {
			for (int j = 0; j < words[i].size(); j++) {
				string curword = normalize_to_lowerwithdigit(words[i][j]);
				if (!m_options.wordEmbFineTune)
					m_word_stats[curword]++;
			}
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
}



void Detector::extractDenseFeatures(vector<Feature>& features, const Instance * pInstance)
{
	features.clear();
	const vector<vector<string> >& words = pInstance->words;
	int seq_size = pInstance->seqsize();
	assert(seq_size < 3);

	//Local and Context sentence dense feature
	for (int sentidx = 0; sentidx < seq_size; sentidx++) {
		Feature feat;
		const vector<string>& curWords = words[sentidx];
		int wordnumber = curWords.size();
		for (int i = 0; i < wordnumber; i++)
			feat.words.push_back(normalize_to_lowerwithdigit(curWords[i]));

		features.push_back(feat);
	}
}

void Detector::extractLinearFeatures(vector<string>& feat, const Instance* pInstance) {
	feat.clear();

	const vector<vector<string> >& words = pInstance->words;
	int seq_size = pInstance->seqsize();
	assert(seq_size < 3);
	//Current sent linear feature
	const vector<string>& lastWords = words[seq_size - 1];
	int wordnumber = lastWords.size();
	string strfeat = "", curWord = "", preWord = "", pre2Word = "";
	for (int i = 0; i < wordnumber; i++) {
		curWord = normalize_to_lowerwithdigit(lastWords[i]);
		strfeat = "F1U=" + curWord;
		feat.push_back(strfeat);
		preWord = i - 1 >= 0 ? lastWords[i - 1] : nullkey;
		strfeat = "F2B=" + preWord + seperateKey + curWord;
		feat.push_back(strfeat);
		pre2Word = i - 2 >= 0 ? lastWords[i - 2] : nullkey;
		strfeat = "F3T=" + pre2Word + seperateKey + preWord + seperateKey + curWord;
		feat.push_back(strfeat);
	}

	//History feature
	if (m_options.linearfeatCat > 1 && seq_size == 2) {
		const vector<string>& historyWords = words[seq_size - 2];
		wordnumber = historyWords.size();
		for (int i = 0; i < wordnumber; i++) {
			strfeat = "F4U=" + historyWords[i];
			feat.push_back(strfeat);
		}
	}
}

void Detector::convert2Example(const Instance* pInstance, Example& exam) {
	exam.clear();

	const string &instlabel = pInstance->label;
	const Alphabet &labelAlpha = m_driver._modelparams.labelAlpha;

	int labelnum = labelAlpha.size();
	for (int i = 0; i < labelnum; i++) {
		string str = labelAlpha.from_id(i);
		if (instlabel.compare(str) == 0)
			exam.m_labels.push_back(1.0);
		else
			exam.m_labels.push_back(0.0);
	}

	//dense feature
	extractDenseFeatures(exam.m_densefeatures, pInstance);

	//linear feature
	if (m_options.linearfeatCat > 0)
		extractLinearFeatures(exam.m_linearfeatures, pInstance);

}

void Detector::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		Example curExam;
		convert2Example(pInstance, curExam);
		vecExams.push_back(curExam);

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
}

void Detector::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	static vector<Instance> decodeInstResults;
	static Instance curDecodeInst;
	bool bCurIterBetter = false;

	m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

	std::cout << "Training example number: " << trainInsts.size() << std::endl;
	std::cout << "Dev example number: " << trainInsts.size() << std::endl;
	std::cout << "Test example number: " << trainInsts.size() << std::endl;

	createAlphabet(trainInsts);
	addTestAlphabet(devInsts);
	addTestAlphabet(testInsts);

	m_word_stats[unknownkey] = m_options.wordCutOff + 1;
	cout << "Total word num: " << m_word_stats.size() << endl;
	m_driver._modelparams.wordAlpha.initial(m_word_stats, m_options.wordCutOff);
	m_driver._modelparams.wordAlpha.set_fixed_flag(true);
	cout << "Remain word num:" << m_driver._modelparams.wordAlpha.size() << endl;

	vector<Example> trainExamples, devExamples, testExamples;

	std::cout << "Instance convert to example... " << std::endl;
	initialExamples(trainInsts, trainExamples);
	initialExamples(devInsts, devExamples);
	initialExamples(testInsts, testExamples);

	if (m_options.wordFile != "") {
		m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordFile, m_options.wordEmbFineTune);
	}
	else{
		m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordEmbSize, m_options.wordEmbFineTune);
	}

	m_driver._hyperparams.setRequired(m_options);
	m_driver.initial(m_maxseq_size);



	dtype bestDIS = 0;

	int inputSize = trainExamples.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test;
	static vector<Example> subExamples;
	int devNum = devExamples.size(), testNum = testExamples.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;

		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
			subExamples.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			for (int idy = start_pos; idy < end_pos; idy++) {
				subExamples.push_back(trainExamples[indexes[idy]]);
			}

			int curUpdateIter = iter * batchBlock + updateIter;
			dtype cost = m_driver.train(subExamples, curUpdateIter);

			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
				//m_driver.checkgrad(subExamples, curUpdateIter + 1);
				std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
			}
			m_driver.updateModel();

		}

		if (devNum > 0) {
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			for (int idx = 0; idx < devExamples.size(); idx++) {
				string result_label;
				predict(devExamples[idx].m_densefeatures, result_label);

				devInsts[idx].Evaluate(result_label, metric_dev);

				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_label);
					decodeInstResults.push_back(curDecodeInst);
				}
			}

			std::cout << "dev:" << std::endl;
			metric_dev.print();

			if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idx = 0; idx < testExamples.size(); idx++) {
					string result_label;
					predict(testExamples[idx].m_densefeatures, result_label);

					testInsts[idx].Evaluate(result_label, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_label);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}



			if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS) {
				std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
				bestDIS = metric_dev.getAccuracy();
				writeModelFile(modelFile);
			}

		}
		// Clear gradients
	}
}

int Detector::predict(const vector<Feature>& features, string& output) {
	int labelIdx;
	m_driver.predict(features, labelIdx);
	output = m_driver._modelparams.labelAlpha.from_id(labelIdx, nullkey);

	if (output == nullkey)
		std::cout << "predict error" << std::endl;
	return 0;
}

void Detector::test(const string& testFile, const string& outputFile, const string& modelFile) {
	loadModelFile(modelFile);
	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts);

	vector<Example> testExamples;
	initialExamples(testInsts, testExamples);

	int testNum = testExamples.size();
	vector<Instance> testInstResults;
	Metric metric_test;
	metric_test.reset();
	for (int idx = 0; idx < testExamples.size(); idx++) {
		string result_label;
		predict(testExamples[idx].m_densefeatures, result_label);
		testInsts[idx].Evaluate(result_label, metric_test);
		Instance curResultInst;
		curResultInst.copyValuesFrom(testInsts[idx]);
		curResultInst.assignLabel(result_label);
		testInstResults.push_back(curResultInst);
	}
	std::cout << "test:" << std::endl;
	metric_test.print();

	m_pipe.outputAllInstances(outputFile, testInstResults);

}


void Detector::loadModelFile(const string& inputModelFile) {

}

void Detector::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "", wordEmbFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;
	int memsize = 1;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
	ah.new_named_int("mem", "memsize", "named_int", "memory allocated for tensor nodes", memsize);

	ah.process(argc, argv);

	Detector detector(memsize);
	detector.m_pipe.max_sentense_size = ComputionGraph::max_sentence_length;
	if (bTrain) {
		detector.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		detector.test(testFile, outputFile, modelFile);
	}

	//test(argv);
	//ah.write_values(std::cout);
}
