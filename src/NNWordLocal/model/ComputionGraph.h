#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph {
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	vector<LookupNode> word_inputs;
	WindowBuilder word_window;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;

	vector<ConcatNode> concat_bilstm;
	GatedPoolBuilder gated_pooling;

	UniNode sent_hidden;
	LinearNode output;

public:
	ComputionGraph() : Graph() {
	}

	~ComputionGraph() {
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length) {

		word_inputs.resize(sent_length);
		word_window.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);

		concat_bilstm.resize(sent_length);
		gated_pooling.resize(sent_length);
	}

	inline void clear() {
		Graph::clear();
		word_inputs.clear();
		word_window.clear();
		left_lstm.clear();
		right_lstm.clear();

		concat_bilstm.clear();
		gated_pooling.clear();

	}

public:
	inline void initial(ModelParams& model, HyperParams& opts,  AlignedMemoryPool* mem = NULL) {
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx].init(model.words.nDim, opts.dropOut, mem);
			word_inputs[idx].setParam(&model.words);
			concat_bilstm[idx].init(opts.rnnhiddensize * 2, -1, mem);
		}
		word_window.init(model.words.nDim, opts.wordcontext, mem);
		left_lstm.init(&model.left_lstm_project, opts.dropOut, true, mem);
		right_lstm.init(&model.right_lstm_project, opts.dropOut, false, mem);
		gated_pooling.init(&model.gatedpool_project, mem);
		sent_hidden.init(opts.hiddensize, opts.dropOut, mem);
		sent_hidden.setParam(&model.sent_hidden_project);
		output.init(opts.labelSize, -1, mem);
		output.setParam(&model.olayer_linear);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false) {
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		// second step: build graph
		int seqsize = features.size();
		//forward
		// word-level neural networks
		const Feature& feature = features[seqsize - 1];
		int wordnum = feature.words.size();
		if (wordnum > max_sentence_length)
			wordnum = max_sentence_length;
		for (int idx = 0; idx < wordnum; idx++) {
			//input
			word_inputs[idx].forward(this, feature.words[idx]);
		}

		//windowlized
		word_window.forward(this, getPNodes(word_inputs, wordnum));

		left_lstm.forward(this, getPNodes(word_window._outputs, wordnum));
		right_lstm.forward(this, getPNodes(word_window._outputs, wordnum));

		for (int idx = 0; idx < wordnum; idx++) {
			//feed-forward
			concat_bilstm[idx].forward(this, &(left_lstm._hiddens[idx]), &(right_lstm._hiddens[idx]));
		}
		gated_pooling.forward(this, getPNodes(concat_bilstm, wordnum));
		sent_hidden.forward(this, &gated_pooling._output);
		output.forward(this, &sent_hidden);
	}

};

#endif /* SRC_ComputionGraph_H_ */