#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph {
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	vector<vector<LookupNode> > word_inputs;
	WindowBuilder word_window;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;

	vector<ConcatNode> concat_bilstm;
	GatedPoolBuilder local_gated_pooling;
	GatedPoolBuilder context_gated_pooling;

	Node padding;
	ConcatNode concat_local_context;
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
	inline void createNodes(int sent_length, int maxseq_size) {

		resizeVec(word_inputs, maxseq_size, sent_length);
		word_window.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);

		concat_bilstm.resize(sent_length);
		local_gated_pooling.resize(sent_length);
		context_gated_pooling.resize(sent_length);

	}

	inline void clear() {
		Graph::clear();
		clearVec(word_inputs);
		word_window.clear();
		left_lstm.clear();
		right_lstm.clear();
		concat_bilstm.clear();
		local_gated_pooling.clear();
		context_gated_pooling.clear();
	}


public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL) {
		int seq_size = word_inputs.size();

		for (int i = 0; i < seq_size; i++) {
			for (int idx = 0; idx < word_inputs[i].size(); idx++) {
				word_inputs[i][idx].init(model.words.nDim, opts.dropOut, mem);
				word_inputs[i][idx].setParam(&model.words);
				if ( i == seq_size -1 )
					concat_bilstm[idx].init(opts.rnnhiddensize * 2, -1, mem);
			}
		}
		word_window.init(model.words.nDim, opts.wordcontext, mem);
		left_lstm.init(&model.left_lstm_project, opts.dropOut, true, mem);
		right_lstm.init(&model.right_lstm_project, opts.dropOut, false, mem);

		local_gated_pooling.init(&model.local_gatedpool_project, mem);
		context_gated_pooling.init(&model.context_gatedpool_project, mem);

		concat_local_context.init(opts.rnnhiddensize * 2 + model.words.nDim, -1, mem);
		sent_hidden.init(opts.hiddensize, opts.dropOut, mem);
		sent_hidden.setParam(&model.sent_tanh_project);
		output.init(opts.labelSize, -1, mem);
		output.setParam(&model.olayer_linear);

		padding.init(opts.wordDim, -1, mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false) {
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		// second step: build graph
		int seq_size = features.size();
		//forward
		// word-level neural networks
		for (int i = 0; i < seq_size; i++) {

			const Feature& feature = features[i];
			int wordnum = feature.words.size();
			if (wordnum > max_sentence_length)
				wordnum = max_sentence_length;
			for (int idx = 0; idx < wordnum; idx++) {
				//input
				word_inputs[i][idx].forward(this, feature.words[idx]);
			}
			if (i == seq_size - 1) {
				//windowlized
				word_window.forward(this, getPNodes(word_inputs[i], wordnum));
				left_lstm.forward(this, getPNodes(word_window._outputs, wordnum));
				right_lstm.forward(this, getPNodes(word_window._outputs, wordnum));

				for (int idx = 0; idx < wordnum; idx++) {
					//feed-forward
					concat_bilstm[idx].forward(this, &(left_lstm._hiddens[idx]), &(right_lstm._hiddens[idx]));
				}
				local_gated_pooling.forward(this, getPNodes(concat_bilstm, wordnum));
			}

			else {
				context_gated_pooling.forward(this, getPNodes(word_inputs[i], wordnum));
			}
		}

		if (seq_size == 1)
			concat_local_context.forward(this, &padding, &local_gated_pooling._output);
		else
			concat_local_context.forward(this, &context_gated_pooling._output, &local_gated_pooling._output);
		sent_hidden.forward(this, &concat_local_context);
		output.forward(this, &sent_hidden);
	}

};

#endif /* SRC_ComputionGraph_H_ */