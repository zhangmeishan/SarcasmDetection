#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	Alphabet featAlpha; //should be intialized outside
	Alphabet labelAlpha; // should be initialized outside
public:
	LookupTable words; // should be initialized outside
	LookupTable history_words; // should be initialized outside
	LSTM1Params left_lstm_project; //left lstm
	LSTM1Params right_lstm_project; //right lstm
	GatedPoolParam local_gatedpool_project; //local gated pooling
	GatedPoolParam context_gatedpool_project; //context gated pooling
	UniParams sent_tanh_project; // sentence hidden
	UniParams olayer_linear; // output
public:
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || history_words.nVSize <= 0 || labelAlpha.size() <= 0) {
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordwindow = 2 * opts.wordcontext + 1;
		opts.inputsize = opts.wordwindow * opts.wordDim;
		int senthiddensize = opts.rnnhiddensize * 2+ words.nDim;

		left_lstm_project.initial(opts.rnnhiddensize, opts.inputsize, mem);
		right_lstm_project.initial(opts.rnnhiddensize, opts.inputsize, mem);
		local_gatedpool_project.initial(opts.rnnhiddensize * 2, opts.rnnhiddensize * 2, mem);
		context_gatedpool_project.initial(opts.wordDim, opts.wordDim, mem);
		sent_tanh_project.initial(opts.hiddensize, senthiddensize, mem);
		opts.labelSize = labelAlpha.size();
		olayer_linear.initial(opts.labelSize, opts.hiddensize, false, mem);

		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		history_words.exportAdaParams(ada);
		left_lstm_project.exportAdaParams(ada);
		right_lstm_project.exportAdaParams(ada);
		local_gatedpool_project.exportAdaParams(ada);
		context_gatedpool_project.exportAdaParams(ada);
		sent_tanh_project.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(sent_tanh_project.W), "sent_tanh_project.W");
		checkgrad.add(&(sent_tanh_project.b), "sent_tanh_project.b");

		checkgrad.add(&(context_gatedpool_project._uni_gate_param.W), "context_gatedpool_project.W");
		checkgrad.add(&(context_gatedpool_project._uni_gate_param.b), "context_gatedpool_project.b");
		checkgrad.add(&(local_gatedpool_project._uni_gate_param.W), "local_gatedpool_project.W");
		checkgrad.add(&(local_gatedpool_project._uni_gate_param.b), "local_gatedpool_project.b");

		checkgrad.add(&(right_lstm_project.cell.W1), "right_lstm_project.cell.W1");

		checkgrad.add(&(words.E), "_words.E");
		checkgrad.add(&(history_words.E), "_history_words.E");

	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */