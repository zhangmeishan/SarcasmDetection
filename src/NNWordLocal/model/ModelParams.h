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
	LSTM1Params left_lstm_project; //left lstm
	LSTM1Params right_lstm_project; //right lstm
	GatedPoolParam gatedpool_project;
	UniParams sent_hidden_project;
	UniParams olayer_linear; // output
public:
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts , AlignedMemoryPool *mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordwindow = 2 * opts.wordcontext + 1;
		opts.inputsize = opts.wordwindow * opts.wordDim;

		left_lstm_project.initial(opts.rnnhiddensize, opts.inputsize, mem);
		right_lstm_project.initial(opts.rnnhiddensize, opts.inputsize, mem);
		gatedpool_project.initial(opts.rnnhiddensize *2, opts.rnnhiddensize * 2,mem);
		sent_hidden_project.initial(opts.hiddensize, opts.rnnhiddensize * 2, mem);

		opts.labelSize = labelAlpha.size();
		olayer_linear.initial(opts.labelSize, opts.hiddensize, false, mem);

		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		left_lstm_project.exportAdaParams(ada);
		right_lstm_project.exportAdaParams(ada);
		gatedpool_project.exportAdaParams(ada);
		sent_hidden_project.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(words.E), "_words.E");
		checkgrad.add(&(left_lstm_project.output.W1), "left_lstm_project.output.W1");
		checkgrad.add(&(gatedpool_project._uni_gate_param.W), "gatedpool_project._uni_gate_param.W");
		checkgrad.add(&(gatedpool_project._uni_gate_param.b), "gatedpool_project._uni_gate_param.b");
		checkgrad.add(&(sent_hidden_project.W), "sent_hiden_project.W");
		checkgrad.add(&(sent_hidden_project.b), "sent_hiden_project.b");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */