#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet featAlpha; //should be intialized outside
	Alphabet labelAlpha; // should be initialized outside
public:
	SparseParams sparselayer;
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts){

		// some model parameters should be initialized outside
		
		opts.labelSize = labelAlpha.size();
		sparselayer.initial(&featAlpha, opts.labelSize);

		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		sparselayer.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(sparselayer.W), "sparse.w");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */