#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	// must assign
	int wordcontext;
	int hiddensize;
	int rnnhiddensize;
	dtype dropOut;

	// must assign
	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization

	//auto generated
	int wordwindow;
	int wordDim;
	int inputsize;
	int labelSize;

public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequired(Options& opt){
		wordcontext = opt.wordcontext;
		hiddensize = opt.hiddenSize;
		rnnhiddensize = opt.rnnHiddenSize;
		dropOut = opt.dropProb;

		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */