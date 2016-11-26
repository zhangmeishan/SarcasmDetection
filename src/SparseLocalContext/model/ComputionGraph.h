#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	SparseNode output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int typeNum){
		
	}

	inline void clear(){
		Graph::clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts){
		output.setParam(&model.sparselayer);
		output.init(opts.labelSize,-1);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<string>& features, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		// second step: build graph
		//forward
		output.forward(this, features);
	}

};

#endif /* SRC_ComputionGraph_H_ */