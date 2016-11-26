/*
 * Driver.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"


//A native neural network classfier using only linear features

class Driver{
public:
	Driver(size_t memsize) : aligned_mem(memsize) {
		_pcg = NULL;
	}

	~Driver() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	ComputionGraph *_pcg;  // build neural graphs
	ModelParams _modelparams;  // model parameters
	HyperParams _hyperparams;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;  // model update
	AlignedMemoryPool aligned_mem;


public:
	inline void initial(int maxseq_size) {
		if (!_hyperparams.bValid()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams, &aligned_mem)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_modelparams.exportModelParams(_ada);
		_modelparams.exportCheckGradParams(_checkgrad);

		_hyperparams.print();

		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length, maxseq_size);
		_pcg->initial(_modelparams, _hyperparams, &aligned_mem);

		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
	}


	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];
			//forward
			_pcg->forward(example.m_densefeatures, true); 

			//loss function
			int seq_size = example.m_densefeatures.size();
			int wordnum = example.m_densefeatures[seq_size - 1].words.size();
			cost += _modelparams.loss.loss(&_pcg->output, example.m_labels, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature> densefeatures, int& results) {
		_pcg->forward(densefeatures);
		_modelparams.loss.predict(&_pcg->output, results);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_densefeatures); //forward here

		int seq_size = example.m_densefeatures.size();

		dtype cost = 0.0;

		cost += _modelparams.loss.cost(&_pcg->output, example.m_labels, 1);

		return cost;
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void updateModel() {
		_ada.update();
		//_ada.update(5.0);
	}

	void writeModel();

	void loadModel();



private:
	inline void resetEval() {
		_eval.reset();
	}


	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_Driver_H_ */
