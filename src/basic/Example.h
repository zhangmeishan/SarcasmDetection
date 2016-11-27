/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include "MyLib.h"

using namespace std;
struct Feature {
public:
	vector<string> words;
public:
	Feature() {
	}

	virtual ~Feature() {
	
	}

	void clear() {
		words.clear();
	}
};

class Example {

public:
	vector<dtype> m_labels;
	vector<Feature> m_densefeatures;
	vector<string> m_linearfeatures;
public:
	Example(){

	}
	virtual ~Example(){

	}

	void clear(){
		m_labels.clear();
		m_densefeatures.clear();
	}


};


#endif /* SRC_EXAMPLE_H_ */
