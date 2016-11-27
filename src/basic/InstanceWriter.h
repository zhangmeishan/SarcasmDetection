#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include <sstream>

using namespace std;

class InstanceWriter : public Writer
{
public:
	InstanceWriter(){}
	~InstanceWriter(){}
	int write(const Instance *pInstance)
	{
		if (!m_outf.is_open()) return -1;

		const vector<vector<string> > &words = pInstance->words;
		int seq_size = words.size();
		for (int i = 0; i < seq_size; i++){
			const string &label = pInstance->label;
			if (i < seq_size - 1)
				m_outf << "history " << endl;
			else if (pInstance->confidence < 0.0)
				m_outf << pInstance->label << endl;
			else
				m_outf << pInstance->label << " " << pInstance->confidence << endl;

		}
		m_outf << endl;
		return 0;

	}
};


#endif

