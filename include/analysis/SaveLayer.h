#ifndef _ANALYSIS_SAVE_LAYER_H
#define _ANALYSIS_SAVE_LAYER_H

#include "Analysis.h"

namespace analysis {
	class SaveLayer : public NoPassAnalysis {

	public:
		SaveLayer();

		SaveLayer(const std::string& path);

		virtual void resize(const Shape& shape);

		virtual void process();

	private:
		std::string _path;
	};

}

#endif
