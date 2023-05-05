#include "analysis/SaveLayer.h"
#include "Experiment.h"

#include "dep/npy.hpp"
#include "dep/filesystem.hpp"

using namespace analysis;

static RegisterClassParameter<SaveLayer, AnalysisFactory> _register("SaveLayer");

SaveLayer::SaveLayer() : NoPassAnalysis(_register), _path("./") {

}

SaveLayer::SaveLayer(const std::string& path) : NoPassAnalysis(_register), _path(path) {

}

void SaveLayer::resize(const Shape&) {

}

void SaveLayer::process() {
	experiment().log() << "===SaveLayer===" << std::endl;

	if(experiment().process_at(layer_index()).has_parameter("w") && experiment().process_at(layer_index()).is_type<Tensor<float>>("w")) {
		const Tensor<float>& w = experiment().process_at(layer_index()).parameter<Tensor<float>>("w").get();
		const Tensor<float>& th = experiment().process_at(layer_index()).parameter<Tensor<float>>("th").get();

		if(w.shape().number() == 4) {

			size_t width = w.shape().dim(0);
			size_t height = w.shape().dim(1);
			size_t in_depth = w.shape().dim(2);
			size_t out_depth = w.shape().dim(3);

			std::vector<float> weights;
			for(size_t i=0; i<width; i++) {
				for(size_t j=0; j<height; j++) {
					for(size_t k=0; k<in_depth; k++) {
						for(size_t l=0; l<out_depth; l++) {
							weights.emplace_back(w.at(i, j, k, l));
						}
					}
				}
			}

			std::vector<float> thresholds;
			for(size_t i=0; i<out_depth; i++) {
				thresholds.emplace_back(th.at(i));
			}
			
			const bool fortran_order{false};

			ghc::filesystem::create_directories(_path);
    		
			const std::vector<long unsigned> shape_weights{width, height, in_depth, out_depth};
			npy::SaveArrayAsNumpy(_path + "/weights.npy", fortran_order, shape_weights.size(), shape_weights.data(), weights);    

    		const std::vector<long unsigned> shape_thresholds{out_depth};
			npy::SaveArrayAsNumpy(_path + "/thresholds.npy", fortran_order, shape_thresholds.size(), shape_thresholds.data(), thresholds);

			experiment().log() << "Model parameters saved at " + _path << std::endl;
		}
		else {
			experiment().log() << "Incompatible w shape." << std::endl;
		}

	}
	else {
		experiment().log() << "No w parameter in this layer." << std::endl;
	}

	experiment().log() << std::endl;
}
