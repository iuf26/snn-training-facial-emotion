#include "analysis/SaveOutputNumpy.h"

using namespace analysis;

static RegisterClassParameter<SaveOutputNumpy, AnalysisFactory> _register("SaveOutputNumpy");

SaveOutputNumpy::SaveOutputNumpy() : UniquePassAnalysis(_register),
	_train_filename(), _test_filename() {
	throw std::runtime_error("Unimplemented");
}

SaveOutputNumpy::SaveOutputNumpy(const std::string& train_filename, const std::string& test_filename) : UniquePassAnalysis(_register),
	_train_filename(train_filename), _test_filename(test_filename), _data_train(), _data_test() {

}

void SaveOutputNumpy::resize(const Shape&) {
 
}

void SaveOutputNumpy::before_train() {
    _data_train.clear();
}

void SaveOutputNumpy::process_train(const std::string& label, const Tensor<float>& sample) {
	std::vector<float> timestamps(sample.shape().product());
    _TensorToVector(sample, timestamps);
    _data_train.emplace_back(timestamps);
}

void SaveOutputNumpy::after_train() {
    size_t n_samples = _data_train.size();
    size_t n_neurons = _data_train[0].size();
    cnpy::npy_save(_train_filename, &_data_train[0], {n_samples, n_neurons}, "w");
}

void SaveOutputNumpy::before_test() {
    _data_test.clear();
}

void SaveOutputNumpy::process_test(const std::string& label, const Tensor<float>& sample) {
	std::vector<float> timestamps(sample.shape().product());
    _TensorToVector(sample, timestamps);
    _data_test.emplace_back(timestamps);
}

void SaveOutputNumpy::after_test() {
    size_t n_samples = _data_train.size();
    size_t n_neurons = _data_train[0].size();
    cnpy::npy_save(_train_filename, &_data_train[0], {n_samples, n_neurons}, "w");
}

void SaveOutputNumpy::_TensorToVector(const Tensor<float>& in, std::vector<float>& out) {
 	size_t width = in.shape().dim(0);
	size_t height = in.shape().dim(1);
	size_t depth = in.shape().dim(2);

	for(size_t x=0; x<width; x++) {
		for(size_t y=0; y<height; y++) {
			for(size_t z=0; z<depth; z++) {
                out.emplace_back(in.at(x, y, z));
            }
        }
    }   
}