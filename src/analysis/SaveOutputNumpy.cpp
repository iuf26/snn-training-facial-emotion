#include "analysis/SaveOutputNumpy.h"

using namespace analysis;

static RegisterClassParameter<SaveOutputNumpy, AnalysisFactory> _register("SaveOutputNumpy");

SaveOutputNumpy::SaveOutputNumpy() : UniquePassAnalysis(_register),
	_train_filename(), _test_filename(), _data_train(), _data_test(), _train_sample_cnt(0), _test_sample_cnt(0) {
	throw std::runtime_error("Unimplemented");
}

SaveOutputNumpy::SaveOutputNumpy(const std::string& train_filename, const std::string& test_filename) : UniquePassAnalysis(_register),
	_train_filename(train_filename), _test_filename(test_filename), _data_train(), _data_test(), _train_sample_cnt(0), _test_sample_cnt(0) {

}

void SaveOutputNumpy::resize(const Shape&) {
 
}

void SaveOutputNumpy::before_train() {
    _data_train.clear();
    _train_sample_cnt = 0;
}

void SaveOutputNumpy::process_train(const std::string& label, const Tensor<float>& sample) {
 	size_t width = sample.shape().dim(0);
	size_t height = sample.shape().dim(1);
	size_t depth = sample.shape().dim(2);

	for(size_t x=0; x<width; x++) {
		for(size_t y=0; y<height; y++) {
			for(size_t z=0; z<depth; z++) {
                _data_tain.emplace_back(sample.at(x, y, z));
            }
        }
    }
    _train_sample_cnt += 1;
}

void SaveOutputNumpy::after_train() {
    int n_neurons = std::static_cast<int>(_data_train.size()/_train_sample_cnt);
    const std::vector<long unsigned> shape{_train_sample_cnt, n_neurons};
    const bool fortran_order{false};
    npy::SaveArrayAsNumpy(_train_filename, fortran_order, shape.size(), shape.data(), _data_train);
}

void SaveOutputNumpy::before_test() {
    _data_test.clear();
    _test_sample_cnt = 0;
}

void SaveOutputNumpy::process_test(const std::string& label, const Tensor<float>& sample) {
 	size_t width = sample.shape().dim(0);
	size_t height = sample.shape().dim(1);
	size_t depth = sample.shape().dim(2);

	for(size_t x=0; x<width; x++) {
		for(size_t y=0; y<height; y++) {
			for(size_t z=0; z<depth; z++) {
                _data_test.emplace_back(sample.at(x, y, z));
            }
        }
    }
    _test_sample_cnt += 1;
}

void SaveOutputNumpy::after_test() {
    int n_neurons = std::static_cast<int>(_data_test.size()/_test_sample_cnt);
    const std::vector<long unsigned> shape{_test_sample_cnt, n_neurons};
    const bool fortran_order{false};
    npy::SaveArrayAsNumpy(_test_filename, fortran_order, shape.size(), shape.data(), _data_test);
}
