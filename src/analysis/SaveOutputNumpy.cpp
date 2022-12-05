#include "analysis/SaveOutputNumpy.h"

using namespace analysis;

static RegisterClassParameter<SaveOutputNumpy, AnalysisFactory> _register("SaveOutputNumpy");

SaveOutputNumpy::SaveOutputNumpy() : UniquePassAnalysis(_register),
	_train_filename(), _test_filename(), _data_train(), _data_test(), _train_sample_cnt(0),
    _test_sample_cnt(0), _width(0), _height(0), _depth(0) {
	throw std::runtime_error("Unimplemented");
}

SaveOutputNumpy::SaveOutputNumpy(const std::string& train_filename, const std::string& test_filename) : UniquePassAnalysis(_register),
	_train_filename(train_filename), _test_filename(test_filename), _data_train(), _data_test(),
    _train_sample_cnt(0), _test_sample_cnt(0), _width(0), _height(0), _depth(0) {

}

void SaveOutputNumpy::resize(const Shape&) {
 
}

void SaveOutputNumpy::before_train() {
    _data_train.clear();
    _train_sample_cnt = 0;
}

void SaveOutputNumpy::process_train(const std::string& label, const Tensor<float>& sample) {
 	_width = sample.shape().dim(0);
	_height = sample.shape().dim(1);
	_depth = sample.shape().dim(2);

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			for(size_t z=0; z<_depth; z++) {
                _data_train.emplace_back(sample.at(x, y, z));
            }
        }
    }
    _train_sample_cnt += 1;
}

void SaveOutputNumpy::after_train() {
    const std::vector<long unsigned> shape{_train_sample_cnt, _width, _height, _depth};
    const bool fortran_order{false};
    npy::SaveArrayAsNumpy(_train_filename, fortran_order, shape.size(), shape.data(), _data_train);
}

void SaveOutputNumpy::before_test() {
    _data_test.clear();
    _test_sample_cnt = 0;
}

void SaveOutputNumpy::process_test(const std::string& label, const Tensor<float>& sample) {
 	_width = sample.shape().dim(0);
	_height = sample.shape().dim(1);
	_depth = sample.shape().dim(2);

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			for(size_t z=0; z<_depth; z++) {
                _data_test.emplace_back(sample.at(x, y, z));
            }
        }
    }
    _test_sample_cnt += 1;
}

void SaveOutputNumpy::after_test() {
    const std::vector<long unsigned> shape{_test_sample_cnt, _width, _height, _depth};
    const bool fortran_order{false};
    npy::SaveArrayAsNumpy(_test_filename, fortran_order, shape.size(), shape.data(), _data_test);
}
