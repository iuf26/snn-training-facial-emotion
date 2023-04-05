#include "process/Norm.h"

using namespace process;


static RegisterClassParameter<Normalizing, ProcessFactory> _register_1("Normalizing");

Normalizing::Normalizing() : UniquePassProcess(_register_1), _max(1) {
}

Normalizing::Normalizing(float max) : UniquePassProcess(_register_1), _max(max) {
}

Shape Normalizing::compute_shape(const Shape& shape) {
	return shape;
}

void Normalizing::process_train(const std::string&, Tensor<float>& sample) {
	size_t size = sample.shape().product();
	for(size_t i=0; i<size; i++) {
		sample.at_index(i) = sample.at_index(i) / _max;
	}

}

void Normalizing::process_test(const std::string&, Tensor<float>& sample) {
	size_t size = sample.shape().product();
	for(size_t i=0; i<size; i++) {
		sample.at_index(i) = sample.at_index(i) / _max;
	}
}
