#include "layer/Nothing.h"

using namespace layer;

static RegisterClassParameter<Nothing, LayerFactory> _register("Nothing");

Nothing::Nothing() : Layer3D(_register) {

}

Shape Nothing::compute_shape(const Shape& previous_shape) {
	parameter<size_t>("filter_number").set(previous_shape.dim(2));
	Layer3D::compute_shape(previous_shape);
	return Shape({_width, _height, _depth});
}


size_t Nothing::train_pass_number() const {
	return 1;
}

void Nothing::process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number) {
}

void Nothing::process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number) {
}

void Nothing::train(const std::string&, const std::vector<Spike>& input_spike, const Tensor<Time>&, std::vector<Spike>& output_spike) {
	output_spike = input_spike;
}

void Nothing::test(const std::string&, const std::vector<Spike>& input_spike, const Tensor<Time>&, std::vector<Spike>& output_spike) {
	output_spike = input_spike;
}

Tensor<float> Nothing::reconstruct(const Tensor<float>& t) const {
}

void Nothing::_exec(const std::vector<Spike>& input_spike, std::vector<Spike>& output_spike) {
}
