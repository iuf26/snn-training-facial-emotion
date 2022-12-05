#include "analysis/SaveOutputNumpy.h"

using namespace analysis;

static RegisterClassParameter<SaveOutputNumpy, AnalysisFactory> _register("SaveOutputNumpy");

SaveOutputNumpy::SaveOutputNumpy() : UniquePassAnalysis(_register),
	_train_filename(), _test_filename() {
	throw std::runtime_error("Unimplemented");
}

SaveOutputNumpy::SaveOutputNumpy(const std::string& train_filename, const std::string& test_filename) : UniquePassAnalysis(_register),
	_train_filename(train_filename), _test_filename(test_filename) {
    //TODO : Init test and train std::vector of shape (N_samples, N_neurons)
}

void SaveOutputNumpy::resize(const Shape&) {

}

void SaveOutputNumpy::before_train() {
    //TODO : Clear train vector
}

void SaveOutputNumpy::process_train(const std::string& label, const Tensor<float>& sample) {
    //TODO : Push sample in the vector
}

void SaveOutputNumpy::after_train() {
    //TODO: Save the vector with the cnpy lib
    //cnpy::npy_save(_train_filename,&_train_data[0],{Nz,Ny,Nx},"w");
    //how to get dimensions ? --> via sample.shape()
}

void SaveOutputNumpy::before_test() {

}

void SaveOutputNumpy::process_test(const std::string& label, const Tensor<float>& sample) {

}

void SaveOutputNumpy::after_test() {

}
