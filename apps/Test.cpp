#include "Experiment.h"
#include "dataset/ETH.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/SaveOutputNumpy.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/GrayScale.h"
#include "process/OnOffFilter.h"

int main(int argc, char** argv) {



	// Initialize experiment
	Experiment<DenseIntermediateExecution> experiment(argc, argv, "./", "tst", 0);

	// Load dataset
	const char* input_path_ptr = std::getenv("INPUT_PATH");
	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}
	std::string input_path(input_path_ptr);
	experiment.add_train<dataset::ETH>(input_path+"train_X.bin", input_path+"train_y.bin");
	experiment.add_test<dataset::ETH>(input_path+"test_X.bin", input_path+"test_y.bin");

	// Preprocessing
	experiment.push<process::GrayScale>();
	experiment.push<process::DefaultOnOffFilter>(7, 0.333, 0.666);
	experiment.push<process::FeatureScaling>();
	experiment.push<LatencyCoding>();

	// Convolutional layer
	auto& conv1 = experiment.push<layer::Convolution>(7, 7, 64);
	conv1.set_name("conv1");
	conv1.parameter<uint32_t>("epoch").set(200);
	conv1.parameter<float>("annealing").set(1);
	conv1.parameter<float>("min_th").set(4);
	conv1.parameter<float>("t_obj").set(0.75);
	conv1.parameter<float>("lr_th").set(0.1);
	conv1.parameter<bool>("wta_infer").set(true);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Gaussian>(0.5, 0.01);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10, 0.0001);
	conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(0.1, 1);

	// Pooling layer
	auto& pool1 = experiment.push<layer::Pooling>(8,8,8,8);
	pool1.set_name("pool1");

	// Activity analysis
	auto& conv1_activity = experiment.output<DefaultOutput>(conv1, 0.0, 1.0);
	conv1_activity.add_analysis<analysis::Coherence>();
	auto& pool1_activity = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_activity.add_analysis<analysis::Activity>();

	// SVM evaluation
	auto& pool1_out = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_out.add_analysis<analysis::Svm>();

	experiment.run(10000);
}
