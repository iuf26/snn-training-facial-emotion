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

	Experiment<DenseIntermediateExecution> experiment(argc, argv, "eth80gs_multi");

	const char* input_path_ptr = std::getenv("INPUT_PATH");

	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	experiment.push<process::GrayScale>();
	experiment.push<process::DefaultOnOffFilter>(7, 1.0, 2.0);
	experiment.push<process::FeatureScaling>();
	experiment.push<LatencyCoding>();

	experiment.add_train<dataset::ETH>(input_path+"train_X.bin", input_path+"train_y.bin");
	experiment.add_test<dataset::ETH>(input_path+"test_X.bin", input_path+"test_y.bin");


	float t_obj1 = 0.75;
	auto& conv1 = experiment.push<layer::Convolution>(5, 5, 64);
	conv1.set_name("conv1");
	conv1.parameter<uint32_t>("epoch").set(200);
	conv1.parameter<float>("annealing").set(0.99);
	conv1.parameter<float>("min_th").set(2.0);
	conv1.parameter<float>("t_obj").set(t_obj1);
	conv1.parameter<float>("lr_th").set(0);
	conv1.parameter<bool>("wta_infer").set(false);
	//conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Gaussian>(0.5, 0.01);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
	//conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(0.1, 1);
	conv1.parameter<STDP>("stdp").set<stdp::Biological>(0.1, 0.2f);

	auto& pool1 = experiment.push<layer::Pooling>(4, 4, 4, 4);
	pool1.set_name("pool1");


	float t_obj2 = 0.85;
	auto& conv2 = experiment.push<layer::Convolution>(5, 5, 128);
	conv2.set_name("conv2");
	conv2.parameter<uint32_t>("epoch").set(200);
	conv2.parameter<float>("annealing").set(0.99);
	conv2.parameter<float>("min_th").set(2.0);
	conv2.parameter<float>("t_obj").set(t_obj2);
	conv2.parameter<float>("lr_th").set(0);
	conv2.parameter<bool>("wta_infer").set(false);
	//conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv2.parameter<Tensor<float>>("w").distribution<distribution::Gaussian>(0.5, 0.01);
	conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(15.0, 0.1);
	//conv2.parameter<STDP>("stdp").set<stdp::Multiplicative>(0.1, 1);
	conv2.parameter<STDP>("stdp").set<stdp::Biological>(0.1, 0.2f);

	auto& pool2 = experiment.push<layer::Pooling>(4, 4, 4, 4);
	pool2.set_name("pool2");

	
	///////////////////////////////
	/////////// OUTPUTS ///////////
	///////////////////////////////

	// pool1 : Activity
	auto& pool1_activity = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_activity.add_analysis<analysis::Activity>();

	// pool2 : Activity
	auto& pool2_activity = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_activity.add_analysis<analysis::Activity>();

	// pool2 : SVM evaluation
	
	auto& pool2_out = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_out.add_postprocessing<process::SumPooling>(12, 12);
	pool2_out.add_postprocessing<process::FeatureScaling>();
	pool2_out.add_analysis<analysis::Svm>();

	auto& pool2_out2 = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_out2.add_postprocessing<process::SumPooling>(8, 8);
	pool2_out2.add_postprocessing<process::FeatureScaling>();
	pool2_out2.add_analysis<analysis::Svm>();

	auto& pool2_out3 = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_out3.add_postprocessing<process::SumPooling>(4, 4);
	pool2_out3.add_postprocessing<process::FeatureScaling>();
	pool2_out3.add_analysis<analysis::Svm>();

	auto& pool2_out4 = experiment.output<TimeObjectiveOutput>(pool2, t_obj2);
	pool2_out4.add_postprocessing<process::SumPooling>(4, 4);
	pool2_out4.add_postprocessing<process::FeatureScaling>();
	pool2_out4.add_analysis<analysis::Svm>();

	experiment.run(10000);
}
