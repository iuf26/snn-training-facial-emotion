#include "Experiment.h"
#include "dataset/Cifar.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/SaveOutputJson.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/GrayScale.h"
#include "process/OnOffFilter.h"

int main(int argc, char** argv) {
	Experiment<DenseIntermediateExecution> experiment(argc, argv, "cifar10");

	const char* input_path_ptr = std::getenv("INPUT_PATH");

	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	experiment.push<process::GrayScale>();
	experiment.push<process::DefaultOnOffFilter>(7, 1.0, 2.0);
	experiment.push<process::FeatureScaling>();
	experiment.push<LatencyCoding>();

	experiment.add_train<dataset::Cifar>(std::vector<std::string>({
		input_path+"data_batch_1.bin",
		input_path+"data_batch_2.bin",
		input_path+"data_batch_3.bin",
		input_path+"data_batch_4.bin",
		input_path+"data_batch_5.bin"
	}));

	experiment.add_test<dataset::Cifar>(std::vector<std::string>({
		input_path+"test_batch.bin"
	}));

	auto& conv1 = experiment.push<layer::Convolution>(5, 5, 64);
	conv1.set_name("conv1");
	conv1.parameter<uint32_t>("epoch").set(100);
	conv1.parameter<float>("annealing").set(0.99); //not specified in the paper Pattern Recognition
	conv1.parameter<float>("min_th").set(2.0); //not specified in the paper Pattern Recognition
	conv1.parameter<float>("t_obj").set(0.9);
	conv1.parameter<float>("lr_th").set(0.1);
	conv1.parameter<bool>("wta_infer").set(false); //not implemented in the public version + not specified in the paper Pattern Recognition
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(6.0, 0.1); //not as in the paper Pattern Recognition
	conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(0.1, 1);

	auto& pool1 = experiment.push<layer::Pooling>(2, 2, 2, 2);
	pool1.set_name("pool1");

	auto& conv2 = experiment.push<layer::Convolution>(5, 5, 128);
	conv2.set_name("conv2");
	conv2.parameter<uint32_t>("epoch").set(100);
	conv2.parameter<float>("annealing").set(0.99);
	conv2.parameter<float>("min_th").set(4.0f);
	conv2.parameter<float>("t_obj").set(0.9);
	conv2.parameter<float>("lr_th").set(0.1);
	conv2.parameter<bool>("wta_infer").set(false);
	conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
	conv2.parameter<STDP>("stdp").set<stdp::Multiplicative>(0.1, 1);

	auto& pool2 = experiment.push<layer::Pooling>(2, 2, 2, 2);
	pool2.set_name("pool2");

	// Activity after conv1
	auto& conv1_anal = experiment.output<DefaultOutput>(conv1, 0.0, 1.0);
	conv1_anal.add_analysis<analysis::Activity>();

	// SVM accuracy after conv1
	auto& conv1_out = experiment.output<DefaultOutput>(conv1, 0.0, 1.0);
	conv1_out.add_postprocessing<process::SumPooling>(2, 2);
	conv1_out.add_postprocessing<process::FeatureScaling>();
	conv1_out.add_analysis<analysis::Svm>();

	// Activity after conv2
	auto& conv2_anal = experiment.output<DefaultOutput>(conv2, 0.0, 1.0);
	conv2_anal.add_analysis<analysis::Activity>();

	// SVM accuracy after conv2
	auto& conv2_out = experiment.output<DefaultOutput>(conv2, 0.0, 1.0);
	conv2_out.add_postprocessing<process::SumPooling>(2, 2);
	conv2_out.add_postprocessing<process::FeatureScaling>();
	conv2_out.add_analysis<analysis::Svm>();

	// Save pool2 features
	auto& pool2_save = experiment.output<NoOutputConversion>(pool2);
	pool2_save.add_analysis<analysis::SaveOutputJson>("pool2_train.json", "pool2_test.json");

	// SVM Accuracy after pool2 (no sum pooling)
	auto& pool2_out = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_out.add_analysis<analysis::Activity>();
	pool2_out.add_analysis<analysis::Svm>();

	experiment.run(10000);
}
