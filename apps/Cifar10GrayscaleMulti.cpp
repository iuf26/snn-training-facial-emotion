#include "Experiment.h"
#include "dataset/Cifar.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/SaveOutputNumpy.h"
#include "analysis/SaveOutputJson.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/GrayScale.h"
#include "process/OnOffFilter.h"

int main(int argc, char** argv) {
	Experiment<DenseIntermediateExecution> experiment(argc, argv, "cifar10gs");

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
	conv1.parameter<uint32_t>("epoch").set(200);
	conv1.parameter<float>("annealing").set(0.99);
	conv1.parameter<float>("min_th").set(2.0);
	conv1.parameter<float>("t_obj").set(0.9);
	conv1.parameter<float>("lr_th").set(0.1);
	conv1.parameter<bool>("wta_infer").set(false);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(5.0, 0.1);
	conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(0.1, 1);

	auto& pool1 = experiment.push<layer::Pooling>(5, 5, 5, 5);
	pool1.set_name("pool1");

	auto& conv2 = experiment.push<layer::Convolution>(3, 3, 128);
	conv2.set_name("conv2");
	conv2.parameter<uint32_t>("epoch").set(200);
	conv2.parameter<float>("annealing").set(0.99);
	conv2.parameter<float>("min_th").set(3.0);
	conv2.parameter<float>("t_obj").set(0.8);
	conv2.parameter<float>("lr_th").set(0.1);
	conv2.parameter<bool>("wta_infer").set(false);
	conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(6.0, 0.1);
	conv2.parameter<STDP>("stdp").set<stdp::Multiplicative>(0.1, 1);

	auto& pool2 = experiment.push<layer::Pooling>(2, 2, 2, 2);
	pool2.set_name("pool2");

	
	///////////////////////////////
	/////////// OUTPUTS ///////////
	///////////////////////////////

	// conv1 : Save features
	//auto& conv1_save = experiment.output<SpikeTiming>(conv1);
	//conv1_save.add_postprocessing<process::MeanPooling>(2, 2); //sum pooling in the spike domain
	//conv1_save.add_analysis<analysis::SaveOutputNumpy>("meanPool_conv1");

	//auto& conv1_save2 = experiment.output<SpikeTiming>(conv1);
	//conv1_save2.add_postprocessing<process::MeanPooling>(2, 2); //sum pooling in the spike domain
	//conv1_save2.add_analysis<analysis::SaveOutputJson>("meanPool_conv1_train.json", "meanPool_conv1_test.json");

	// conv1 : Activity
	//auto& conv1_analysis = experiment.output<DefaultOutput>(conv1, 0.0, 1.0);
	//conv1_analysis.add_analysis<analysis::Activity>();

	// conv1 : SVM evaluation
	//auto& conv1_out = experiment.output<DefaultOutput>(conv1, 0.0, 1.0);
	//conv1_out.add_postprocessing<process::SumPooling>(2, 2);
	//conv1_out.add_postprocessing<process::FeatureScaling>();
	//conv1_out.add_analysis<analysis::Svm>();

	// pool1 : Save features
	auto& pool1_save = experiment.output<SpikeTiming>(pool1);
	pool1_save.add_analysis<analysis::SaveOutputNumpy>("pool1");

	// pool1 : Activity
	auto& pool1_activity = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_activity.add_analysis<analysis::Activity>();

	// pool1 : SVM evaluation
	auto& pool1_out = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_out.add_postprocessing<process::SumPooling>(2, 2);
	pool1_out.add_analysis<analysis::Svm>();

	// pool2 : Save features
	auto& pool2_save = experiment.output<SpikeTiming>(pool2);
	pool2_save.add_analysis<analysis::SaveOutputNumpy>("pool2");

	// pool2 : Activity
	auto& pool2_activity = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_activity.add_analysis<analysis::Activity>();

	// pool2 : SVM evaluation
	auto& pool2_out = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_out.add_postprocessing<process::SumPooling>(2, 2);
	pool2_out.add_analysis<analysis::Svm>();

	experiment.run(10000);
}
