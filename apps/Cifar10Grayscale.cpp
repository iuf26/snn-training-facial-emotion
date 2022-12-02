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

	float t_obj = 0.7f;
	float th_lr = 0.001f;
	float w_lr = 0.001f;

	auto& conv1 = experiment.push<layer::Convolution>(5, 5, 64);
	conv1.set_name("conv1");
	conv1.parameter<uint32_t>("epoch").set(100);
	conv1.parameter<float>("annealing").set(1.0f); //not specified in the paper Pattern Recognition
	conv1.parameter<float>("min_th").set(0.0f); //not specified in the paper Pattern Recognition
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(th_lr);
	conv1.parameter<bool>("wta_infer").set(false); //not implemented in the public version + not specified in the paper Pattern Recognition
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(2.0, 0.1); //not as in the paper Pattern Recognition
	conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(w_lr, 1);

	// Save conv1 output
	// IMPOSSIBLE BECAUSE TOO BIG
	//auto& conv1_save = experiment.output<NoOutputConversion>(conv1);
	//conv1_save.add_analysis<analysis::SaveOutputJson>("conv1_train.json", "conv1_test.json");

	// Save mean pool1 output
	//auto& pool1_save = experiment.output<SpikeTiming>(conv1);
	//pool1_save.add_postprocessing<process::MeanPooling>(2, 2); //sum pooling in the spike domain
	//pool1_save.add_analysis<analysis::SaveOutputJson>("meanPool_conv1_train.json", "meanPool_conv1_test.json");

	// Output analysis
	auto& conv1_out = experiment.output<DefaultOutput>(conv1, 0.0, 1.0);
	conv1_out.add_postprocessing<process::SumPooling>(2, 2);
	conv1_out.add_postprocessing<process::FeatureScaling>();
	conv1_out.add_analysis<analysis::Svm>();

#ifdef ENABLE_QT
	conv1.plot_threshold(true);
	conv1.plot_reconstruction<process::GaussianTemporalCodingColor>(true);
#endif

	experiment.run(10000);
}
