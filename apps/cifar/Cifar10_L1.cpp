#include "Experiment.h"
#include "dataset/Cifar.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/Whiten.h"
#include "process/WhitenPatches.h"
#include "process/SeparateSign.h"

int main(int argc, char **argv)
{
	Experiment<SparseIntermediateExecution> experiment(argc, argv, "cifar10");

	const char *input_path_ptr = std::getenv("INPUT_PATH");

	if (input_path_ptr == nullptr)
	{
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	experiment.template push<process::WhiteningPatches>(9, 0.01, 0.15, 2, 1000000);
	//experiment.push<process::WhitenPatchesLoader>("whiten_filters");
	//experiment.push<process::Whitening>(0.01, 1.0);
	experiment.push<process::SeparateSign>();
	experiment.push<process::SampleScaling>();
	experiment.push<LatencyCoding>();

	experiment.add_train<dataset::Cifar>(std::vector<std::string>({input_path + "data_batch_1.bin",
																   input_path + "data_batch_2.bin",
																   input_path + "data_batch_3.bin",
																   input_path + "data_batch_4.bin",
																   input_path + "data_batch_5.bin"}));

	experiment.add_test<dataset::Cifar>(std::vector<std::string>({input_path + "test_batch.bin"}));

	float t_obj = 0.97f;
	float th_lr = 1.0f;
	float w_lr = 0.1f;

	auto &conv1 = experiment.push<layer::Convolution>(5, 5, 64);
	conv1.set_name("conv1");
	conv1.parameter<bool>("draw").set(false);
	conv1.parameter<bool>("inhibition").set(true);
	conv1.parameter<bool>("save_weights").set(false);
	conv1.parameter<uint32_t>("epoch").set(100);
	conv1.parameter<float>("annealing").set(0.95f);
	conv1.parameter<float>("min_th").set(4.0f);
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(th_lr);
	conv1.parameter<bool>("wta_infer").set(true);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
	conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(w_lr, 1.0);

	auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
	conv1_out.add_postprocessing<process::SumPooling>(2, 2);
	conv1_out.add_postprocessing<process::FeatureScaling>();
	conv1_out.add_analysis<analysis::Activity>();
	conv1_out.add_analysis<analysis::Coherence>();
	conv1_out.add_analysis<analysis::Svm>();

#ifdef ENABLE_QT
	conv1.plot_threshold(true);
	conv1.plot_reconstruction<process::GaussianTemporalCodingColor>(true);
#endif

	experiment.run(10000);
}
