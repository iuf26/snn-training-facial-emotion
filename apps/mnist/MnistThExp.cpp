#include "Experiment.h"
#include "dataset/Mnist.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "stdp/Proportional.h"
#include "layer/Convolution.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "layer/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "stdp/Linear.h"
#include "stdp/BiologicalMultiplicative.h"
#include "analysis/SaveOutput.h"

int main(int argc, char **argv)
{
	Experiment<SparseIntermediateExecution> experiment(argc, argv, "mnist");

	experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
	experiment.push<process::FeatureScaling>();
	experiment.push<LatencyCoding>();

	const char *input_path_ptr = std::getenv("INPUT_PATH");

	if (input_path_ptr == nullptr)
	{
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	experiment.add_train<dataset::Mnist>(input_path + "train-images.idx3-ubyte", input_path + "train-labels.idx1-ubyte");
	experiment.add_test<dataset::Mnist>(input_path + "t10k-images.idx3-ubyte", input_path + "t10k-labels.idx1-ubyte");

	float th_lr = 0.1f;
	float w_lr = 1.0f;


	int ks=argc>2?atoi(argv[1]):5;
	int fn=argc>3?atoi(argv[2]):32;
	float t_obj=argc>4?atoi(argv[3]):0.75;
	
	auto &conv1 = experiment.push<layer::Convolution>(ks, ks, fn); 
	conv1.set_name("conv1");
	conv1.parameter<bool>("draw").set(false);
	conv1.parameter<bool>("save_weights").set(true);
	conv1.parameter<bool>("inhibition").set(true);
	conv1.parameter<uint32_t>("epoch").set(100);
	conv1.parameter<float>("annealing").set(0.95f);
	conv1.parameter<float>("min_th").set(1.0f);
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(th_lr);
	conv1.parameter<bool>("wta_infer").set(false);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
	conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

#ifdef ENABLE_QT
	conv1.plot_threshold(true);
	conv1.plot_reconstruction(true);
#endif

	auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
	conv1_out.add_postprocessing<process::SumPooling>(2, 2);
	conv1_out.add_postprocessing<process::FeatureScaling>();
	conv1_out.add_analysis<analysis::Activity>();
	conv1_out.add_analysis<analysis::Coherence>();
	conv1_out.add_analysis<analysis::Svm>();

	experiment.run(10000);

	return experiment.wait();
}
