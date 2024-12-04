#include "Benchmark.h"
#include "process/OnOffFilter.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/Whiten.h"
#include "process/SeparateSign.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "stdp/Biological.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "analysis/Svm.h"
#include "dataset/Cifar.h"
#include "execution/SparseIntermediateExecution.h"
#include "stdp/Multiplicative.h"

template <typename T>
struct Handler
{

	static void run(Experiment<T> &experiment, const std::map<std::string, float> &variables)
	{
		const char *input_path_ptr = std::getenv("INPUT_PATH");

		if (input_path_ptr == nullptr)
		{
			throw std::runtime_error("Require to define INPUT_PATH variable");
		}

		std::string input_path(input_path_ptr);

		experiment.template push<process::Whitening>(0.01, 1.0);
		experiment.template push<process::SeparateSign>();
		experiment.template push<process::SampleScaling>();
		experiment.template push<LatencyCoding>();

		experiment.template add_train<dataset::Cifar>(std::vector<std::string>({input_path + "data_batch_1.bin",
																				input_path + "data_batch_2.bin",
																				input_path + "data_batch_3.bin",
																				input_path + "data_batch_4.bin",
																				input_path + "data_batch_5.bin"}));

		experiment.template add_test<dataset::Cifar>(std::vector<std::string>({input_path + "test_batch.bin"}));

		float t_obj = variables.at("t_obj");
		size_t size = variables.at("size");
		float th_lr = 1.0f;
		float w_lr = 0.1f;

		auto &conv1 = experiment.template push<layer::Convolution>(5, 5, size);
		conv1.set_name("conv1");
		conv1.template parameter<bool>("draw").set(false);
		conv1.template parameter<bool>("inhibition").set(true);
		conv1.template parameter<bool>("save_weights").set(false);
		conv1.template parameter<uint32_t>("epoch").set(100);
		conv1.template parameter<float>("annealing").set(0.95f);
		conv1.template parameter<float>("min_th").set(4.0f);
		conv1.template parameter<float>("t_obj").set(t_obj);
		conv1.template parameter<float>("lr_th").set(th_lr);
		conv1.template parameter<bool>("wta_infer").set(true);
		conv1.template parameter<Tensor<float>>("w").template distribution<distribution::Uniform>(0.0, 1.0);
		conv1.template parameter<Tensor<float>>("th").template distribution<distribution::Gaussian>(10.0, 0.1);
		conv1.template parameter<STDP>("stdp").template set<stdp::Multiplicative>(w_lr, 1.0);

		auto &conv1_out = experiment.template output<TimeObjectiveOutput>(conv1, t_obj);
		conv1_out.template add_postprocessing<process::SumPooling>(2, 2);
		conv1_out.template add_postprocessing<process::FeatureScaling>();
		conv1_out.template add_analysis<analysis::Activity>();
		conv1_out.template add_analysis<analysis::Coherence>();
		conv1_out.template add_analysis<analysis::Svm>();
	}
};

int main(int argc, char **argv)
{
	Benchmark<Handler, SparseIntermediateExecution> bench(argc, argv, "bench1extra", 3);
	bench.add_variable("t_obj", {0.91f, 0.92f, 0.93f, 0.94f, 0.96f, 0.97f, 0.98f, 0.99f, 1.0f});
	bench.add_variable("size", {64.0f, 128.0f, 256.0f, 512.0f, 1024.0f});

	int mem = 128;
	int unit = 5;
	bench.run(mem / unit);
}
