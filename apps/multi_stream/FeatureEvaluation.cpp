#include "Experiment.h"
#include "dataset/LoadSavedFeatures.h"
#include "dataset/Image.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Stream.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecution.h"
#include "execution/EmptyExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/SeparateSign.h"
#include "process/MaxScaling.h"

int main(int argc, char **argv)
{
	std::string _from_exps[1] = { "KTH-2-M23-5-L2-2D1D_2D"}; 
	for (std::string _from_exp : _from_exps)
	{
		std::string _dataset = "Experiment";
		size_t _filter_nbr = 64;

		const char *exp_stream_ptr = std::getenv("INPUT_PATH");

		if (exp_stream_ptr == nullptr)
			throw std::runtime_error("Require to define INPUT_PATH variable");

		// add code to re-fill the saved values from the text files and fuse them here.
		std::string _file_path = std::filesystem::current_path();

		// The name of the experiment_space is tha name of the dataset, this name is used for the log text file. // flag that permits saving the exp output tensors or not
		Experiment<EmptyExecution> experiment(argc, argv, _dataset + "_" + _from_exp, false, false);

		experiment.add_train<dataset::LoadSavedFeatures>("/home/melassal/Workspace/Results/Features-2d1dvs3d/" + _from_exp + "/train/", 1);
		experiment.add_test<dataset::LoadSavedFeatures>("/home/melassal/Workspace/Results/Features-2d1dvs3d/" + _from_exp + "/test/", 1);

		auto &svm = experiment.push<layer::Stream>(1, 1, 1, _filter_nbr);

		// add another SVM step to classify the fused result.
		auto &fused_out = experiment.output<NoOutputConversion>(svm);
		fused_out.add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}