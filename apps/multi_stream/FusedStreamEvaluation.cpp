#include "Experiment.h"
#include "dataset/TwoStream.h"
#include "dataset/Image.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Stream.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/FusedExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/EarlyFusion.h"
#include "process/LateFusion.h"
#include "process/SimplePreprocessing.h"
#include "process/SeparateSign.h"
/**
 * @brief This function runs Image expirements with a 2D convolutional SNN. This class contains a sequential run of pre-prosessing,
 * neural coding, convolutions, pooling and visualization.
 *
 * @param DefaultOnOffFilter The on-center off-center filter. This specific filter is used with grey-scale information.
 * @param FeatureScaling This normalizes the input pixels into values between zero and one.
 * @param LatencyCoding The type of neural coding chosen for this expirement transforms the normalized pixel values into timestamps.
 * @param w_lr The weights learning rate.
 * @param th_lr The threshould learning rate.
 * @param t_obj The objective time that neuron firing should converge to. This is for threshould adaptation.
 */

int main(int argc, char **argv)
{
	// std::string _from_exps[1] = {"Weiz16_TS_MG_daria_1"};
	// for (std::string _from_exp : _from_exps)
	{
		std::string _dataset = "FusedStreams";

		const char *input_path_ptr = std::getenv("INPUT_PATH");

		if (input_path_ptr == nullptr)
			throw std::runtime_error("Require to define INPUT_PATH variable");

		std::string input_path(input_path_ptr);
		size_t _filter_nbr = 64;
		
		// add code to re-fill the saved values from the text files and fuse them here.
		std::string _file_path = std::filesystem::current_path();

		// The name of the experiment_space is tha name of the dataset, this name is used for the log text file. // flag that permits saving the exp output tensors or not
		Experiment<FusedExecution> experiment_fused(argc, argv, _dataset + "_W", false, false);
		// /home/melassal/Workspace/CSNN/csnn-simulator-build/Weizmann_FEATURES/WeizS_TS_FS_daria
		experiment_fused.add_train<dataset::TwoStream>("/home/melassal/Workspace/Results/Features/Weiz_cut/" + input_path + "/train/", 1, 0);
		experiment_fused.add_test<dataset::TwoStream>("/home/melassal/Workspace/Results/Features/Weiz_cut/" + input_path + "/test/", 1, 0);
		// experiment_fused.add_train<dataset::TwoStream>(_file_path + "/ExtractedFeatures/" + input_path + "/train/", 1, 0);
		// experiment_fused.add_test<dataset::TwoStream>(_file_path + "/ExtractedFeatures/" + input_path + "/test/", 1, 0);// Weizmann_FEATURES_SP

		auto &svm = experiment_fused.push<layer::Stream>(1, 1, 1, _filter_nbr);

		// add another SVM step to classify the fused result.
		auto &fused_out = experiment_fused.output<NoOutputConversion>(svm);
		fused_out.add_analysis<analysis::Svm>();

		experiment_fused.run(10000);
	}
}