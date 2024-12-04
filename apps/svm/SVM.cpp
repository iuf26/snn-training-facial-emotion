#include "Experiment.h"
#include "dataset/Video.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/ProcessExecution.h"
#include "execution/FusedExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "layer/Stream.h"
#include "process/SimplePreprocessing.h"
#include "process/CompositeChannels.h"
#include "process/OnOffFilter.h"
#include "process/OnOffTempFilter.h"
#include "process/EarlyFusion.h"
#include "process/SeparateSign.h"
#include "process/ResizeInput.h"
#include "tool/AutoFrameNumberSelector.h"
#include "process/MotionGridV1.h"

int main(int argc, char **argv)
{
	// for (int _repeat = 1; _repeat < 4; _repeat++)
	{
		std::string _dataset = "SVM";
		Experiment<ProcessExecution> experiment(argc, argv, _dataset, false, true);
		// number of frames per video.
		size_t _video_frames = 10;
		// The new dimentions of a video frame, set to zero if default dimentions are needed.
		size_t _frame_size_width = 80;
		size_t _frame_size_height = 60;

		size_t _train_sample_per_video = 0, _test_sample_per_video = 0;
		// number of frames to skip, this speeds up the action.
		size_t _th_mv = 0, _frame_gap = 3;
		size_t _grey = 1, _draw = 0;

		const char *input_path_ptr = std::getenv("INPUT_PATH");

		if (input_path_ptr == nullptr)
		{
			throw std::runtime_error("Require to define INPUT_PATH variable");
		}

		std::string input_path(input_path_ptr);

		// experiment.push<process::ResizeInput>(experiment.name(), _frame_size_width, _frame_size_height);
		// experiment.push<process::SimplePreprocessing>(experiment.name(), 1, _draw);
		// experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
		// experiment.push<process::FeatureScaling>();
		// experiment.push<process::SimplePreprocessing>(experiment.name(), 1, _draw);
		// experiment.push<process::MotionGridV1>(experiment.name(), _draw, 320, 144);

		// The location of the dataset Videos, seperated into train and test folders that contain labeled folders of videos.
		experiment.add_train<dataset::Video>(input_path + "/train/", _video_frames, _frame_gap, _th_mv, _train_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);
		experiment.add_test<dataset::Video>(input_path + "/test/", _video_frames, _frame_gap, _th_mv, _test_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);

		auto &svm = experiment.push<layer::Stream>(1, 1, 1, 1);

		// add another SVM step to classify the fused result.
		auto &svm_out = experiment.output<NoOutputConversion>(svm);
		svm_out.add_postprocessing<process::SumPooling>(20, 20);
		// svm_out.add_postprocessing<process::TemporalPooling>(2);
		svm_out.add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}