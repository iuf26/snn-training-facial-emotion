#include "Experiment.h"
#include "dataset/Video.h"
#include "dataset/Image.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecutionNew.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/SimplePreprocessing.h"
#include "process/MaxScaling.h"
#include "process/CompositeChannels.h"
#include "process/OnOffFilter.h"
#include "process/OnOffTempFilter.h"
#include "process/EarlyFusion.h"
#include "process/LateFusion.h"
#include "process/SeparateSign.h"
#include "tool/AutoFrameNumberSelector.h"

/**
 *  use this loop to find the ideal t_obj, for (float tobj = 0.10f; tobj <= 1.01f; tobj += 0.05f) float rounded_down = floorf(tobj * 100) / 100;
 */

int main(int argc, char **argv)
{
	for (int _repeat = 0; _repeat < 1; _repeat++)
	{
		std::string _dataset = "Image_weights";

		Experiment<SparseIntermediateExecutionNew> experiment(argc, argv, _dataset, false, true);
		// number of frames per video.
		size_t _video_frames = 8, _train_sample_per_video = 5, _test_sample_per_video = 0;

		// The new dimentions of a video frame, set to zero if default dimentions are needed.
		size_t _frame_size_width = 50, _frame_size_height = 50;
		// number of frames to skip, this speeds up the action.
		size_t _th_mv = 0;
		size_t _frame_gap = 2;

		size_t filter_size = 5;
		size_t sampling_size = 500; //(_frame_size_height * _frame_size_height) / filter_size;

		experiment.push<process::SimplePreprocessing>(experiment.name(), 0);

		experiment.push<process::MaxScaling>();
		experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);

		const char *input_path_ptr = std::getenv("INPUT_PATH");

		if (input_path_ptr == nullptr)
		{
			throw std::runtime_error("Require to define INPUT_PATH variable");
		}

		std::string input_path(input_path_ptr);

		experiment.push<LatencyCoding>();

		// The location of the dataset Videos, seperated into train and test folders that contain labeled folders of videos.
		experiment.add_train<dataset::Image>(input_path + "/train", _video_frames, _frame_gap, _th_mv, _train_sample_per_video); //, experiment.name(), 1);
		experiment.add_test<dataset::Image>(input_path + +"/test", _video_frames, _frame_gap, _th_mv, _test_sample_per_video); //, experiment.name(), 1);

		float t_obj = 0.65;
		float t_obj1 = 0.35;
		float t_obj2 = 0.15;

		float th_lr = 1.0f;
		float w_lr = 0.1f;

		// First convolution layer, name, width, hight, depth of filter(NUMBER OF FEATURES nf - filter number). stride x and y are fixed to 1 which means there is a big overlapping while extracting features.
		// This function takes the following(Layer Name, Kernel width, kernel height, number of kernels, and a flag to draw the weights if 1 or not if 0)
		auto &conv1 = experiment.push<layer::Convolution>(filter_size, filter_size, 16);
		conv1.set_name("conv1");
		conv1.parameter<bool>("draw").set(true);
		conv1.parameter<bool>("save_weights").set(true);
		conv1.parameter<bool>("inhibition").set(true);
		conv1.parameter<uint32_t>("epoch").set(sampling_size);
		conv1.parameter<float>("annealing").set(0.95f);
		conv1.parameter<float>("min_th").set(1.0f);
		conv1.parameter<float>("t_obj").set(t_obj);
		conv1.parameter<float>("lr_th").set(th_lr);
		// conv1.parameter<bool>("wta_infer").set(false);
		conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
		conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		auto &pool1 = experiment.push<layer::Pooling>(2, 2, 2, 2);
		pool1.set_name("pool1");

		auto &conv2 = experiment.push<layer::Convolution>(filter_size, filter_size, 32);
		conv2.set_name("conv2");
		conv2.parameter<bool>("draw").set(false);
		conv2.parameter<bool>("save_weights").set(true);
		conv2.parameter<bool>("inhibition").set(true);
		conv2.parameter<uint32_t>("epoch").set(sampling_size);
		conv2.parameter<float>("annealing").set(0.95f);
		conv2.parameter<float>("min_th").set(1.0f);
		conv2.parameter<float>("t_obj").set(t_obj1);
		conv2.parameter<float>("lr_th").set(th_lr);
		// conv2.parameter<bool>("wta_infer").set(false);
		conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
		conv2.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		auto &pool2 = experiment.push<layer::Pooling>(2, 2, 2, 2);
		pool2.set_name("pool2");

		auto &fc1 = experiment.push<layer::Convolution>(filter_size, filter_size, 64);
		fc1.set_name("fc1");
		fc1.parameter<bool>("draw").set(false);
		fc1.parameter<bool>("save_weights").set(true);
		fc1.parameter<bool>("inhibition").set(true);
		fc1.parameter<uint32_t>("epoch").set(sampling_size);
		fc1.parameter<float>("annealing").set(0.95f);
		fc1.parameter<float>("min_th").set(1.0f);
		fc1.parameter<float>("t_obj").set(t_obj2);
		fc1.parameter<float>("lr_th").set(th_lr);
		// fc1.parameter<bool>("wta_infer").set(false);
		fc1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		fc1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
		fc1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		// #ifdef ENABLE_QT
		// 		conv1.plot_threshold(true);
		// 		conv1.plot_reconstruction(true);
		// #endif

		auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
		conv1_out.add_postprocessing<process::SumPooling>(20, 20);
		// conv1_out.add_postprocessing<process::TemporalPooling>(1);
		conv1_out.add_postprocessing<process::FeatureScaling>();
		conv1_out.add_analysis<analysis::Activity>();
		conv1_out.add_analysis<analysis::Coherence>();
		conv1_out.add_analysis<analysis::Svm>();

		auto &conv2_out = experiment.output<TimeObjectiveOutput>(conv2, t_obj1);
		conv2_out.add_postprocessing<process::SumPooling>(20, 20);
		// conv2_out.add_postprocessing<process::TemporalPooling>(1);
		conv2_out.add_postprocessing<process::FeatureScaling>();
		conv2_out.add_analysis<analysis::Activity>();
		conv2_out.add_analysis<analysis::Coherence>();
		conv2_out.add_analysis<analysis::Svm>();

		auto &fc1_out = experiment.output<TimeObjectiveOutput>(fc1, t_obj2);
		fc1_out.add_postprocessing<process::SumPooling>(20, 20);
		fc1_out.add_postprocessing<process::TemporalPooling>(1);
		fc1_out.add_postprocessing<process::FeatureScaling>();
		fc1_out.add_analysis<analysis::Activity>();
		fc1_out.template add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}