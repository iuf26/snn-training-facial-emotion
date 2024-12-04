#include "Experiment.h"
#include "dataset/Video.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "process/ResidualConnection.h"
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
#include "process/Acceleration.h"
#include "process/ResizeInput.h"
#include "process/SimplePreprocessing.h"
#include "process/OrientationAmplitude.h"
#include "process/MaxScaling.h"
#include "process/SaveFeatures.h"
#include "process/OnOffFilter.h"
#include "process/OnOffTempFilter.h"
#include "process/EarlyFusion.h"
#include "process/Flatten.h"
#include "process/SeparateSign.h"
#include "tool/AutoFrameNumberSelector.h"
#include "process/SpikingBackgroundSubtraction.h"
#include "process/MotionGrid.h"
#include "process/SpikingMotionGrid.h"
#include "process/Amplification.h"
#include "process/AddSaltPepperNoise.h"

/**
 *  use this loop to find the ideal t_obj, for (float tobj = 0.10f; tobj <= 1.01f; tobj += 0.05f) float rounded_down = floorf(tobj * 100) / 100;
 */
int main(int argc, char **argv)
{
	for (int _repeat = 1; _repeat < 4; _repeat++) // RUN MORE MG exp UCF, RUN EXP THAT I DON4T LIKE THE RESULTS OF
	{
		std::string _dataset = "KTH_resnet";

		Experiment<SparseIntermediateExecutionNew> experiment(argc, argv, _dataset, false, true);

		// The new dimentions of a video frame, set to zero if default dimentions are needed.
		size_t _frame_size_width = 40, _frame_size_height = 30;
		// number of sets of frames per video.
		size_t _video_frames = 10, _train_sample_per_video = 0, _test_sample_per_video = 0;
		// number of frames to skip, this speeds up the action.
		size_t _th_mv = 0, _frame_gap = 4, _grey = 1, _draw = 0;
		// filter sizes
		size_t filter_size = 5, filter_number = 8, tmp_filter_size = 1, tmp_pooling_size = 2, temp_stride = 1;
		size_t sampling_size = 200;

		experiment.push<process::MaxScaling>();
		// experiment.push<process::SimplePreprocessing>(experiment.name(), 0, _draw);
		// experiment.push<process::DefaultOnOffFilter>(24, 0.5, 5.0);
		experiment.push<process::DefaultOnOffTempFilter>(experiment.name(), 24, 5, 0.5, 5.0, 0.5, 5.0);
		// experiment.push<process::FeatureScaling>();

		const char *input_path_ptr = std::getenv("INPUT_PATH");
		if (input_path_ptr == nullptr)
			throw std::runtime_error("Require to define INPUT_PATH variable");
		std::string input_path(input_path_ptr);

		experiment.push<LatencyCoding>();
		// experiment.push<process::SaveFeatures>(experiment.name(), "Input");

		experiment.add_train<dataset::Video>(input_path + "/train/", _video_frames, _frame_gap, _th_mv, _train_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);
		experiment.add_test<dataset::Video>(input_path + "/test/", _video_frames, _frame_gap, _th_mv, _test_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);

		float t_obj = 0.65;
		float th_lr = 0.09f;
		float w_lr = 0.009f;

		auto &pool1 = experiment.push<layer::Pooling3D>(1, 1, tmp_pooling_size, 1, 1, temp_stride);
		pool1.set_name("pool1");

		// This function takes the following(Layer Name, Kernel width, kernel height, number of kernels, and a flag to draw the weights if 1 or not if 0)
		auto &conv1 = experiment.push<layer::Convolution3D>(filter_size, filter_size, tmp_filter_size, filter_number, "", 1, 1, 1);
		conv1.set_name("conv1");
		conv1.parameter<bool>("draw").set(false);
		conv1.parameter<bool>("save_weights").set(false);
		conv1.parameter<bool>("save_random_start").set(false);
		conv1.parameter<bool>("log_spiking_neuron").set(false);
		conv1.parameter<bool>("inhibition").set(true);
		conv1.parameter<uint32_t>("epoch").set(sampling_size);
		conv1.parameter<float>("annealing").set(0.95f);
		conv1.parameter<float>("min_th").set(1.0f);
		conv1.parameter<float>("t_obj").set(t_obj);
		conv1.parameter<float>("lr_th").set(th_lr);
		conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
		conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj, experiment.name(), conv1.name(), false);
		// conv1_out.add_postprocessing<process::SaveFeatures>(experiment.name(), conv1.name());
		// conv1_out.add_postprocessing<process::ResidualConnection>(experiment.name(), "");
		conv1_out.add_postprocessing<process::SumPooling>(10, 10);
		conv1_out.add_postprocessing<process::TemporalPooling>(5);
		conv1_out.add_postprocessing<process::FeatureScaling>();
		conv1_out.add_analysis<analysis::Activity>();
		conv1_out.add_analysis<analysis::Coherence>();
		conv1_out.add_analysis<analysis::Svm>();

		auto &pool2 = experiment.push<layer::Pooling3D>(1, 1, tmp_pooling_size, 1, 1, temp_stride);
		pool2.set_name("pool2");

		// This function takes the following(Layer Name, Kernel width, kernel height, number of kernels, and a flag to draw the weights if 1 or not if 0)
		auto &conv2 = experiment.push<layer::Convolution3D>(filter_size, filter_size, tmp_filter_size, filter_number, "", 1, 1, 1);
		conv2.set_name("conv2");
		conv2.parameter<bool>("draw").set(false);
		conv2.parameter<bool>("save_weights").set(false);
		conv2.parameter<bool>("save_random_start").set(false);
		conv2.parameter<bool>("log_spiking_neuron").set(false);
		conv2.parameter<bool>("inhibition").set(true);
		conv2.parameter<uint32_t>("epoch").set(sampling_size);
		conv2.parameter<float>("annealing").set(0.95f);
		conv2.parameter<float>("min_th").set(1.0f);
		conv2.parameter<float>("t_obj").set(t_obj);
		conv2.parameter<float>("lr_th").set(th_lr);
		conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
		conv2.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		auto &conv2_out = experiment.output<TimeObjectiveOutput>(conv2, t_obj);
		// conv2_out.add_postprocessing<process::SaveFeatures>(experiment.name(), conv2.name());
		conv2_out.add_postprocessing<process::ResidualConnection>(experiment.name(), conv1.name());
		conv2_out.add_postprocessing<process::SumPooling>(10, 10);
		conv2_out.add_postprocessing<process::TemporalPooling>(5);
		conv2_out.add_postprocessing<process::FeatureScaling>();
		conv2_out.add_analysis<analysis::Activity>();
		conv2_out.add_analysis<analysis::Coherence>();
		conv2_out.add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}