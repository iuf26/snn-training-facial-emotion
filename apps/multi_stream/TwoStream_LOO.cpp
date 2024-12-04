#include "Experiment.h"
#include "dataset/Video.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
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
#include "process/MotionGrid.h"
#include "process/MotionGridV1.h"
#include "process/MotionGridV5.h"
#include "process/OnOffFilter.h"
#include "process/OnOffTempFilter.h"
#include "process/EarlyFusion.h"
#include "process/LateFusion.h"
#include "process/SeparateSign.h"
#include "tool/VideoFrameSelector.h"
#include "tool/AutoFrameNumberSelector.h"
#include "dataset/TwoStream.h"
#include "layer/Stream.h"
#include "process/ResizeInput.h"
#include "process/SimplePreprocessing.h"
#include "process/OrientationAmplitude.h"
#include "execution/FusedExecution.h"
#include "process/MaxScaling.h"
#include "process/GaussianFilter.h"
#include "process/TemporalGaussianFilter.h"

/** TORUN
 *  use this loop to find the ideal t_obj, for (float tobj = 0.10f; tobj <= 1.01f; tobj += 0.05f) float rounded_down = floorf(tobj * 100) / 100;
 */
int main(int argc, char **argv)
{
	// std::string subjects[10] = {"alba", "amel", "andreas", "chiara", "clare", "daniel", "florian", "hedlena", "julien", "nicolas"};
	std::string subjects[9] = {"daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"};

	for (std::string subject : subjects)
		for (int _repeat = 1; _repeat < 4; _repeat++)
		{
			std::string _dataset = "Weiz-TS-3D-cuttoff20-" + subject;
			// std::string _dataset = "Weizmann-TS-EFM1-" + subject;
			// The name of the experiment_space is tha name of the dataset, this name is used for the log text file. // flag that permits saving the exp output tensors or not.
			Experiment<SparseIntermediateExecutionNew> experiment_space(argc, argv, _dataset, false, true);
			Experiment<SparseIntermediateExecutionNew> experiment_time(argc, argv, _dataset + "_time", false, true);
			Experiment<FusedExecution> experiment_fused(argc, argv, _dataset + "_fused", false, false);

			// The new dimentions of a video frame, set to zero if default dimentions are needed.
			// size_t _frame_size_width = 48, _frame_size_height = 64;
			size_t _frame_size_width = 91, _frame_size_height = 72;
			// number of sets of frames per video.
			size_t _video_frames = 10, _train_sample_per_video = 0, _test_sample_per_video = 0,
				   _train_sample_per_video_2 = 0, _test_sample_per_video_2 = 0;
			// number of frames to skip, this speeds up the action.
			size_t _th_mv = 0, _frame_gap_train = 3, _frame_gap_test = 3;
			size_t _method = 1, _grey = 1, _draw = 0;
			// filter sizes
			size_t filter_size = 5, filter_number = 64, tmp_filter_size = 2, tmp_pooling_size = 1; // tmp_filter_size == 2 ? 2 : 1;
			size_t sampling_size = 800;															   //(_frame_size_height * _frame_size_width) / (filter_size * filter_size);
			size_t spatial_stride = 1, temporal_stride = 1;

			const char *input_path_ptr = std::getenv("INPUT_PATH");

			if (input_path_ptr == nullptr)
				throw std::runtime_error("Require to define INPUT_PATH variable");

			std::string input_path(input_path_ptr);

			// The location of the dataset images, seperated into train and test folders that contain labeled folders of images.
			experiment_space.add_train<dataset::Video>(input_path + "/" + subject + "/train", _video_frames, _frame_gap_test, _th_mv, _train_sample_per_video, _grey, experiment_space.name(), _draw, _frame_size_width, _frame_size_height);
			experiment_space.add_test<dataset::Video>(input_path + "/" + subject + "/test", _video_frames, _frame_gap_test, _th_mv, _test_sample_per_video, _grey, experiment_space.name(), _draw, _frame_size_width, _frame_size_height);

			experiment_time.add_train<dataset::Video>(input_path + "/" + subject + "/train", _video_frames, _frame_gap_test, _th_mv, _train_sample_per_video_2, _grey, experiment_time.name(), _draw, _frame_size_width, _frame_size_height);
			experiment_time.add_test<dataset::Video>(input_path + "/" + subject + "/test", _video_frames, _frame_gap_test, _th_mv, _test_sample_per_video_2, _grey, experiment_time.name(), _draw, _frame_size_width, _frame_size_height);

			// experiment.push<process::CompositeChannels2>(experiment.name(), 1, 50);
			// experiment.push<process::CompositeChannels>(experiment.name(), 1);
			// experiment_time.push<process::OrientationAmplitude>(experiment_time.name());
			// experiment_time.push<process::EarlyFusion>(experiment_time.name(), _draw, _video_frames, 0);
			// experiment_time.push<process::SimplePreprocessing>(experiment_time.name(), 1);
			// experiment_time.push<process::MotionGridV1>(experiment_time.name(), 0, 91, 72, 48, 4, 3, 50);
			// experiment_time.push<process::MotionGrid>(experiment_time.name(), 0, 100, 150, 48, 6, 12, 50);
			// experiment_time.push<process::SimplePreprocessing>(experiment_time.name(), 1);
			// experiment_time.push<process::MotionGridV1>(experiment_time.name(), 0, 100, 80, 48, 4, 12, 1);
			// experiment_time.push<process::OrientationAmplitude>(experiment_time.name());
			// 13 3 // 21 5 // 29 7
			// experiment_time.push<process::TemporalGaussianFilter>(experiment_time.name(), 0, 7, 1.5);
			// experiment_space.push<process::TemporalGaussianFilter>(experiment_space.name(), 0, 7, 1.5);
			// experiment_space.push<process::GaussianFilter>(experiment_space.name(), 0, 7, 1.5);
			// experiment_time.push<process::GaussianFilter>(experiment_time.name(), 0, 7, 1.5);
			// experiment_time.push<process::SimplePreprocessing>(experiment_time.name(), 1, 0);
			//experiment_time.push<process::MotionGrid>(experiment_time.name(), 0, 150, 100, 48, 6, 12, 50);

			experiment_space.push<process::DefaultOnOffFilter>(7, 0.1, 4.0, 20, experiment_time.name(), 0);
			experiment_time.push<process::DefaultOnOffFilter>(7, 0.1, 4.0, 20, experiment_time.name(), 0);

			// experiment_time.push<process::EarlyFusion>(experiment_time.name(), _draw, _video_frames, 1);

			experiment_space.push<process::MaxScaling>();
			// experiment_space.push<process::FeatureScaling>();
			experiment_time.push<process::MaxScaling>();
			// experiment_time.push<process::FeatureScaling>();

			// experiment_space.push<process::DefaultOnOffTempFilter>(experiment_space.name(), 24, 5, 0.5, 5.0, 0.5, 5.0, 0);
			// experiment_time.push<process::DefaultOnOffTempFilter>(experiment_time.name(), 24, 5, 0.5, 5.0, 0.5, 5.0, 0);

			// Temporal coding used, the data value "float" is transformed and reprisented by the spike timestamp.
			experiment_space.push<LatencyCoding>();
			experiment_time.push<LatencyCoding>();

			float t_obj = 0.65;

			float th_lr = 1.0f;
			float w_lr = 0.1f;

			//------------- SPACE -------------
			// auto &pool1_s = experiment_space.push<layer::Pooling3D>(spatial_stride, spatial_stride, tmp_pooling_size, spatial_stride, spatial_stride, temporal_stride);
			// pool1_s.set_name("pool1_s");

			auto &conv1_s = experiment_space.push<layer::Convolution3D>(filter_size, filter_size, 1, filter_number, "", 1, 1, temporal_stride);
			conv1_s.set_name("conv1_s");
			conv1_s.parameter<bool>("draw").set(false);
			conv1_s.parameter<bool>("save_weights").set(true);
			conv1_s.parameter<bool>("save_random_start").set(false);
			conv1_s.parameter<bool>("log_spiking_neuron").set(false);
			conv1_s.parameter<bool>("inhibition").set(true);
			conv1_s.parameter<uint32_t>("epoch").set(sampling_size);
			conv1_s.parameter<float>("annealing").set(0.95f);
			conv1_s.parameter<float>("min_th").set(1.0f);
			conv1_s.parameter<float>("t_obj").set(t_obj);
			conv1_s.parameter<float>("lr_th").set(th_lr);
			conv1_s.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
			conv1_s.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
			conv1_s.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 1.0);

			// TimeObjectiveOutput converts timestamps back into float values.
			auto &conv1_s_out = experiment_space.output<TimeObjectiveOutput>(conv1_s, t_obj);
			conv1_s_out.add_postprocessing<process::SumPooling>(20, 20);
			conv1_s_out.add_postprocessing<process::TemporalPooling>(2);
			// conv1_s_out.add_postprocessing<process::FeatureScaling>();
			conv1_s_out.add_analysis<analysis::Activity>();
			conv1_s_out.add_analysis<analysis::Coherence>();
			conv1_s_out.add_analysis<analysis::Svm>();

			experiment_space.run(10000);

			//------------- TIME -------------
			// auto &pool1_t = experiment_time.push<layer::Pooling3D>(spatial_stride, spatial_stride, tmp_pooling_size, spatial_stride, spatial_stride, temporal_stride);
			// pool1_t.set_name("pool1_t");

			auto &conv1_t = experiment_time.push<layer::Convolution3D>(filter_size, filter_size, tmp_filter_size, filter_number, "", 1, 1, temporal_stride);
			conv1_t.set_name("conv1_t");
			conv1_t.parameter<bool>("draw").set(false);
			conv1_t.parameter<bool>("save_weights").set(true);
			conv1_t.parameter<bool>("save_random_start").set(false);
			conv1_t.parameter<bool>("log_spiking_neuron").set(false);
			conv1_t.parameter<bool>("inhibition").set(true);
			conv1_t.parameter<uint32_t>("epoch").set(sampling_size);
			conv1_t.parameter<float>("annealing").set(0.95f);
			conv1_t.parameter<float>("min_th").set(1.0f);
			conv1_t.parameter<float>("t_obj").set(t_obj);
			conv1_t.parameter<float>("lr_th").set(th_lr);
			conv1_t.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
			conv1_t.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
			conv1_t.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 1.0);

			auto &conv1_t_out = experiment_time.output<TimeObjectiveOutput>(conv1_t, t_obj);
			conv1_t_out.add_postprocessing<process::SumPooling>(20, 20);
			conv1_t_out.add_postprocessing<process::TemporalPooling>(2);
			// conv1_t_out.add_postprocessing<process::FeatureScaling>();
			conv1_t_out.add_analysis<analysis::Activity>();
			conv1_t_out.add_analysis<analysis::Coherence>();
			conv1_t_out.add_analysis<analysis::Svm>();

			experiment_time.run(10000);

			//------------- FUSION -------------
			// add code to re-fill the saved values from the text files and fuse them here.
			std::string _file_path = std::filesystem::current_path();

			experiment_fused.add_train<dataset::TwoStream>(_file_path + "/ExtractedFeatures/" + experiment_space.name() + "/train/", _method, _draw);
			experiment_fused.add_test<dataset::TwoStream>(_file_path + "/ExtractedFeatures/" + experiment_space.name() + "/test/", _method, _draw);

			auto &svm = experiment_fused.push<layer::Stream>(1, 1, 1, filter_number);

			// add another SVM step to classify the fused result.
			auto &fused_out = experiment_fused.output<NoOutputConversion>(svm);
			fused_out.add_analysis<analysis::Svm>();

			experiment_fused.run(10000);
		}
}