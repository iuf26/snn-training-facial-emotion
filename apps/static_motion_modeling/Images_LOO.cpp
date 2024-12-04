#include "Experiment.h"
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
	std::string subjects[9] = {"daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"}; // for (int _repeat = 0; _repeat < 5; _repeat++){
	for (std::string subject : subjects)
	{
		std::string _dataset = "Image_w";
		// The new dimentions of a video frame, set to zero if default dimentions are needed.
		size_t _frame_size_width = 0;
		size_t _frame_size_height = 0;

		size_t _temporal_depth = 1;
		// The name of the experiment is tha name of the dataset, this name is used for the log text file.
		Experiment<SparseIntermediateExecutionNew> experiment(argc, argv, _dataset, false, false);
		// experiment.push<process::SimplePreprocessing>(experiment.name(), 1);
		// experiment.push<process::EarlyFusion>(experiment.name(), 1, 4);

		experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
		const char *input_path_ptr = std::getenv("INPUT_PATH");

		if (input_path_ptr == nullptr)
			throw std::runtime_error("Require to define INPUT_PATH variable");

		// Get the input path of the dataset.
		std::string input_path(input_path_ptr);

		// Temporal coding used, the data value "float" is transformed and reprisented by the spike timestamp.
		experiment.push<LatencyCoding>();

		// The location of the dataset images, seperated into train and test folders that contain labeled folders of images.
		experiment.add_train<dataset::Image>(input_path  + "/" + subject + "/train", _temporal_depth);
		experiment.add_test<dataset::Image>(input_path + "/" + subject + "/test", _temporal_depth);

		float t_obj = 0.75;
		float t_obj1 = 0.75;
		float t_obj2 = 0.75;

		float th_lr = 1.0f;
		float w_lr = 0.1f;

		// First convolution layer, name, width, hight, depth of filter(NUMBER OF FEATURES nf - filter number). stride x and y are fixed to 1 which means there is a big overlapping while extracting features.
		// This function takes the following(Layer Name, Kernel width, kernel height, number of kernels, and a flag to draw the weights if 1 or not if 0)
		auto &conv1 = experiment.push<layer::Convolution>(5, 5, 16);
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

		auto &pool1 = experiment.push<layer::Pooling>(2, 2, 2, 2);
		pool1.set_name("pool1");

		auto &conv2 = experiment.push<layer::Convolution>(5, 5, 16);
		conv2.set_name("conv2");
		conv2.parameter<bool>("draw").set(false);
		conv2.parameter<bool>("save_weights").set(true);
		conv2.parameter<bool>("inhibition").set(true);
		conv2.parameter<uint32_t>("epoch").set(100);
		conv2.parameter<float>("annealing").set(0.95f);
		conv2.parameter<float>("min_th").set(1.0f);
		conv2.parameter<float>("t_obj").set(t_obj1);
		conv2.parameter<float>("lr_th").set(th_lr);
		conv2.parameter<bool>("wta_infer").set(false);
		conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
		conv2.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		auto &pool2 = experiment.push<layer::Pooling>(2, 2, 2, 2);
		pool2.set_name("pool2");

		auto &fc1 = experiment.push<layer::Convolution>(5, 5, 16);
		fc1.set_name("fc1");
		fc1.parameter<bool>("draw").set(false);
		fc1.parameter<bool>("save_weights").set(true);
		fc1.parameter<bool>("inhibition").set(true);
		fc1.parameter<uint32_t>("epoch").set(100);
		fc1.parameter<float>("annealing").set(0.95f);
		fc1.parameter<float>("min_th").set(1.0f);
		fc1.parameter<float>("t_obj").set(t_obj2);
		fc1.parameter<float>("lr_th").set(th_lr);
		fc1.parameter<bool>("wta_infer").set(false);
		fc1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		fc1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
		fc1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

#ifdef ENABLE_QT
		conv1.plot_threshold(true);
		conv1.plot_reconstruction(true);
#endif

		auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
		conv1_out.add_postprocessing<process::SumPooling>(10, 10);
		conv1_out.add_postprocessing<process::FeatureScaling>();
		conv1_out.add_analysis<analysis::Activity>();
		conv1_out.add_analysis<analysis::Coherence>();
		conv1_out.add_analysis<analysis::Svm>();

		auto &conv2_out = experiment.output<TimeObjectiveOutput>(conv2, t_obj1);
		conv2_out.add_postprocessing<process::SumPooling>(10, 10);
		conv2_out.add_postprocessing<process::FeatureScaling>();
		conv2_out.add_analysis<analysis::Activity>();
		conv2_out.add_analysis<analysis::Coherence>();
		conv2_out.add_analysis<analysis::Svm>();

		auto &fc1_out = experiment.output<TimeObjectiveOutput>(fc1, t_obj2);
		fc1_out.add_postprocessing<process::SumPooling>(10, 10);
		fc1_out.add_postprocessing<process::FeatureScaling>();
		fc1_out.add_analysis<analysis::Activity>();
		fc1_out.template add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}