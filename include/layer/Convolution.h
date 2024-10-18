#ifndef _CONVOLTUION_H
#define _CONVOLTUION_H

#include "Layer.h"
#include "Stdp.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include "tool/Operations.h"
#include "plot/Threshold.h"
#include "plot/Evolution.h"
// #include <execution>
// #include <mutex>
/**
 * @brief A layer can be 2D or 3D depending on the type of convolution chosen by the user.
 * Convolution is a type of filtering applied to a certain input, it extracts certain features, the number of features = the number of filters.
 * In the firsqt layer, the features are low level (curve, color, etc...) 
 * If you add more layers, you can get more sophisticated features(eyes, nose, etc...)
*/
namespace layer
{

	class Convolution;

	namespace _priv
	{

#ifdef SMID_AVX256
		class ConvolutionImpl
		{

		public:
			ConvolutionImpl(Convolution &model);

			void resize();
			void train(const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
			void test(const std::vector<Spike> &input_spike, const Tensor<Time> &, std::vector<Spike> &output_spike);

		private:
			Convolution &_model;
			Tensor<float> _a;
			Tensor<float> _inh;
			Tensor<bool> _wta;
		};

#else
		class ConvolutionImpl
		{

		public:
			ConvolutionImpl(Convolution &model);

			void resize();
			void train(const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
			void train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
			void test(const std::vector<Spike> &input_spike, const Tensor<Time> &, std::vector<Spike> &output_spike);

		private:
			Convolution &_model;
			std::string _label; // the label of the cuttent sample
			Tensor<float> _a;
			Tensor<bool> _inh;
			Tensor<bool> _wta;
		};
#endif
	}

	/**
	 * @brief Convolution is a type of filtering applied on an input image, extracts certain features depending on a number of filters.
	 * the features can be low level (curve, color, etc...) or more sophisticated features(eyes, nose, etc...). The number of features is equal to the number of filters (kernels).
	 * 
	 * @param filter_width size_t - the width of the convolutional kernel
	 * @param filter_height size_t - the height of the convolutional kernel
	 * @param filter_number size_t - the number of the convolutional kernels, how many filters per layer
	 * 
  	 * @param stride_x size_t - The step of the convolutional filter in the x diresction
	 * @param stride_y size_t - The step of the convolutional filter in the y diresction
	 * @param padding_x size_t - added padding to the filter in the x direction
	 * @param padding_y size_t - added padding to the filter in the y direction
	 */
	class Convolution : public Layer3D
	{

		friend class _priv::ConvolutionImpl;

	public:
		Convolution();
		Convolution(size_t filter_width, size_t filter_height, size_t filter_number = 6, size_t stride_x = 1, size_t stride_y = 1, size_t padding_x = 0, size_t padding_y = 0);
		Convolution(const Convolution &that) = delete;
		Convolution &operator=(const Convolution &that) = delete;

		virtual Shape compute_shape(const Shape &previous_shape);

		virtual size_t train_pass_number() const;
		virtual void process_train_sample(const std::string &label, Tensor<float> &sample, size_t current_pass, size_t current_index, size_t number);
		virtual void process_test_sample(const std::string &label, Tensor<float> &sample, size_t current_index, size_t number);

		virtual void train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
		virtual void test(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
		virtual void on_epoch_end();

		virtual Tensor<float> reconstruct(const Tensor<float> &t) const;

		void plot_threshold(bool only_in_train);
		void plot_evolution(bool only_in_train);

		virtual bool load_params(const std::string& filename);
		virtual bool save_params(const std::string& filename);
	private:
		uint32_t _epoch_number;

		uint32_t _current_epoch_number;
		uint32_t _last_epoch_number;
		int _sample_type_count;
		uint32_t _sample_number;
		uint32_t _sample_count;
		uint32_t _drawn_weights;
		std::string _file_path;

		float _annealing;

		float _min_th;
		float _t_obj;
		float _lr_th;

		bool _draw;
		bool _save_weights;
		bool _inhibition;
		
		Tensor<float> _w;
		Tensor<float> _th;
		STDP *_stdp;
		size_t _input_depth;
		size_t _input_conv_depth;

		bool _wta_infer;

		_priv::ConvolutionImpl _impl;
	};
}
#endif
