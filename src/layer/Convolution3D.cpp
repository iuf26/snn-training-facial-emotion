#include "layer/Convolution3D.h"
#include "Experiment.h"
#include <execution>
#include <mutex>

using namespace layer;

static RegisterClassParameter<Convolution3D, LayerFactory> _register("Convolution3D");
/**
 * Convolution3D is a type of filter applied on an input immage,
 * @param annealing used as a faactor ro update other paremeteers later
 * @param min_th minimum threashould of a neuron
 * @param t_obj t expected, time expeected for a neuron to fire
 * @param lr_th learning rate of the neuron threshould, in order to have some kind of homeostasious the threshould of the neurons are updated using a certyian delta threshould.
 * @param w synapctic weights
 * @param stdp learning rule - spike time dependant plasticity
 */
Convolution3D::Convolution3D() : Layer4D(_register),
								 _inhibition(true), _model_path(""), _draw(false), _epoch_number(0), _annealing(1.0), _min_th(0), _t_obj(0), _lr_th(0),
								 _w(), _th(), _stdp(nullptr), _input_depth(0), _input_conv_depth(0), _impl(*this)
{
	add_parameter("draw", _draw);
	add_parameter("save_weights", _save_weights);
	add_parameter("save_random_start", _save_random_start);
	add_parameter("log_spiking_neuron", _log_spiking_neuron);
	add_parameter("inhibition", _inhibition);
	add_parameter("epoch", _epoch_number);
	add_parameter("annealing", _annealing, 1.0f); // Simulated annealing is a method for solving unconstrained and bound-constrained optimization problems. this parameter is used to modify other parameters
	add_parameter("min_th", _min_th);			  // minimum threashould
	add_parameter("t_obj", _t_obj);				  // time object
	add_parameter("lr_th", _lr_th);				  // delta threshould
	add_parameter("w", _w);						  // synaptic weights
	add_parameter("th", _th);					  // internal threashould of neuron
	add_parameter("stdp", _stdp);				  // learning rule - spike time dependant plasticity
}

Convolution3D::Convolution3D(size_t filter_number, size_t filter_width, size_t filter_height, size_t filter_depth, std::string model_path,
							 size_t stride_x, size_t stride_y, size_t stride_k, size_t padding_x, size_t padding_y, size_t padding_k)
	: Layer4D(_register, filter_number, filter_width, filter_height, filter_depth, stride_x, stride_y, stride_k, padding_x, padding_y, padding_k),
	  _inhibition(true), _model_path(model_path), _draw(false), _save_weights(false), _save_random_start(false), _log_spiking_neuron(false), _annealing(1.0),
	  _min_th(0), _t_obj(0), _lr_th(0), _sample_number(0), _sample_count(0), _spike_count(0), _drawn_weights(0), _saved_weights(0), _logged_spiking_neuron(0), _saved_random_start(0),
	  _w(), _th(), _stdp(nullptr), _input_depth(0), _impl(*this)
{
	add_parameter("draw", _draw);
	add_parameter("save_weights", _save_weights);
	add_parameter("save_random_start", _save_random_start);
	add_parameter("log_spiking_neuron", _log_spiking_neuron);
	add_parameter("inhibition", _inhibition);
	add_parameter("epoch", _epoch_number);
	add_parameter("annealing", _annealing, 1.0f);
	add_parameter("min_th", _min_th);
	add_parameter("t_obj", _t_obj);
	add_parameter("lr_th", _lr_th);
	add_parameter("w", _w);
	add_parameter("th", _th);
	add_parameter("stdp", _stdp);

	// _patch_coo_collection = false;

	// for (size_t i = 0; i < experiment()->process_number(); i++)
	// 	if (experiment()->process_at(i).name() == "PatchCoordinates")
	// 		_patch_coo_collection = true;

	_file_path = std::filesystem::current_path();
}

/**
 * @brief The convolutional layer is initialised with random weights. The w tensor takes the same dimentions as the filerts.
 *
 * @param previous_shape
 * @param random_generator
 */

Shape Convolution3D::compute_shape(const Shape &previous_shape)
{
	Layer4D::compute_shape(previous_shape);

	_input_depth = previous_shape.dim(2);
	_input_conv_depth = previous_shape.number() > 3 ? previous_shape.dim(3) : 1;
	// width, height, channels, filterNumber, temporalDepth]
	parameter<Tensor<float>>("w").shape(_filter_width, _filter_height, _input_depth, _filter_number, _filter_conv_depth);
	parameter<Tensor<float>>("th").shape(_filter_number);

	_impl.resize();
	// TODO: _conv_depth or filter_depth here?
	return Shape({_width, _height, _depth, _conv_depth});
}

/**
 * @brief This number is different relative to process
 * The train_pass_number gives the number of epochs if the process is convolution.
 *
 * @return size_t number of epochs
 */
size_t Convolution3D::train_pass_number() const
{
	return _epoch_number + 1;
}

void Convolution3D::process_train_sample(const std::string &label, Tensor<float> &sample, size_t current_pass, size_t current_index, size_t number)
{
	// The training
	if (current_index == 0)
	{
		if (current_pass < _epoch_number)
		{
			_current_epoch_number = current_pass;
			_current_width = 1;
			_current_height = 1;
			_current_conv_depth = 1;
			std::cout << "\rEpoch " << current_pass << "/" << _epoch_number;

			on_epoch_start();
		}
		else
		{
			_current_width = _width;
			_current_height = _height;
			_current_conv_depth = _conv_depth;
			std::cout << std::endl
					  << "Process train set" << std::endl;
		}
	}

	std::vector<Spike> input_spike;
	std::vector<Spike> output_spike;

	if (current_pass < _epoch_number)
	{
		size_t x = 0;
		size_t y = 0;
		size_t z = 0;
		size_t k = 0;
		float t = 0.0;
		// size_t _empty_sample_count = 0;
		// bool _sample_contain_info = Tensor<float>::tensor_contain_info(sample);

		// do // take the random patches around places where a spike exists
		// {

		if (_filter_width < _width)
		{
			std::uniform_int_distribution<size_t> rand_x(0, _width - _filter_width);
			x = rand_x(experiment()->random_generator());
		}
		if (_filter_height < _height)
		{
			std::uniform_int_distribution<size_t> rand_y(0, _height - _filter_height);
			y = rand_y(experiment()->random_generator());
		}
		if (_filter_conv_depth < _conv_depth)
		{
			std::uniform_int_distribution<size_t> rand_y(0, _conv_depth - _filter_conv_depth);
			k = rand_y(experiment()->random_generator());
		}

		std::uniform_int_distribution<size_t> rand_z(0, _input_depth - 1);
		z = rand_z(experiment()->random_generator());
		t = sample.at(x, y, z, k);

		// if (!_sample_contain_info)
		// {
		// 	_empty_sample_count++;
		// 	break;
		// }
		// } while (t == 0.0 || t > 1); //(t > 0.0 && t < 1);

		// even if _filter_conv_depth == 1, we are still taking random patches with a temporal depth.
		Tensor<Time> input_time(Shape({_filter_width, _filter_height, _input_depth, _filter_conv_depth}));
		for (size_t cx = 0; cx < _filter_height; cx++)
		{
			for (size_t cy = 0; cy < _filter_width; cy++)
			{
				for (size_t cz = 0; cz < _input_depth; cz++)
				{
					for (size_t ck = 0; ck < _filter_conv_depth; ck++)
					{
						input_time.at(cx, cy, cz, ck) = sample.at(cx + x, cy + y, cz, ck + k);
					}
				}
			}
		}

		SpikeConverter::to_spike(input_time, input_spike);
		train(label, input_spike, input_time, output_spike);
	}
	else
	{
		SpikeConverter::to_spike(sample, input_spike);
		_sample_number = number;
		test(label, input_spike, sample, output_spike);
		sample = Tensor<float>(shape());
		SpikeConverter::from_spike(output_spike, sample);
	}

	if (current_index == number - 1 && current_pass < _epoch_number)
	{
		on_epoch_end();
	}
}

void Convolution3D::process_test_sample(const std::string &label, Tensor<float> &sample, size_t current_index, size_t number)
{
	if (current_index == 0)
	{
		std::cout << "Process test set" << std::endl;
		_current_width = _width;
		_current_height = _height;
		_current_conv_depth = _conv_depth;
	}

	std::vector<Spike> input_spike;
	SpikeConverter::to_spike(sample, input_spike);
	std::vector<Spike> output_spike;
	_sample_number = number;
	test(label, input_spike, sample, output_spike);
	sample = Tensor<float>(shape());
	SpikeConverter::from_spike(output_spike, sample);
}

void Convolution3D::train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike)
{
	_impl.train(label, input_spike, input_time, output_spike);
}

void Convolution3D::test(const std::string &, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike)
{
	_impl.test(input_spike, input_time, output_spike);
}

void Convolution3D::on_epoch_end()
{
	_lr_th *= _annealing;
	_stdp->adapt_parameters(_annealing);
}

// This function is not extended because it's only for drawing.
Tensor<float> Convolution3D::reconstruct(const Tensor<float> &t) const
{
	size_t ki = 1;

	size_t output_width = t.shape().dim(0);
	size_t output_height = t.shape().dim(1);
	size_t output_depth = t.shape().dim(2);
	size_t output_conv_depth = t.shape().number() > 3 ? t.shape().dim(3) : 1;

	Tensor<float> out(Shape({output_width * _stride_x + _filter_width - 1, output_height * _stride_y + _filter_height - 1, _input_depth}));
	out.fill(0);

	Tensor<float> norm(Shape({output_width * _stride_x + _filter_width - 1, output_height * _stride_y + _filter_height - 1, _input_depth}));
	norm.fill(0);

	for (size_t k = 0; k < output_conv_depth; k++)
		for (size_t x = 0; x < output_width; x++)
			for (size_t y = 0; y < output_height; y++)
			{

				std::vector<size_t> is;
				for (size_t z = 0; z < output_depth; z++)
				{
					is.push_back(z);
				}

				if (t.shape().number() > 3)
					std::sort(std::begin(is), std::end(is), [&t, x, y, k](size_t i1, size_t i2)
							  { return t.at(x, y, i1, k) > t.at(x, y, i2, k); });
				else
					std::sort(std::begin(is), std::end(is), [&t, x, y](size_t i1, size_t i2)
							  { return t.at(x, y, i1) > t.at(x, y, i2); });

				for (size_t i = 0; i < ki; i++)
				{
					if (t.at(x, y, is[i]) >= 0.0)
					{
						for (size_t xf = 0; xf < _filter_width; xf++)
						{
							for (size_t yf = 0; yf < _filter_height; yf++)
							{
								for (size_t zf = 0; zf < _input_depth; zf++)
								{
									if (t.shape().number() > 3)
									{
										out.at(x * _stride_x + xf, y * _stride_y + yf, zf) += _w.at(xf, yf, zf, is[i], k) * t.at(x, y, is[i], k);
										norm.at(x * _stride_x + xf, y * _stride_y + yf, zf) += t.at(x, y, is[i], k);
									}
									else
									{
										out.at(x * _stride_x + xf, y * _stride_y + yf, zf) += _w.at(xf, yf, zf, is[i], k) * t.at(x, y, is[i]);
										norm.at(x * _stride_x + xf, y * _stride_y + yf, zf) += t.at(x, y, is[i]);
									}
								}
							}
						}
					}
				}
			}

	size_t s = out.shape().product();
	for (size_t i = 0; i < s; i++)
	{
		if (norm.at_index(i) != 0)
			out.at_index(i) /= norm.at_index(i);
	}

	out.range_normalize();

	return out;
}

Tensor<float> Convolution3D::construct_features(const Tensor<float> &t) const
{
	size_t output_width = t.shape().dim(0);
	size_t output_height = t.shape().dim(1);
	size_t output_depth = t.shape().dim(2);
	size_t output_conv_depth = t.shape().dim(3);

	Tensor<float> out(Shape({output_width * _stride_x + _filter_width - 1, output_height * _stride_y + _filter_height - 1, _input_depth,
							 output_conv_depth * _stride_k + _filter_conv_depth - 1}));
	out.fill(0);

	// Save the out Tensor, it is the drawn sample.
	return out;
}

#ifdef ENABLE_QT
void Convolution3D::plot_threshold(bool only_in_train)
{
	add_plot<plot::Threshold>(only_in_train, _th);
}

void Convolution3D::plot_evolution(bool only_in_train)
{
	add_plot<plot::Evolution>(only_in_train, _w);
}
#endif

#ifdef SMID_AVX256
#include <immintrin.h>

#define AVX_256_N 8

static __m256i _generate_mask(int n)
{

	int f = 0x00000000;
	int t = 0xFFFFFFFF;

	switch (n)
	{

	case 0:
		return _mm256_setr_epi32(f, f, f, f, f, f, f, f);
	case 1:
		return _mm256_setr_epi32(t, f, f, f, f, f, f, f);
	case 2:
		return _mm256_setr_epi32(t, t, f, f, f, f, f, f);
	case 3:
		return _mm256_setr_epi32(t, t, t, f, f, f, f, f);
	case 4:
		return _mm256_setr_epi32(t, t, t, t, f, f, f, f);
	case 5:
		return _mm256_setr_epi32(t, t, t, t, t, f, f, f);
	case 6:
		return _mm256_setr_epi32(t, t, t, t, t, t, f, f);
	case 7:
		return _mm256_setr_epi32(t, t, t, t, t, t, t, f);
	default:
		return _mm256_setr_epi32(t, t, t, t, t, t, t, t);
	}
}

_priv::Convolution3DImpl::Convolution3DImpl(Convolution3D &model) : _model(model), _a(), _inh(), _wta()
{
}

void _priv::Convolution3DImpl::resize()
{
	_a = Tensor<float>(Shape({_model.width(), _model.height(), _model.depth()}));
	_inh = Tensor<float>(Shape({_model.width(), _model.height(), _model.depth()}));
	_wta = Tensor<bool>(Shape({_model.width(), _model.height()}));
}

void _priv::Convolution3DImpl::train(const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &)
{

	size_t depth = _model.depth();

	Tensor<float> &w = _model._w;
	Tensor<float> &th = _model._th;

	size_t n = depth / AVX_256_N;
	size_t r = depth % AVX_256_N;

	__m256 __c1 = _mm256_setzero_ps();
	__m256 __c2 = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF));

	__m256i __mask = _generate_mask(r);

	for (size_t i = 0; i < n; i++)
	{
		_mm256_storeu_ps(_a.ptr_index(i * AVX_256_N), __c1);
	}

	if (r > 0)
	{
		_mm256_maskstore_ps(_a.ptr_index(n * AVX_256_N), __mask, __c1);
	}

	for (const Spike &spike : input_spike)
	{

		for (size_t i = 0; i < n; i++)
		{
			__m256 __w = _mm256_loadu_ps(w.ptr(spike.x, spike.y, spike.z, spike.k, i * AVX_256_N));
			__m256 __a = _mm256_loadu_ps(_a.ptr_index(i * AVX_256_N));
			__m256 __th = _mm256_loadu_ps(th.ptr_index(i * AVX_256_N));
			__a = _mm256_add_ps(__a, __w);
			_mm256_storeu_ps(_a.ptr_index(i * AVX_256_N), __a);
			__m256 __c = _mm256_cmp_ps(__a, __th, _CMP_GE_OQ);

			if (_mm256_testz_ps(__c, __c2) == 0)
			{

				for (size_t j = 0; j < AVX_256_N; j++)
				{
					if (_a.at_index(i * AVX_256_N + j) > th.at_index(i * AVX_256_N + j))
					{
						for (size_t z1 = 0; z1 < depth; z1++)
						{
							th.at(z1) -= _model._lr_th * (spike.time - _model._t_obj);
							if (z1 != i * AVX_256_N + j)
							{
								th.at(z1) -= _model._lr_th / static_cast<float>(depth - 1);
							}
							else
							{
								th.at(z1) += _model._lr_th;
							}
							th.at(z1) = std::max<float>(_model._min_th, th.at(z1));
						}

						for (size_t x = 0; x < _model._filter_width; x++)
						{
							for (size_t y = 0; y < _model._filter_height; y++)
							{
								for (size_t z = 0; z < _model._input_depth; z++)
								{
									w.at(x, y, z, i * AVX_256_N + j) = _model._stdp->process(w.at(x, y, z, i * AVX_256_N + j), input_time.at(x, y, z), spike.time);
								}
							}
						}

						return;
					}
				}
			}
		}

		if (r > 0)
		{
			__m256 __w = _mm256_maskload_ps(w.ptr(spike.x, spike.y, spike.z, spike.k, n * AVX_256_N), __mask);
			__m256 __a = _mm256_maskload_ps(_a.ptr_index(n * AVX_256_N), __mask);
			__a = _mm256_add_ps(__a, __w);
			_mm256_maskstore_ps(_a.ptr_index(n * AVX_256_N), __mask, __a);

			for (size_t j = 0; j < r; j++)
			{
				if (_a.at_index(n * AVX_256_N + j) > th.at_index(n * AVX_256_N + j))
				{
					for (size_t z1 = 0; z1 < depth; z1++)
					{
						th.at(z1) -= _model._lr_th * (spike.time - _model._t_obj);
						if (z1 != n * AVX_256_N + j)
						{
							th.at(z1) -= _model._lr_th / static_cast<float>(depth - 1);
						}
						else
						{
							th.at(z1) += _model._lr_th;
						}
						th.at(z1) = std::max<float>(_model._min_th, th.at(z1));
					}

					for (size_t x = 0; x < _model._filter_width; x++)
					{
						for (size_t y = 0; y < _model._filter_height; y++)
						{
							for (size_t z = 0; z < _model._input_depth; z++)
							{
								w.at(x, y, z, n * AVX_256_N + j) = _model._stdp->process(w.at(x, y, z, n * AVX_256_N + j), input_time.at(x, y, z), spike.time);
							}
						}
					}
					return;
				}
			}
		}
	}
}

void _priv::Convolution3DImpl::test(const std::vector<Spike> &input_spike, const Tensor<Time> &, std::vector<Spike> &output_spike)
{
	size_t depth = _model.depth();
	Tensor<float> &w = _model._w;
	Tensor<float> &th = _model._th;

	size_t n = depth / AVX_256_N;
	size_t r = depth % AVX_256_N;

	__m256 __c1 = _mm256_setzero_ps();

	uint32_t mask = 0xFFFFFFFF;

	__m256i __mask = _generate_mask(r);

	for (size_t x = 0; x < _model._current_width; x++)
	{
		for (size_t y = 0; y < _model._current_height; y++)
		{
			for (size_t i = 0; i < n; i++)
			{
				_mm256_storeu_ps(_a.ptr(x, y, i * AVX_256_N), __c1);
				_mm256_storeu_ps(_inh.ptr(x, y, i * AVX_256_N), __c1);
			}

			if (r > 0)
			{
				_mm256_maskstore_ps(_a.ptr(x, y, n * AVX_256_N), __mask, __c1);
				_mm256_maskstore_ps(_inh.ptr(x, y, n * AVX_256_N), __mask, __c1);
			}
		}
	}

	if (_model._wta_infer)
	{
		_wta.fill(false);
	}

	for (const Spike &spike : input_spike)
	{

		std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>> output_spikes;
		_model.forward(spike.x, spike.y, spike.k, output_spikes);

		for (const auto &entry : output_spikes)
		{
			uint16_t x = std::get<0>(entry);
			uint16_t y = std::get<1>(entry);
			uint16_t w_x = std::get<2>(entry);
			uint16_t w_y = std::get<3>(entry);

			if (_model._wta_infer && _wta.at(x, y))
			{
				continue;
			}

			for (size_t i = 0; i < n; i++)
			{
				__m256 __w = _mm256_loadu_ps(w.ptr(w_x, w_y, spike.z, spike.k, i * AVX_256_N));
				__m256 __a = _mm256_loadu_ps(_a.ptr(x, y, i * AVX_256_N));
				__m256 __th = _mm256_loadu_ps(th.ptr(i * AVX_256_N));
				__a = _mm256_add_ps(__a, __w);
				_mm256_storeu_ps(_a.ptr(x, y, i * AVX_256_N), __a);
				__m256 __c = _mm256_cmp_ps(__a, __th, _CMP_GE_OQ);

				__m256 __m = _mm256_loadu_ps(_inh.ptr(x, y, i * AVX_256_N));

				if (_mm256_testc_ps(__m, __c) == 0)
				{
					for (size_t j = 0; j < AVX_256_N; j++)
					{
						if (_a.at(x, y, i * AVX_256_N + j) >= th.at_index(i * AVX_256_N + j) && *reinterpret_cast<uint32_t *>(_inh.ptr(x, y, i * AVX_256_N + j)) != mask)
						{
							_wta.at(x, y) = true;
							output_spike.emplace_back(spike.time, x, y, i * AVX_256_N + j);
						}
					}

					_mm256_storeu_ps(_inh.ptr(x, y, i * AVX_256_N), __c);
				}
			}

			if (r > 0)
			{
				__m256 __w = _mm256_maskload_ps(w.ptr(w_x, w_y, spike.z, spike.k, n * AVX_256_N), __mask);
				__m256 __a = _mm256_maskload_ps(_a.ptr(x, y, n * AVX_256_N), __mask);
				__m256 __th = _mm256_maskload_ps(th.ptr(n * AVX_256_N), __mask);
				__a = _mm256_add_ps(__a, __w);
				_mm256_maskstore_ps(_a.ptr(x, y, n * AVX_256_N), __mask, __a);
				__m256 __c = _mm256_cmp_ps(__a, __th, _CMP_GE_OQ);

				for (size_t j = 0; j < r; j++)
				{
					if (_a.at(x, y, n * AVX_256_N + j) >= th.at_index(n * AVX_256_N + j) && *reinterpret_cast<uint32_t *>(_inh.ptr(x, y, n * AVX_256_N + j)) != mask)
					{
						_wta.at(x, y) = true;
						output_spike.emplace_back(spike.time, x, y, n * AVX_256_N + j);
					}
				}

				_mm256_maskstore_ps(_inh.ptr(x, y, n * AVX_256_N), __mask, __c);
			}
		}
	}
}

#else
_priv::Convolution3DImpl::Convolution3DImpl(Convolution3D &model) : _model(model), _a(), _inh()
{
}

/**
 * @brief Decides the size of the convolution layer.
 *
 * @param _a A tensor of the activations of all the neurons in the layer.
 * @param _inh A tensor of the inhibition values of all the neurons in the layer.
 */
void _priv::Convolution3DImpl::resize()
{
	// these values are the total size of the convolutional layer.
	_a = Tensor<float>(Shape({_model.width(), _model.height(), _model.depth(), _model.conv_depth()}));
	_inh = Tensor<bool>(Shape({_model.width(), _model.height(), _model.depth(), _model.conv_depth()}));
}

void _priv::Convolution3DImpl::train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time,
									 std::vector<Spike> &output_spike)
{
	this->_label = label;
	this->train(input_spike, input_time, output_spike);
}

/**
 * @brief This function is the core of the training that happrns in a convolutional SNN.
 * In this function, the values of the weights, threshoulds and activations of the SNN are updated.
 *
 * @param input_spike
 * @param input_time
 * @param output_spike
 */

void _priv::Convolution3DImpl::train(const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike)
{
	std::mutex _convolution_train_mutex; // mutex to aviod access violation durring multithreaded section
	///////////////////////////////
	std::string delimiter = ";.";
	std::string _exp_name = _label.substr(0, _label.find(delimiter));
	_model._exp_name = _exp_name;
	_label.erase(0, _exp_name.length() + delimiter.length());
	std::string _layerIndex = _label.substr(0, _label.find(delimiter));
	_label.erase(0, _layerIndex.length() + delimiter.length());
	if (_model._draw || _model._log_spiking_neuron || _model._save_weights || _model._save_random_start)
		std::filesystem::create_directories(_model._file_path + "/Weights/" + _exp_name + "/" + _layerIndex + "/");

	if (_model._current_epoch_number == 0 && _model._save_random_start && _model._saved_random_start == 0)
	{
		SaveWeights(_model._file_path + "/Weights/" + _exp_name + "/" + _layerIndex + "/" + _exp_name + "_random_start.json", _label, _model._w);
		_model._saved_random_start = 1;
	}

	size_t depth = _model.depth();
	size_t conv_depth = _model.conv_depth();
	Tensor<float> &w = _model._w;
	Tensor<float> &th = _model._th;

	std::fill(std::begin(_a), std::end(_a), 0);

	for (const Spike &spike : input_spike)
	{
		for (size_t z = 0; z < depth; z++) // the number of filters
		{
			_a.at(0, 0, z, 0) += w.at(spike.x, spike.y, spike.z, z, spike.k);

			// integrate the weight value in the neurons activation (multiple spikes are integrated to surpass the internal threshould of the neuron)
			if (_a.at(0, 0, z, 0) >= th.at(z)) // a spike is fired
			{
				for (size_t z1 = 0; z1 < depth; z1++)
				{
					th.at(z1) -= _model._lr_th * (spike.time - _model._t_obj);

					if (z1 != z)
						th.at(z1) -= _model._lr_th / static_cast<float>(depth - 1);
					else
						th.at(z1) += _model._lr_th;

					th.at(z1) = std::max<float>(_model._min_th, th.at(z1));
				}

				for (size_t x = 0; x < _model._filter_width; x++)
					for (size_t y = 0; y < _model._filter_height; y++)
						for (size_t zi = 0; zi < _model._input_depth; zi++)
							for (size_t k = 0; k < _model._filter_conv_depth; k++)
							{
								w.at(x, y, zi, z, k) = _model._stdp->process(w.at(x, y, zi, z, k), input_time.at(x, y, zi, k), spike.time);
							}

				/// @brief for visualization.
				if (_model._current_epoch_number == _model._epoch_number - 1 && _model._draw && _model._drawn_weights == 0)
				{
					Tensor<float>::draw_weight_tensor(_model._file_path + "/Weights/" + _exp_name + "/" + _layerIndex + "/" + _exp_name + "_N:" + std::to_string(z), w);
					_model._drawn_weights = 1;
				}

				if (_model._log_spiking_neuron)
				{
					if (_model._logged_spiking_neuron == 0)
					{
						std::filesystem::create_directories(_model._file_path + "/UpdatedFilter/" + _model._exp_name + "/");
						_model._logged_spiking_neuron = 1;
					}
					LogUpdatedFilter(_model._file_path + "/UpdatedFilter/" + _model._exp_name + "/" + _model._exp_name + "_" + _layerIndex, z);
				}

				if (_model._current_epoch_number == _model._epoch_number - 1 && _model._save_weights && _model._saved_weights == 0)
				{
					SaveWeights(_model._file_path + "/Weights/" + _exp_name + "/" + _layerIndex + "/" + _exp_name + ".json", _label, w);
					_model._saved_weights = 1;
				}

				if (_model._inhibition)
					return;
			}
		}
	}
}

void _priv::Convolution3DImpl::test(const std::vector<Spike> &input_spike, const Tensor<Time> &, std::vector<Spike> &output_spike)
{
	size_t depth = _model.depth();
	_model._sample_count++;

	Tensor<float> &w = _model._w;
	Tensor<float> &th = _model._th;

	std::fill(std::begin(_a), std::end(_a), 0);
	std::fill(std::begin(_inh), std::end(_inh), false);

	std::mutex _convolution_test_mutex; // mutex to aviod access violation durring multithreaded section

	// std::for_each(std::execution::par, input_spike.begin(), input_spike.end(), [&](const Spike &spike)
	for (const Spike &spike : input_spike)
	{
		std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>> output_spikes;
		_model.forward(spike.x, spike.y, spike.k, output_spikes);

		for (const auto &entry : output_spikes)
		{
			uint16_t x = std::get<0>(entry);
			uint16_t y = std::get<1>(entry);
			uint16_t k = std::get<2>(entry);
			uint16_t w_x = std::get<3>(entry);
			uint16_t w_y = std::get<4>(entry);
			uint16_t w_k = std::get<5>(entry);

			for (size_t z = 0; z < depth; z++)
			{
				// The neurons that have their inhibition flag set to ture are ignored.
				if (_inh.at(x, y, z, k) && _model._inhibition)
				{
					continue;
				}
				// The rest of the neurons that have their inh flag set to false get their activations updated.
				//_convolution_test_mutex.lock();
				_a.at(x, y, z, k) += w.at(w_x, w_y, spike.z, z, w_k);
				//_convolution_test_mutex.unlock();
				// If the activation crossed the threshould, the neuron has fired a spike, and it's _inh flag is set to true so that it doesn't fire again.
				if (_a.at(x, y, z, k) >= th.at(z))
				{
					//_convolution_test_mutex.lock();
					output_spike.emplace_back(spike.time, x, y, z, k);
					// The neuron that fires once is not allowed to fire again in this sample, so _inh is set to true.
					_inh.at(x, y, z, k) = true;
					//_convolution_test_mutex.unlock();

					/// @brief counting the spikes.
					_model._spike_count++;
					// std::cout << "\r[Spike count: " + std::to_string(_model._spike_count) + "]";
					// std::cout.flush();
				}
			}
		}
	}
	//});
	draw_progress(_model._sample_count, _model._sample_number);

	if (_model._sample_count == _model._sample_number)
	{
		std::cout << "\r[Spike count: " + std::to_string(_model._spike_count) + "] \n";
		std::filesystem::create_directories(_model._file_path + "/SpikeNumber/" + _model._exp_name + "/");
		LogSpikeNumber(_model._file_path + "/SpikeNumber/" + _model._exp_name + "/" + _model._exp_name, _model._spike_count);
		// experiment()->log()<< << "[Spike count: " << std::to_string(_model._spike_count) << "] \n";
		_model._sample_count = 0;
		_model._spike_count = 0;
	}
}

#endif
