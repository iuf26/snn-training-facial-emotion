#include "Layer.h"
#include "Experiment.h"

Layer::~Layer()
{
}

void Layer::set_size(size_t width, size_t height)
{
	if (width < 1 || height < 1)
	{
		throw std::runtime_error("Min size is 1x1");
	}
	if (width > _width || height > _height)
	{
		throw std::runtime_error("Size can't exceed layer dimension");
	}

	_current_width = width;
	_current_height = height;
	_current_conv_depth = 1;
}

void Layer::set_size(size_t width, size_t height, size_t conv_depth)
{
	if (width < 1 || height < 1 || conv_depth < 1)
	{
		throw std::runtime_error("Min size is 1x1x1");
	}
	if (width > _width || height > _height || conv_depth > _conv_depth)
	{
		throw std::runtime_error("Size can't exceed layer dimension");
	}

	_current_width = width;
	_current_height = height;
	_current_conv_depth = conv_depth;
}

size_t Layer::width() const
{
	return _width;
}

size_t Layer::height() const
{
	return _height;
}

size_t Layer::depth() const
{
	return _depth;
}

size_t Layer::conv_depth() const
{
	return _conv_depth;
}

bool Layer::require_sorted() const
{
	return _require_sorted;
}

#ifdef ENABLE_QT
void Layer::plot_time(bool only_in_train, size_t n, float min, float max) {
	add_plot<plot::TimeHistogram>(only_in_train, experiment(), index()+1, n, min, max);
}

void Layer::_add_plot(Plot* plot, bool only_in_train) {
	experiment()->add_plot(plot, only_in_train ? index() : -1);
}
#endif

std::vector<const Layer*> Layer::_previous_layer_list() const {
	std::vector<const Layer*> layers;
	for(int i=index(); i>=0; i--) {
		if(dynamic_cast<const Layer*>(&experiment()->process_at(i))) {
			layers.push_back(dynamic_cast<const Layer*>(&experiment()->process_at(i)));
		}
	}
	return layers;
}

//
//	Layer3D
//

Shape Layer3D::compute_shape(const Shape& previous_shape) {
	parameter<size_t>("filter_width").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("filter_height").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("filter_number").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("stride_x").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("stride_y").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("padding_x").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("padding_y").ensure_initialized(experiment()->random_generator());

	size_t previous_width = previous_shape.dim(0);
	size_t previous_height = previous_shape.dim(1);

	if(previous_width+2*_padding_x < _filter_width || previous_height+2*_padding_y<_filter_height) {
		throw std::runtime_error("Filter dimension need to be smaller than the input");
	}

	_width = (previous_width+2*_padding_x-_filter_width)/_stride_x+1;
	_height = (previous_height+2*_padding_y-_filter_height)/_stride_y+1;
	_depth = _filter_number;

	return Shape({_width, _height, _depth});
}

/**
 * @brief This function represents the forwarding action of sending a spike from one neuron to the other, (feed forward network FFN).
 * 
 * @complexity  
 * @param x_in the x coordinate of the spike
 * @param y_in the y coordinate of the spike
 * @param output 
 */
// Search for output neurons integrating input of position (x_in, y_in)
// For each involved output neuron, we get the position of the input in its window (i.e. the weight modulating the input)
void Layer3D::forward(uint16_t x_in, uint16_t y_in, std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>> &output)
{
	// The new coordinates of the spike during the convolution process.
	// This equation is calculating the new placement after applying the filter.
	// s_x means start_x coordicate, s_y means start_y coordicate.
	// if x_in = 21 & y_in = 5 then s_x = 17 & s_y = 1
	size_t s_x = x_in + _padding_x >= _filter_width - _stride_x ? (x_in + _padding_x - (_filter_width - _stride_x)) / _stride_x : 0;
	size_t s_y = y_in + _padding_y >= _filter_height - _stride_y ? (y_in + _padding_y - (_filter_height - _stride_y)) / _stride_y : 0;

	// l_x last neuron covered by the filter of the filter in the x direction.
	// if x_in = 21 & y_in = 5 then l_x = 21 & l_y = 5 without stride
	size_t l_x = (x_in + _padding_x) / _stride_x;
	size_t l_y = (y_in + _padding_y) / _stride_y;

	// from the position of the spike, and till the end of the filter OR the end of the layer.
	for (size_t x = s_x; x <= l_x && x < _current_width; x++)
	{
		for (size_t y = s_y; y <= l_y && y < _current_height; y++)
		{
			// w_x means weight_x coordicate
			size_t w_x = x_in + _padding_x - x * _stride_x;
			size_t w_y = y_in + _padding_y - y * _stride_y;

			output.emplace_back(x, y, w_x, w_y);
		}
	}
}

std::pair<uint16_t, uint16_t> Layer3D::to_input_coord(uint16_t x, uint16_t y, uint16_t w_x, uint16_t w_y) const
{
	if (x + w_x < _padding_x || y + w_y < _padding_y)
		return std::pair<uint16_t, uint16_t>(std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max());
	else
		return std::pair<uint16_t, uint16_t>(x + w_x - _padding_x, y + w_y - _padding_y);
}

bool Layer3D::is_valid_input_coord(const std::pair<uint16_t, uint16_t> &coord) const
{
	return coord.first != std::numeric_limits<uint16_t>::max();
}

std::pair<uint16_t, uint16_t> Layer3D::receptive_field_of(const std::pair<uint16_t, uint16_t> &in) const
{
	return std::pair<uint16_t, uint16_t>((in.first - 1) * _stride_x + _filter_width, (in.second - 1) * _stride_y + _filter_height);
}


//
//	Layer4D
//

Shape Layer4D::compute_shape(const Shape &previous_shape)
{
	parameter<size_t>("filter_number").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("filter_width").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("filter_height").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("filter_conv_depth").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("stride_x").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("stride_y").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("stride_k").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("padding_x").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("padding_y").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("padding_k").ensure_initialized(experiment()->random_generator());

	size_t previous_width = previous_shape.dim(0);
	size_t previous_height = previous_shape.dim(1);
	// we don't need channels number because after the first layer the channels have been corrilated, 
	// so it is safe to fill _depth with the filternumber
	// size_t previous_channels_number = previous_shape.dim(2);
	size_t previous_conv_depth = previous_shape.number() > 3 ? previous_shape.dim(3) : 1;

	if (previous_width + 2 * _padding_x < _filter_width || 
		previous_height + 2 * _padding_y < _filter_height) // || previous_conv_depth + 2 * _padding_k < _filter_conv_depth)
	{
		throw std::runtime_error("Filter dimension need to be smaller than the input");
	}
	// ((120 + (2*0) - 5)/1) +1 = 116
	_width = (previous_width + 2 * _padding_x - _filter_width) / _stride_x + 1;
	_height = (previous_height + 2 * _padding_y - _filter_height) / _stride_y + 1;
	
	_depth = _filter_number;
	_conv_depth = (previous_conv_depth + 2 * _padding_k - _filter_conv_depth) / _stride_k + 1;

	return Shape({_width, _height, _depth, _conv_depth});
}

/**
 * @brief This function represents the forwarding action of sending a spike from one neuron to the other, (feed forward network FFN).
 * 
 * @complexity  
 * @param x_in the x coordinate of the spike width
 * @param y_in the y coordinate of the spike height
 * @param k_in the k coordinate of the spike temporal depth
 * @param output the output of this function is output spikes of dimentions x, y, k, and weight values w_x, w_y, w_k
 */
void Layer4D::forward(uint16_t x_in, uint16_t y_in, uint16_t k_in, std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>> &output)
{
	// The new coordinates of the spike during the convolution process.
	// This equation is calculating the new placement after applying the filter.
	// s_x means spike_x coordicate, s_y means spike_y coordicate.
	size_t s_x = x_in + _padding_x >= _filter_width - _stride_x ? (x_in + _padding_x - (_filter_width - _stride_x)) / _stride_x : 0;
	size_t s_y = y_in + _padding_y >= _filter_height - _stride_y ? (y_in + _padding_y - (_filter_height - _stride_y)) / _stride_y : 0;
	size_t s_k = k_in + _padding_k >= _filter_conv_depth - _stride_k ? (k_in + _padding_k - (_filter_conv_depth - _stride_k)) / _stride_k : 0;

	// l_x number of neurons of convolutional kernel in the x y and k dimensions.
	size_t l_x = (x_in + _padding_x) / _stride_x;
	size_t l_y = (y_in + _padding_y) / _stride_y;
	size_t l_k = (k_in + _padding_k) / _stride_k;
	// from the position of the spike, and till the end of the cube OR the end of the layer.
	for (size_t x = s_x; x <= l_x && x < _current_width; x++)
		for (size_t y = s_y; y <= l_y && y < _current_height; y++)
			for (size_t k = s_k; k <= l_k && k < _current_conv_depth; k++)
			{
				// The new weights of the synapses where the spike came from.
				size_t w_x = x_in + _padding_x - x * _stride_x;
				size_t w_y = y_in + _padding_y - y * _stride_y;
				size_t w_k = k_in + _padding_k - k * _stride_k;

				output.emplace_back(x, y, k, w_x, w_y, w_k);
			}
}

std::tuple<uint16_t, uint16_t, uint16_t> Layer4D::to_input_coord(uint16_t x, uint16_t y, uint16_t k, uint16_t w_x, uint16_t w_y, uint16_t w_k) const
{
	if (x + w_x < _padding_x || y + w_y < _padding_y || k + w_k < _padding_k)
		return std::tuple<uint16_t, uint16_t, uint16_t>(std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max());
	else
		return std::tuple<uint16_t, uint16_t, uint16_t>(x + w_x - _padding_x, y + w_y - _padding_y, k + w_k - _padding_k);
}

bool Layer4D::is_valid_input_coord(const std::tuple<uint16_t, uint16_t, uint16_t> &coord) const
{
	return std::get<0>(coord) != std::numeric_limits<uint16_t>::max();
}

std::tuple<uint16_t, uint16_t, uint16_t> Layer4D::receptive_field_of(const std::tuple<uint16_t, uint16_t, uint16_t> &in) const
{
	return std::tuple<uint16_t, uint16_t, uint16_t>((std::get<0>(in) - 1) * _stride_x + _filter_width,
													(std::get<1>(in) - 1) * _stride_y + _filter_height,
													(std::get<2>(in) - 1) * _stride_k + _filter_conv_depth);
}
