#include "dataset/Spikes.h"
#include "dep/npy.hpp"

using namespace dataset;


Spikes::Spikes(const std::string& spikes_filename, const std::string& label_filename, unsigned int width, unsigned int height, unsigned int depth, const std::string& dataset_name) :
	_spikes_filename(spikes_filename), _label_filename(label_filename),
	_spikes_file(spikes_filename, std::ios::in | std::ios::binary), _label_file(label_filename, std::ios::in | std::ios::binary),
	_name(dataset_name), 
	_width(width), _height(height), _depth(depth), 
	_idx(0), _idx_local(0), _shape(), _size(0) {

	if(!_spikes_file.is_open()) {
		throw std::runtime_error("Can't open "+spikes_filename);
	}
	if(!_label_file.is_open()) {
		throw std::runtime_error("Can't open "+label_filename);
	}

	std::cerr<<"START LOADING SPIKING DATA\n";
	std::vector<unsigned long> in_shape,label_shape;
	bool fortran_order{false};

	npy::LoadArrayFromNumpy(_spikes_filename, in_shape, fortran_order, _data);
	npy::LoadArrayFromNumpy(_label_filename, label_shape, fortran_order,  _label);
	std::vector<long unsigned int> shape(3);
	shape[0]=_width;shape[1]=_height;shape[2]=_depth;
	_shape = Shape(shape);
	_size = _data.size()/_shape.product();

	std::cerr<<"#record "<<_size<<" vs total size "<<_data.size()<<"\n";
	_idx = 0;
	_idx_local = 0;

	_prepare_next();
}

bool Spikes::has_next() const {
	return _idx < _size;
}


std::pair<std::string, Tensor<InputType>> Spikes::next() {
	std::pair<std::string, Tensor<InputType>> out(std::to_string(_label[_idx]), _shape);

	for(size_t x=0; x<_shape.dim(0); x++) {
		for(size_t y=0; y<_shape.dim(1); y++) {
			for(size_t z=0; z<_shape.dim(2); z++) {
					out.second.at(x, y, z) = _data[_idx_local++];
			}
		}
	}
	
	_idx++;

	_prepare_next();

	return out;
}

void Spikes::reset() {
	_idx = 0;
	_idx_local = 0;
	_prepare_next();
}


void Spikes::close() {
	_data.clear();
	_label.clear();
}

size_t Spikes::size() const {
	return _size;
}

std::string Spikes::to_string() const {
	return _name+"("+_spikes_filename+", "+_label_filename+")";
}

const Shape& Spikes::shape() const {
	return _shape;
}

void Spikes::_prepare_next() {
}
