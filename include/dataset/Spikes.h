#ifndef _DATASET_SPIKES_H
#define _DATASET_SPIKES_H

#include <string>
#include <cassert>
#include <fstream>
#include <limits>

#include "Tensor.h"
#include "Input.h"


namespace dataset {

	// Generic data loader for spikes 
	//
	class Spikes : public Input {

	public:
		Spikes(const std::string& image_filename, const std::string& label_filename, unsigned int width, unsigned int height, unsigned int depth, const std::string& dataset_name);

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape& shape() const;

	private:
		void _prepare_next();

		std::string _spikes_filename;
		std::string _label_filename;

		std::ifstream _spikes_file;
		std::ifstream _label_file;


		Shape _shape;
		unsigned long _size;
		unsigned long _idx,_idx_local;

		std::vector<float> _data;
		std::vector<int> _label;
	
		unsigned int _width, _height, _depth;	

		std::string _name;
	};

}

#endif
