#ifndef _PROCESS_NORM_H
#define _PROCESS_NORM_H

#include "Tensor.h"
#include "Process.h"

namespace process {

	class Normalizing : public UniquePassProcess {

	public:
		Normalizing();
		Normalizing(float max);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		float _max;
	};
}

#endif
