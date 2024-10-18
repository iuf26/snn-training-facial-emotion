#ifndef _OUTPUT_CONVERTER_H
#define _OUTPUT_CONVERTER_H

#include "ClassParameter.h"
#include "Spike.h"
#include "Process.h"
#include "tool/Operations.h"

class OutputConverter : public AbstractProcess {

public:
	template<typename T, typename Factory>
	OutputConverter(const RegisterClassParameter<T, Factory>& registration) : AbstractProcess(registration) {

	}

	virtual Shape compute_shape(const Shape& shape);
	virtual size_t train_pass_number() const;
	virtual void process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number);
	virtual void process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number);

	virtual Tensor<float> process(const Tensor<float>& in) = 0;
};

class OutputConverterFactory : public ClassParameterFactory<OutputConverter, OutputConverterFactory> {

public:
	OutputConverterFactory() : ClassParameterFactory<OutputConverter, OutputConverterFactory>("OutputConverter") {

	}

};

class NoOutputConversion : public OutputConverter {

public:
	NoOutputConversion();

	virtual Tensor<float> process(const Tensor<float>& in);
};

class DefaultOutput : public OutputConverter {

public:
	DefaultOutput();
	DefaultOutput(Time min, Time max);

	virtual Tensor<float> process(const Tensor<float>& in);

private:
	Time _min;
	Time _max;

};

/**
 * @brief This function is used fçor spike to feature conversion, to change the values back from "Time" to "float".
 * Because the SVM doesn't work with spikes. So after the features are extracted, 
 * they need to be changes back from spike timestamps into regular values before classification.
 * 
 * @param Conv The convolutional layer
 * @param t_obj The objective time at which neurons are expected to fire (used in this equation as T_output_min).
 * @param exp_name The name of the experimant.
 * @param layer_name The name of the layer.
 * @param save_timestamps A flag that allows saving the data as timestamps.
 */
class TimeObjectiveOutput : public OutputConverter {

public:
    TimeObjectiveOutput();
	TimeObjectiveOutput(Time t_obj);
	TimeObjectiveOutput(Time t_obj, std::string exp_name, std::string layer_name, size_t save_timestamps = 0);

	virtual Tensor<float> process(const Tensor<float>& in);

private:
	Time _t_obj;
	// to save the data
	size_t _save_timestamps;
	std::string _exp_name;
	std::string _layer_name;
	std::string _file_path;

};

class SoftMaxOutput : public OutputConverter {

public:
	SoftMaxOutput();

	virtual Tensor<float> process(const Tensor<float>& in);
};

class WTAOutput : public OutputConverter {

public:
	WTAOutput();

	virtual Tensor<float> process(const Tensor<float>& in);
};

class SpikeTiming : public OutputConverter {

public:
	SpikeTiming();

	virtual Tensor<float> process(const Tensor<float>& in);
};
#endif
