#ifndef _STDP_BIOLOGICAL_H
#define _STDP_BIOLOGICAL_H

#include "Stdp.h"
/**
* @brief The spike timing-dependent plasticity learning rule, if the pre-synaptic neuron spikes first, the synaptic weight undergoes LTA,
* If the post synaptic spike fires first, the synaptic weight undergoes LTD.
* 
* @param alpha The weights learning rate, which is a multiplicative factor that helps converge to a stable state while training.
* @param tau The STDP time constant.
*/
namespace stdp {

	class Biological : public STDP {

	public:
		Biological();
		Biological(float ap, float am, Time tau);
		Biological(float alpha, Time tau);

		virtual float process(float w, const Time pre, Time post);
		virtual void adapt_parameters(float factor);
	private:
		float _ap;
		float _am;
		float _tau;
	};

}
#endif
