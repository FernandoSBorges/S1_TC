

# adapted from: https://github.com/BlueBrain/neurodamus/blob/2c7052096c22fb5183fdef120d080608748da48d/neurodamus/core/stimuli.py#L271 
class CurrentStim():
    def add_ornstein_uhlenbeck(tau, sigma, mean, duration, dt=0.025,seed=100000, plotFig=False):
        from neuron import h
        import numpy as np
        
        """
        Adds an Ornstein-Uhlenbeck process with given correlation time,
        standard deviation and mean value.

        tau: correlation time [ms], white noise if zero
        sigma: standard deviation [uS]
        mean: mean value [uS]
        duration: duration of signal [ms]
        dt: timestep [ms]
        """
        from math import sqrt, exp

        """Creates a default RNG, currently based on ACG"""
        rng = h.Random(seed)

        # rng = RNG()  # Creates a default RNG
        # if not self._rng:
        #     logging.warning("Using a default RNG for Ornstein-Uhlenbeck process")

        # tvec = h.Vector()
        # tvec.indgen(self._cur_t, self._cur_t + duration, dt)  # time vector
        tvec = h.Vector(np.linspace(0, duration, int(duration/dt)))
        ntstep = len(tvec)  # total number of timesteps

        svec = h.Vector(ntstep, 0)  # stim vector

        noise = h.Vector(ntstep)  # Gaussian noise
        rng.normal(0.0, 1.0)
        noise.setrand(rng)  # generate Gaussian noise

        if tau < 1e-9:
            svec = noise.mul(sigma)  # white noise
        else:
            mu = exp(-dt / tau)  # auxiliar factor [unitless]
            A = sigma * sqrt(1 - mu * mu)  # amplitude [uS]
            noise.mul(A)  # scale noise by amplitude [uS]

            # Exact update formula (independent of dt) from Gillespie 1996
            for n in range(1, ntstep):
                svec.x[n] = svec[n - 1] * mu + noise[n]  # signal [uS]

        svec.add(mean)  # shift signal by mean value [uS]

        # self._add_point(self._base_amp)
        # self.time_vec.append(tvec)
        # self.stim_vec.append(svec)
        # self._cur_t += duration
        # self._add_point(self._base_amp)
        
        if plotFig:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(30,5))
            plt.plot(list(tvec),list(svec),'k')
            plt.savefig('test_fig_vec_OrnsteinUhlenbeck.png')
            
            plt.figure(figsize=(30,5))
            plt.plot(list(tvec)[0:1000],list(svec)[0:1000],'k')
            plt.savefig('test_fig_vec_OrnsteinUhlenbeck_slice.png')
            
            plt.figure()
            plt.hist(list(svec),100)
            plt.savefig('test_fig_vec_OrnsteinUhlenbeck_hist.png')

        return tvec,svec


if __name__ == "__main__":
    import math
    mean_percect = 0.0  # percent of threshold current (or holding current, not sure)
    threshold_current = 0.039407827
    holding_current = 0.0488804
    
    mean = (mean_percect/100)*threshold_current
    variance = 0.001

    tvec,svec = CurrentStim.add_ornstein_uhlenbeck(tau=1e-9,sigma=math.sqrt(variance),mean=mean,duration=10250.0,dt=0.025,seed=100000,plotFig=True)