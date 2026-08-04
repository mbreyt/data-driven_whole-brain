import sys
import jax, optax
from scripts.python.config import PROJECT_DIR

# Importing this module requires the repository root on sys.path, e.g.
#   sys.path.insert(0, '/path/to/data-driven_whole-brain')
# The notebooks and Snakemake workflows do this before importing.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import scripts.python.noise_generator as noise_generator
from scripts.python.ml_models import make_delay_helper, TVB, MontBrio
import numpy as np
import jax.numpy as jnp
from scipy.ndimage import shift
from brian2 import *
import os
import scipy.stats as st

class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1))  # return to start of line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed - self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")
    


def run_fre(eta, J, X0, sim_len, seed_p=None, standalone=False, noise=None, sigma=.8):
    if seed_p != None:
        noise = noise_generator.spectral_exponent(sim_len*10000,  1.1, seed_p, sigma=sigma)
    if np.all(noise==0):
        noise = noise_generator.spectral_exponent(sim_len*10000, 1.1, 0, sigma=sigma)
    W, L = jnp.array([[0.]]), jnp.array([[0.]])
    n_nodes = 1
    dt = .001
    v_c = 20.
    dh = make_delay_helper(weights=W, lengths=L, dt=dt, v_c=v_c)
    tvb_p = {
        'dh': dh,
        'W': W,
    }

    dfun = MontBrio(coupled=True, scaling_factor=1, J=J)

    @jax.jit
    def mpr_r_positive(rv):
        r, v = rv[:,:1], rv[:,1:]
        return jnp.c_[ r*(r>0), v ]

    def gfun(x, sqrt_dt, nsig=0.0): # combo scaling_factor .01 dt 1 sig 0.0004 g .55 OK
        steps, nodes, stvar = x.shape
        sigs = jnp.tile(jnp.sqrt(2*jnp.array([nsig, 2*nsig])), (steps, nodes, 1))
        return x*sqrt_dt*sigs

    tvb = TVB(tvb_p, dfun=dfun, nst_vars=2, n_pars=0, gfun=gfun, dt=dt, adhoc=mpr_r_positive, stimulus=jnp.array(noise), chunksize=1000, initial_cond=jnp.array([0.,X0]).reshape(1,2))
    rv, b = tvb.apply({}, jnp.repeat(jnp.array([[eta, J]]), n_nodes, axis=0), .0, sim_len=sim_len, seed=1, initial_cond=True, mlp=False, stimulus_yn=True)
    if standalone:
        return np.c_[rv.squeeze(), np.tile(eta, len(rv)), np.tile(J, len(rv)), noise.squeeze()]
    return rv



def run_qif(J, eta, v0, sim_len, delta = 1, N = 10 ** 4, p=1.0, exponent=1.1, initial_v=None, seed_p=None, sigma=1, uniform=False, run_fre_yn=True):
    start_scope()
    vt = 100.0 * volt  # infinity -
    vr = -vt  # infinity +
    ## Neurons constant 
    c1 = 1 / volt  # constant of the voltage
    tau_m = 1.0 * second  # time constant of the membrane
    vt = 100.0 * volt  # infinity -
    vr = -vt  # infinity +
    # synaptic input
    R = 1 * ohm  # resistance of the input current
    taue = 10.0e-4 * second  # time constant of the synapse
    # noise_orig = noise_generator.spectral_exponent(N=int(sim_len*1e3), exponent=1.1)

    noise_orig = noise_generator.spectral_exponent(sim_len*10000, exponent, seed_p, sigma=sigma)
    noise = TimedArray(noise_orig * amp, dt= .1 * ms)

    neurons = NeuronGroup(N, '''dv/dt = (v*v * c1 + I * R)/tau_m  : volt (unless refractory)
                            I = J * ge / N  + n + I_stim: ampere 
                            I_stim = noise(t): ampere
                            n : ampere
                            ge : ampere
                            ''',
                            dt = .1 * ms,
                    threshold='v>vt', reset='''v = vr ''',
                    method='euler', refractory= 2 / vt * volt * second,
                    )

    neurons.n = ((np.tan(np.pi*(np.arange(N)/N)-.5))+eta) * amp
    # initial condition
    neurons.v = v0 * volt

    # Creation of the connectivity between neurons
    conn = Synapses(neurons, neurons, on_pre={
                                'up': 'ge_post += 1/taue * second * amp',
                                'down': 'ge_post -= 1/taue * second * amp',
                                },
                        delay={
                            'up': 10 * ms,
                            'down':  11 * ms,
                            })
    conn.connect(p=p)

    # Recording of the neurons
    m_r = PopulationRateMonitor(neurons)
    m_v = StateMonitor(neurons, 'v', record=True, dt=1.*ms)
    m_s = SpikeMonitor(neurons)

    stim = StateMonitor(neurons[:1], 'I_stim', record=True)
    spikes = SpikeMonitor(neurons)

    # run the simulation
    net = Network(neurons, conn, m_r, m_v, stim, m_s)
    vs = []
    vs_var = []
    for i in range(sim_len):
        run(sim_len//sim_len * second, report=ProgressBar(), report_period=1 * second)
        vv = m_v.get_states('v')
        vv = m_v.v[np.isfinite(m_v.v).all(axis=1)]/ volt
        vv[vv==-100] = 0
        vs_var.append(np.var(vv, axis=0))
        vs.append(np.mean(vv, axis=0))
        del vv
        del m_v
        m_v = StateMonitor(neurons, 'v', record=True, dt=1.*ms)


    time = np.array(m_r.t/second)
    # r = m_r.smooth_rate(window='flat', width=0.1 * second)/hertz
    width = 0.01
    dt = 1e-4
    width_dt = int(width / 2 / dt) * 2 + 1
    used_width = width_dt * dt
    window = np.ones(width_dt)
    rates = np.copy(np.array(m_r.rate))
    rr = shift(rates, 0, cval=0)
    r = np.convolve(rr, window * 1.0 / sum(window), mode="same")

    v_m = np.hstack(vs)
    v_var = np.hstack(vs_var)
    stim = stim.I_stim/amp

    if run_fre_yn:
        rv, b = run_fre(eta, v0, sim_len, noise = noise_orig[::10])
        rslt = {'t': time, 'r': r[::10], 'v': v_m, 'J': J, 'eta': eta, 'I': stim.squeeze(), 'p': p, 'fre': rv.reshape(-1,2)[::10]}
    else:
        rslt = {'t': time, 'r': r, 'v': v_m,'J': J, 'eta': eta, 'I': stim.squeeze(), 'p': p, 'v_var': v_var}
    return rslt#, vv, spikes.t/second, spikes.i


def run_qif_step(J, eta, v0, sim_len, delta = 1, N = 10 ** 4, p=1.0, exponent=1.1, initial_v=None, seed_p=None, sigma=1, uniform=False, run_fre_yn=True, stim=np.array([]), dt_i=1.):
    start_scope()
    vt = 100.0 * volt  # infinity -
    vr = -vt  # infinity +
    ## Neurons constant 
    c1 = 1 / volt  # constant of the voltage
    tau_m = 1.0 * second  # time constant of the membrane
    vt = 100.0 * volt  # infinity -
    vr = -vt  # infinity +
    # synaptic input
    R = 1 * ohm  # resistance of the input current
    taue = 10.0e-4 * second  # time constant of the synapse
    noise = TimedArray(stim * amp, dt= 1 * ms)
    
    neurons = NeuronGroup(N, '''dv/dt = (v*v * c1 + I * R)/tau_m  : volt (unless refractory)
                            I = J * ge / N  + n + I_stim: ampere 
                            I_stim = noise(t): ampere
                            n : ampere
                            ge : ampere
                            ''',
                            dt = dt_i * ms,
                    threshold='v>vt', reset='''v = vr ''',
                    method='euler', refractory= 2 / vt * volt * second,
                    )

    neurons.n = ((np.tan(np.pi*(np.arange(N)/N)-.5))+eta) * amp
    np.random.shuffle(neurons.n)

    # initial condition
    neurons.v = v0 * volt

    # Creation of the connectivity between the neurons
    conn = Synapses(neurons, neurons, on_pre={
                                'up': 'ge_post += 1/taue * second * amp',
                                'down': 'ge_post -= 1/taue * second * amp',
                                },
                        delay={
                            'up': 10 * ms,
                            'down':  11 * ms,
                            })
    conn.connect(p=p)


    # Recording of the neurons
    m_r = PopulationRateMonitor(neurons)
    m_v = StateMonitor(neurons, 'v', record=True, dt=1.*ms)
    m_s = SpikeMonitor(neurons)
    stim = StateMonitor(neurons[:1], 'I_stim', record=True)
    spikes = SpikeMonitor(neurons)
    
    # run the simulation
    net = Network(neurons, conn, m_r, m_v, stim, m_s)
    net.run(sim_len * second, report=ProgressBar(), report_period=1 * second)

    time = np.array(m_r.t/second)
    width = 0.01
    dt = 1e-4
    width_dt = int(width / 2 / dt) * 2 + 1
    used_width = width_dt * dt
    window = np.ones(width_dt)
    rates = np.copy(np.array(m_r.rate))
    rr = shift(rates, 0, cval=0)
    r = np.convolve(rr, window * 1.0 / sum(window), mode="same")

    v = m_v.v[np.isfinite(m_v.v).all(axis=1)]/ volt
    v[v==-100] = 0
    v_m = np.mean(v, axis=0)
    stim = stim.I_stim/amp

    if run_fre_yn:
        # noise_orig = np.r_[np.zeros(10000), np.(30000)*3, np.ones(20000)/10000]
        rv = run_fre(eta, J, v0, sim_len, noise = stim.squeeze()[::10])
        rslt = {'t': time[::10], 'r': r[::10], 'v': v_m, 'J': J, 'eta': eta, 'I': stim.squeeze(), 'p': p, 'fre': rv.reshape(-1,2)[::10]}
    else:
        rslt = {'t': time[::10], 'r': r[::10], 'v': v_m, 'J': J, 'eta': eta, 'I': stim.squeeze(), 'p': p, 'spikes':m_v.v} 
    return rslt#, vv, spikes.t/second, spikes.i

