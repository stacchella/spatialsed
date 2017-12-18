from copy import deepcopy
import numpy as np
import os
import itertools
from prospect.models import priors
from prospect.sources import MultiComponentCSPBasis
from prospect.models.sedmodel import SedModel
from sedpy.observate import load_filters

tophat = None

# --------------
# RUN_PARAMS
# --------------
run_params = {'verbose':True,
              'debug':False,
              'outfile':'spatial_demo',
              # dynesty params
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'rwalk', # sampling method
              'nested_walks': 30, # MC walks
              'nested_nlive_batch': 200, # size of live point "batches"
              'nested_nlive_init': 200, # number of initial live points
              'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
              'nested_dlogz_init': 0.01,
              'nested_stop_kwargs': {'post_thresh': 0.02, 'n_mc': 50},  # higher threshold, more MCMC
              # Mock data parameters
              'snr': 20.0,
              'add_noise': False,
              # Input mock model parameters
              'mass': np.array([4e10,1e10]),
              'logzsol': np.array([0.0,-0.5]),
              'tage': np.array([12.,4.]),
              'tau': np.array([1.,10]),
              'dust2': np.array([0.2,0.6]),
              'zred': 2.,
              # Data manipulation parameters
              'logify_spectrum':False,
              'normalize_spectrum':False,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# OBS
# --------------
def load_obs(snr=10.0, add_noise=True, **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.

    :param snr:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters.

    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock spectrum
    """

    # first, load the filters. let's use the GOODSN filter set.
    # XX: rewrite as necessary to find the filters folder
    filter_folder = os.getenv('APPS')+'/spatialsed/filters/'
    fname_all = os.listdir(filter_folder)
    fname_goodsn = [f.split('.')[0] for f in fname_all if 'goodsn' in f]

    # now separate into components. we will generate separate observations for
    # HST bands.
    fname_hst = ['f435w','f606w','f775w','f850lp','f125w','f140w','f160w']
    n_hst = len(fname_hst)
    n_blended = len(fname_goodsn) - n_hst
    component = np.array(np.zeros(n_hst).tolist() + np.ones(n_hst).tolist() + np.repeat(-1,n_blended).tolist(),dtype=int)

    # generate filter list. repeat HST filters
    fname_ground = [s for s in fname_goodsn if s.split('_')[0] not in fname_hst]
    fnames = 2*[f+'_goodsn' for f in fname_hst] + fname_ground
    filters = load_filters(fnames, directory=filter_folder)

    # now generate data
    # we will need the models to make a mock
    sps = load_sps(**kwargs)
    mod = load_model(**kwargs)

    # we will also need an obs dictionary
    obs = {}
    obs['filters'] = filters
    obs['component'] = component
    obs['wavelength'] = None

    # Now we get the mock params from the kwargs dict
    params = {}
    for p in mod.params.keys():
        if p in kwargs:
            params[p] = np.atleast_1d(kwargs[p])

    # Generate the photometry, add noise
    mod.params.update(params)
    spec, phot, _ = mod.mean_model(mod.theta, obs, sps=sps)
    pnoise_sigma = phot / snr
    if add_noise:
        pnoise = np.random.normal(0, 1, len(phot)) * pnoise_sigma
        maggies = phot + pnoise
    else:
        maggies = phot.copy()

    # Now store output in standard format
    obs['maggies'] = maggies
    obs['maggies_unc'] = pnoise_sigma
    obs['mock_snr'] = snr
    obs['phot_mask'] = np.ones(len(phot), dtype=bool)

    # we also keep the unessential mock information
    obs['true_spectrum'] = spec.copy()
    obs['true_maggies'] = phot.copy()
    obs['mock_params'] = deepcopy(mod.params)

    return obs

    """ plotting code
    wave_eff = np.log10([filt.wave_effective for filt in obs['filters']])
    flux = np.log10(obs['maggies'])
    fluxerr = np.zeros_like(flux)+0.05
    bulge = obs['component'] == 0
    disk = obs['component'] == 1
    total = obs['component'] == -1

    plt.errorbar(wave_eff,flux,yerr=fluxerr,color='black',linestyle=' ',label='total',fmt='o')
    plt.errorbar(wave_eff[bulge], flux[bulge], yerr=fluxerr[bulge], color='red', linestyle=' ', fmt='o', label='bulge')
    plt.errorbar(wave_eff[disk], flux[disk], yerr=fluxerr[disk], color='blue', linestyle=' ', fmt='o', label='disk')
    plt.legend()
    plt.show()

    """

  
# --------------
# SPS Object
# --------------


def load_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = MultiComponentCSPBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def load_gp(**extras):
    return None, None

# --------------
# MODEL_PARAMS
# --------------

# You'll note below that we have 5 free parameters:
# mass, logzsol, tage, tau, dust2
# They are all scalars.
# mass and tau have logUniform priors (i.e. TopHat priors in log(mass) and
# log(tau)), the rest have TopHat priors.
# You should adjust the prior ranges (particularly in mass) to suit your objects.
#
# The other parameters are all fixed, but we may want to explicitly set their
# values, which can be done here, to override any defaults in python-FSPS


model_params = []

# --- Distance ---
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.1,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

# --- SFH --------
# FSPS parameter
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 4,  # This is delay-tau
                        'units': 'type',
                        'prior': None})

model_params.append({'name': 'mass', 'N': 2,
                        'isfree': True,
                        'init': 1e10,
                        'init_disp': 1e9,
                        'units': r'M_\odot',
                        'prior': priors.LogUniform(mini=1e8, maxi=1e12)})

model_params.append({'name': 'logzsol', 'N': 2,
                        'isfree': True,
                        'init': -0.3,
                        'init_disp': 0.3,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': priors.TopHat(mini=-1.5, maxi=0.19)})

# If zcontinuous > 1, use 3-pt smoothing
model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'prior': None})

# FSPS parameter
model_params.append({'name': 'tau', 'N': 2,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.5,
                        'units': 'Gyr',
                        'prior': priors.LogUniform(mini=0.101, maxi=100)})

# FSPS parameter
model_params.append({'name': 'tage', 'N': 2,
                        'isfree': True,
                        'init': 5.0,
                        'init_disp': 3.0,
                        'units': 'Gyr',
                        'prior': priors.TopHat(mini=0.01, maxi=14.0)})


# --- Dust ---------
# FSPS parameter
model_params.append({'name': 'dust2', 'N': 2,
                        'isfree': True,
                        'init': 0.35,
                        'reinit': True,
                        'init_disp': 0.3,
                        'units': 'Diffuse dust optical depth towards all stars at 5500AA',
                        'prior': priors.TopHat(mini=0.0, maxi=2.0)})

# FSPS parameter
model_params.append({'name': 'dust_index', 'N': 2,
                        'isfree': True,
                        'init': np.array([0.0, 0.0]),
                        'units': 'power law slope of the attenuation curve for diffuse dust',
                        'prior': priors.TopHat(mini=-0.4, maxi=0.4)})

# FSPS parameter
model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': 'power law slope of the attenuation curve for young-star dust',
                        'prior': None,})

# FSPS parameter
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 4,  # power-laws
                        'units': 'index',
                        'prior': None})

# FSPS parameter
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index',
                        'prior': None})

# An example of the parameters controlling the dust emission SED.  There are others!
model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'units': 'MMP83 local MW intensity',
                        'prior': None})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': False,
                        'init': 2.0,
                        'units': 'MMP83 local MW intensity',
                        'prior': None})


###### Nebular Emission ###########
# Here is a really simple function that takes a **dict argument, picks out the
# `logzsol` key, and returns the value.  This way, we can have gas_logz find
# the value of logzsol and use it, if we uncomment the 'depends_on' line in the
# `gas_logz` parameter definition.
#
# One can use this kind of thing to transform parameters as well (like making
# them linear instead of log, or divide everything by 10, or whatever.) You can
# have one parameter depend on several others (or vice versa).  Just remember
# that a parameter with `depends_on` must always be fixed.

def stellar_logzsol(logzsol=0.0, **extras):
    return logzsol

model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior': None})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior': None})

model_params.append({'name': 'nebemlineinspec', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'prior': None})

model_params.append({'name': 'gas_logz', 'N': 2,
                        'isfree': False,
                        'init': np.array([0.0, 0.0]),
                        'depends_on': stellar_logzsol,
                        'units': r'log Z/Z_\odot',
                        'prior': priors.TopHat(mini=-2.0, maxi=0.5)})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-4.0, maxi=-1.0)})


# --- Calibration ---------
# Only important if using a NoiseModel
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': 'mags',
                        'prior': priors.TopHat(mini=0.0, maxi=0.2)})


def load_model(zred=None, **extras):
    # In principle (and we've done it) you could have the model depend on
    # command line arguments (or anything in run_params) by making changes to
    # `model_params` here before instantiation the SedModel object.  Up to you.

    # Here we are going to set the intial value (and the only value, since it
    # is not a free parameter) of the redshift parameter to whatever was used
    # to generate the mock, listed in the run_params dictionary.
    pn = [p['name'] for p in model_params]
    zind = pn.index('zred')
    model_params[zind]['init'] = zred
    
    return SedModel(model_params)

