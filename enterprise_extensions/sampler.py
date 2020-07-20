# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function)
import numpy as np
import os
from enterprise import constants as const
import pickle
import healpy as hp
from scipy.stats import skewnorm, truncnorm


from enterprise import constants as const
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

class JumpProposal(object):

    def __init__(self, pta, snames=None, empirical_distr=None, f_stat_file=None):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.ndim = sum(p.size or 1 for p in pta.params)
        self.plist = [p.name for p in pta.params]

        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size

        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct

        # collecting signal parameters across pta
        if snames is None:
            allsigs = np.hstack([[qq.signal_name for qq in pp._signals]
                                                 for pp in pta._signalcollections])
            self.snames = dict.fromkeys(np.unique(allsigs))
            for key in self.snames: self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames: self.snames[key] = list(set(self.snames[key]))
        else:
            self.snames = snames

        # empirical distributions
        if empirical_distr is not None and os.path.isfile(empirical_distr):
            try:
                with open(empirical_distr, 'rb') as f:
                    pickled_distr = pickle.load(f)
            except:
                try:
                    with open(empirical_distr, 'rb') as f:
                        pickled_distr = pickle.load(f)
                except:
                    print('I can\'t open the empirical distribution pickle file!')
                    pickled_distr = None

            self.empirical_distr = pickled_distr

        elif isinstance(empirical_distr,list):
            pass
        else:
            self.empirical_distr = None

        if self.empirical_distr is not None:
            # only save the empirical distributions for parameters that are in the model
            mask = []
            for idx,d in enumerate(self.empirical_distr):
                if d.ndim == 1:
                    if d.param_name in pta.param_names:
                        mask.append(idx)
                else:
                    if d.param_names[0] in pta.param_names and d.param_names[1] in pta.param_names:
                        mask.append(idx)
            if len(mask) > 1:
                self.empirical_distr = [self.empirical_distr[m] for m in mask]
            else:
                self.empirical_distr = None

        #F-statistic map
        if f_stat_file is not None and os.path.isfile(f_stat_file):
            npzfile = np.load(f_stat_file)
            self.fe_freqs = npzfile['freqs']
            self.fe = npzfile['fe']

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param = np.random.choice(self.params)

        # if vector parameter jump in random component
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'red noise'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_empirical_distr(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        if self.empirical_distr is not None:

            # randomly choose one of the empirical distributions
            distr_idx = np.random.randint(0, len(self.empirical_distr))

            if self.empirical_distr[distr_idx].ndim == 1:

                idx = self.pnames.index(self.empirical_distr[distr_idx].param_name)
                q[idx] = self.empirical_distr[distr_idx].draw()

                lqxy = (self.empirical_distr[distr_idx].logprob(x[idx]) -
                        self.empirical_distr[distr_idx].logprob(q[idx]))

            else:

                oldsample = [x[self.pnames.index(p)]
                             for p in self.empirical_distr[distr_idx].param_names]
                newsample = self.empirical_distr[distr_idx].draw()

                for p,n in zip(self.empirical_distr[distr_idx].param_names, newsample):
                    q[self.pnames.index(p)] = n

                lqxy = (self.empirical_distr[distr_idx].logprob(oldsample) -
                        self.empirical_distr[distr_idx].logprob(newsample))

        return q, float(lqxy)

    def draw_from_dm_gp_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'dm_gp'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_dm1yr_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        dm1yr_names = [dmname for dmname in self.pnames if 'dm_s1yr' in dmname]
        dmname = np.random.choice(dm1yr_names)
        idx = self.pnames.index(dmname)
        if 'log10_Amp' in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif 'phase' in dmname:
            q[idx] = np.random.uniform(0, 2*np.pi)

        return q, 0

    def draw_from_dmexpdip_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        dmexp_names = [dmname for dmname in self.pnames if 'dmexp' in dmname]
        dmname = np.random.choice(dmexp_names)
        idx = self.pnames.index(dmname)
        if 'log10_Amp' in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif 'log10_tau' in dmname:
            q[idx] = np.random.uniform(0, 2.5)
        elif 'sign_param' in dmname:
            q[idx] = np.random.uniform(-1.0, 1.0)

        return q, 0

    def draw_from_dmexpcusp_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        dmexp_names = [dmname for dmname in self.pnames if 'dm_cusp' in dmname]
        dmname = np.random.choice(dmexp_names)
        idx = self.pnames.index(dmname)
        if 'log10_Amp' in dmname:
            q[idx] = np.random.uniform(-10, -2)
        elif 'log10_tau' in dmname:
            q[idx] = np.random.uniform(0, 2.5)
        #elif 't0' in dmname:
        #    q[idx] = np.random.uniform(53393.0, 57388.0)
        elif 'sign_param' in dmname:
            q[idx] = np.random.uniform(-1.0, 1.0)

        return q, 0

    def draw_from_dmx_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'dmx_signal'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_gwb_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('gw_log10_A')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_dipole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('dipole_log10_A')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_monopole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('monopole_log10_A')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_altpol_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        polnames = [pol for pol in self.pnames if 'log10Apol' in pol]
        if 'kappa' in self.pnames:
            polnames.append('kappa')
        pol = np.random.choice(polnames)
        idx = self.pnames.index(pol)
        if pol == 'log10Apol_tt':
            q[idx] = np.random.uniform(-18, -12)
        elif pol == 'log10Apol_st':
            q[idx] = np.random.uniform(-18, -12)
        elif pol == 'log10Apol_vl':
            q[idx] = np.random.uniform(-18, -15)
        elif pol == 'log10Apol_sl':
            q[idx] = np.random.uniform(-18, -16)
        elif pol == 'kappa':
            q[idx] = np.random.uniform(0, 10)

        return q, 0

    def draw_from_ephem_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'phys_ephem'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_bwm_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'bwm'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_cw_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'cw'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_cw_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_h')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_dm_sw_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'gp_sw'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_signal_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0
        std = ['linear timing model',
               'red noise',
               'phys_ephem',
               'gw',
               'cw',
               'bwm',
               'gp_sw',
               'ecorr_sherman-morrison',
               'ecorr',
               'efac',
               'equad',
               ]
        non_std = [nm for nm in self.snames.keys() if nm not in std]
        # draw parameter from signal model
        signal_name = np.random.choice(non_std)
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_par_prior(self, par_names):
        # Preparing and comparing par_names with PTA parameters
        par_names = np.atleast_1d(par_names)
        par_list = []
        name_list = []
        for par_name in par_names:
            pn_list = [n for n in self.plist if par_name in n]
            if pn_list:
                par_list.append(pn_list)
                name_list.append(par_name)
        if not par_list:
            raise UserWarning("No parameter prior match found between {} and PTA.object."
                              .format(par_names))
        par_list = np.concatenate(par_list,axis=None)

        def draw(x, iter, beta):
            """Prior draw function generator for custom par_names.
            par_names: list of strings

            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # randomly choose parameter
            idx_name = np.random.choice(par_list)
            idx = self.plist.index(idx_name)

            # if vector parameter jump in random component
            param = self.params[idx]
            if param.size:
                idx2 = np.random.randint(0, param.size)
                q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            # scalar parameter
            else:
                q[self.pmap[str(param)]] = param.sample()

            # forward-backward jump probability
            lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                    param.get_logpdf(q[self.pmap[str(param)]]))

            return q, float(lqxy)

        name_string = '_'.join(name_list)
        draw.__name__ = 'draw_from_{}_prior'.format(name_string)
        return draw
    
    
class JumpProposalCW(object):

    def __init__(self, pta, fgw=8e-9,psr_dist = None, snames=None, empirical_distr=None, f_stat_file=None):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.ndim = sum(p.size or 1 for p in pta.params)
        self.plist = [p.name for p in pta.params]

        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size

        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct

        # collecting signal parameters across pta
        if snames is None:
            allsigs = np.hstack([[qq.signal_name for qq in pp._signals]
                                                 for pp in pta._signalcollections])
            self.snames = dict.fromkeys(np.unique(allsigs))
            for key in self.snames: self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames: self.snames[key] = list(set(self.snames[key]))
        else:
            self.snames = snames
            
        self.fgw = fgw
        self.psr_dist = psr_dist

        # empirical distributions
        if empirical_distr is not None and os.path.isfile(empirical_distr):
            try:
                with open(empirical_distr, 'rb') as f:
                    pickled_distr = pickle.load(f)
            except:
                try:
                    with open(empirical_distr, 'rb') as f:
                        pickled_distr = pickle.load(f)
                except:
                    print('I can\'t open the empirical distribution pickle file!')
                    pickled_distr = None

            self.empirical_distr = pickled_distr

        elif isinstance(empirical_distr,list):
            pass
        else:
            self.empirical_distr = None

        if self.empirical_distr is not None:
            # only save the empirical distributions for parameters that are in the model
            mask = []
            for idx,d in enumerate(self.empirical_distr):
                if d.ndim == 1:
                    if d.param_name in pta.param_names:
                        mask.append(idx)
                else:
                    if d.param_names[0] in pta.param_names and d.param_names[1] in pta.param_names:
                        mask.append(idx)
            if len(mask) > 1:
                self.empirical_distr = [self.empirical_distr[m] for m in mask]
            else:
                self.empirical_distr = None

        #F-statistic map
        if f_stat_file is not None and os.path.isfile(f_stat_file):
            npzfile = np.load(f_stat_file)
            self.fe_freqs = npzfile['freqs']
            self.fe = npzfile['fe']

    def draw_from_par_log_uniform(self, par_dict):
        # Preparing and comparing par_dict.keys() with PTA parameters
        par_list = []
        name_list = []
        for par_name in par_dict.keys():
            pn_list = [n for n in self.plist if par_name in n and 'log' in n]
            if pn_list:
                par_list.append(pn_list)
                name_list.append(par_name)
        if not par_list:
            raise UserWarning("No parameter dictionary match found between {} and PTA.object."
                              .format(par_dict.keys()))
        par_list = np.concatenate(par_list,axis=None)

        def draw(x, iter, beta):
            """log uniform prior draw function generator for custom par_names.
            par_dict: dictionary with {"par_names":(lower bound,upper bound)}
                                      { "string":(float,float)}

            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # draw parameter from signal model
            idx_name = np.random.choice(par_list)
            idx = self.plist.index(idx_name)
            q[idx] = np.random.uniform(par_dict[par_name][0],par_dict[par_name][1])

            return q, 0

        name_string = '_'.join(name_list)
        draw.__name__ = 'draw_from_{}_log_uniform'.format(name_string)
        return draw

    def draw_from_signal(self, signal_names):
        # Preparing and comparing signal_names with PTA signals
        signal_names = np.atleast_1d(signal_names)
        signal_list = []
        name_list = []
        for signal_name in signal_names:
            try:
                param_list = self.snames[signal_name]
                signal_list.append(param_list)
                name_list.append(signal_name)
            except:
                pass
        if not signal_list:
            raise UserWarning("No signal match found between {} and PTA.object!"
                              .format(signal_names))
        signal_list = np.concatenate(signal_list,axis=None)

        def draw(x, iter, beta):
            """Signal draw function generator for custom signal_names.
            signal_names: list of strings

            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # draw parameter from signal model
            param = np.random.choice(signal_list)
            if param.size:
                idx2 = np.random.randint(0, param.size)
                q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            # scalar parameter
            else:
                q[self.pmap[str(param)]] = param.sample()

            # forward-backward jump probability
            lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                    param.get_logpdf(q[self.pmap[str(param)]]))

            return q, float(lqxy)

        name_string = '_'.join(name_list)
        draw.__name__ = 'draw_from_{}_signal'.format(name_string)
        return draw

    def fe_jump(self, x, iter, beta):

        q = x.copy()
        lqxy = 0
        
        fe_limit = np.max(self.fe)
        
        #draw skylocation and frequency from f-stat map
        accepted = False
        while accepted==False:
            log_f_new = self.params[self.pimap['log10_fgw']].sample()
            f_idx = (np.abs(np.log10(self.fe_freqs) - log_f_new)).argmin()

            gw_theta = np.arccos(self.params[self.pimap['cos_gwtheta']].sample())
            gw_phi = self.params[self.pimap['gwphi']].sample()
            hp_idx = hp.ang2pix(hp.get_nside(self.fe), gw_theta, gw_phi)

            fe_new_point = self.fe[f_idx, hp_idx]
            if np.random.uniform()<(fe_new_point/fe_limit):
                accepted = True

        #draw other parameters from prior
        cos_inc = self.params[self.pimap['cos_inc']].sample()
        psi = self.params[self.pimap['psi']].sample()
        phase0 = self.params[self.pimap['phase0']].sample()
        log10_h = self.params[self.pimap['log10_h']].sample()
        

        #put new parameters into q
        signal_name = 'cw'
        for param_name, new_param in zip(['log10_fgw','gwphi','cos_gwtheta','cos_inc','psi','phase0','log10_h'],
                                           [log_f_new, gw_phi, np.cos(gw_theta), cos_inc, psi, phase0, log10_h]):
            q[self.pimap[param_name]] = new_param
        
        #calculate Hastings ratio
        log_f_old = x[self.pimap['log10_fgw']]
        f_idx_old = (np.abs(np.log10(self.fe_freqs) - log_f_old)).argmin()
        
        gw_theta_old = np.arccos(x[self.pimap['cos_gwtheta']])
        gw_phi_old = x[self.pimap['gwphi']]
        hp_idx_old = hp.ang2pix(hp.get_nside(self.fe), gw_theta_old, gw_phi_old)
        
        fe_old_point = self.fe[f_idx_old, hp_idx_old]
        if fe_old_point>fe_limit:
            fe_old_point = fe_limit
            
        log10_h_old = x[self.pimap['log10_h']]
        phase0_old = x[self.pimap['phase0']]
        psi_old = x[self.pimap['psi']]
        cos_inc_old = x[self.pimap['cos_inc']]
        
        hastings_extra_factor = self.params[self.pimap['log10_h']].get_pdf(log10_h_old)
        hastings_extra_factor *= 1/self.params[self.pimap['log10_h']].get_pdf(log10_h)
        hastings_extra_factor = self.params[self.pimap['phase0']].get_pdf(phase0_old)
        hastings_extra_factor *= 1/self.params[self.pimap['phase0']].get_pdf(phase0)
        hastings_extra_factor = self.params[self.pimap['psi']].get_pdf(psi_old)
        hastings_extra_factor *= 1/self.params[self.pimap['psi']].get_pdf(psi)
        hastings_extra_factor = self.params[self.pimap['cos_inc']].get_pdf(cos_inc_old)
        hastings_extra_factor *= 1/self.params[self.pimap['cos_inc']].get_pdf(cos_inc)        
        
        lqxy = np.log(fe_old_point/fe_new_point * hastings_extra_factor)

        return q, float(lqxy)
    
    
    
    
    
    def draw_from_many_par_prior(self, par_names, string_name):
        # Preparing and comparing par_names with PTA parameters
        par_names = np.atleast_1d(par_names)
        par_list = []
        name_list = []
        for par_name in par_names:
            pn_list = [n for n in self.plist if par_name in n]
            if pn_list:
                par_list.append(pn_list)
                name_list.append(par_name)
        if not par_list:
            raise UserWarning("No parameter prior match found between {} and PTA.object."
                              .format(par_names))
        par_list = np.concatenate(par_list,axis=None)

        def draw(x, iter, beta):
            """Prior draw function generator for custom par_names.
            par_names: list of strings
            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # randomly choose parameter
            idx_name = np.random.choice(par_list)
            idx = self.plist.index(idx_name)

            # if vector parameter jump in random component
            param = self.params[idx]
            if param.size:
                idx2 = np.random.randint(0, param.size)
                q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            # scalar parameter
            else:
                q[self.pmap[str(param)]] = param.sample()

            # forward-backward jump probability
            lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                    param.get_logpdf(q[self.pmap[str(param)]]))

            return q, float(lqxy)

        name_string = string_name
        draw.__name__ = 'draw_from_{}_prior'.format(name_string)
        return draw
    
    def phase_psi_reverse_jump(self, x, iter, beta):
        ##written by SJV for 11yr CW
        q = x.copy()
        lqxy = 0

        param = np.random.choice([str(p) for p in self.pnames if 'phase' in p])
        
        if param == 'phase0':
            q[self.pnames.index('phase0')] = np.mod(x[self.pnames.index('phase0')] + np.pi, 2*np.pi)
            q[self.pnames.index('psi')] = np.mod(x[self.pnames.index('psi')] + np.pi/2, np.pi)
        else:
            q[self.pnames.index(param)] = np.mod(x[self.pnames.index(param)] + np.pi, 2*np.pi)
                
        return q, float(lqxy)
    
    def fix_cyclic_pars(self, prepar, postpar, iter, beta):
        ##written by SJV for 11yr CW
        q = postpar.copy()
        
        for param in self.params:
            if 'phase' in param.name:
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], 2*np.pi)
            elif param.name == 'psi':
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], np.pi)
            elif param.name == 'gwphi':
                #if param._pmin == 0 and param._pmax == 2*np.pi:
                if str(param).split('=')[1].split(',')[0] == 0 and str(param).split('=')[-1].split(')')[0] == str(2*np.pi):
                    q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], 2*np.pi)
                
        return q, 0

    def fix_psr_dist(self, prepar, postpar, iter, beta):
        ##written by SJV for 11yr CW
        q = postpar.copy()
        
        for param in self.params:
            if 'p_dist' in param.name:
                
                psr_name = param.name.split('_')[0]
                
                while self.psr_dist[psr_name][0] + self.psr_dist[psr_name][1]*q[self.pmap[str(param)]] < 0:
                    q[self.pmap[str(param)]] = param.sample()
                
        return q, 0
    
    def draw_strain_psi(self, x, iter, beta):
        #written by SJV for 11yr CW, adapted for targeted search by CAW
        
        q = x.copy()
        lqxy = 0
        
        # draw a new value of psi, then jump in log10_h so that either h*cos(2*psi) or h*sin(2*psi) are conserved
        which_jump = np.random.random()
        
        if 'log10_h' in self.pnames:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] + np.log10(ratio)       
                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] + np.log10(ratio)
        elif 'log10_fgw' in self.pnames:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])

                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                
            # draw one and calculate the other!!!
            cw_params = [ p for p in self.pnames if p in ['log10_mc', 'log10_fgw']]
            myparam = np.random.choice(cw_params)
            
            idx = 0
            for i,p in enumerate(self.params):
                 if p.name == myparam:
                    idx = i
            param = self.params[idx]
            
            if myparam == 'log10_mc':
                q[self.pnames.index('log10_mc')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_fgw')] = 3/2*(-5/3*q[self.pnames.index('log10_mc')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + np.log10(ratio))

            else:
                q[self.pnames.index('log10_fgw')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_mc')] = 3/5*(-2/3*q[self.pnames.index('log10_fgw')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + np.log10(ratio))
        else:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] + 3/5*np.log10(ratio)       
                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] + 3/5*np.log10(ratio)
                
        return q, float(lqxy)
    
    def draw_strain_inc(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # half of the time, jump so that you conserve h*(1 + cos_inc^2)
        # the rest of the time, jump so that you conserve h*cos_inc
        
        which_jump = np.random.random()
        

        

        if 'log10_h' in self.pnames:
            #written by SJV for 11yr CW, adapted for targeted search (strain not sampled) by CAW

            
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] \
                                                    + np.log10(1+x[self.pnames.index('cos_inc')]**2) \
                                                    - np.log10(1+q[self.pnames.index('cos_inc')]**2)
                        
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] \
                                                    + np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])
        elif 'log10_fgw' in self.pnames:
            
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                ratio =  np.log10(1+x[self.pnames.index('cos_inc')]**2) - np.log10(1+q[self.pnames.index('cos_inc')]**2)

                                               
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                ratio = np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])

            cw_params = [ p for p in self.pnames if p in ['log10_mc', 'log10_fgw']]
            myparam = np.random.choice(cw_params)
            
            idx = 0
            for i,p in enumerate(self.params):
                 if p.name == myparam:
                    idx = i
            param = self.params[idx]
            
            if myparam == 'log10_mc':
                q[self.pnames.index('log10_mc')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_fgw')] = 3/2*(-5/3*q[self.pnames.index('log10_mc')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + ratio)

            else:
                q[self.pnames.index('log10_fgw')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_mc')] = 3/5*(-2/3*q[self.pnames.index('log10_fgw')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + ratio)
                    
            
        else:
        
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] \
                                                    + 3/5*np.log10(1+x[self.pnames.index('cos_inc')]**2) \
                                                    - 3/5*np.log10(1+q[self.pnames.index('cos_inc')]**2)
                        
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] \
                                                    + 3/5*np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])
                    
        return q, float(lqxy)
    
    def draw_strain_skewstep(self, x, iter, beta):
        ##written by SJV for 11yrCW
        
        q = x.copy()
        lqxy = 0
        
        a = 2
        s = 1
        
        diff = skewnorm.rvs(a, scale=s)
        q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] - diff
        lqxy = skewnorm.logpdf(-diff, a, scale=s) - skewnorm.logpdf(diff, a, scale=s)
        
        return q, float(lqxy)
    
    def draw_gwtheta_comb(self, x, iter, beta):
        ##written by SJV for 11yrCW

        q = x.copy()
        lqxy = 0
        
        # the variance of the Gaussian we are drawing from is very small
        # to account for the comb-like structure of the posterior
        sigma = const.c/self.fgw/const.kpc
        
        # now draw an integer to go to a nearby spike
        N = int(0.1/sigma)
        n = np.random.randint(-N,N)
        newval = np.arccos(x[self.pnames.index('cos_gwtheta')]) \
                    + (sigma/2)*np.random.randn() + n*sigma
        
        q[self.pnames.index('cos_gwtheta')] = np.cos(newval)
                
        return q, float(lqxy)

    def draw_gwphi_comb(self, x, iter, beta):
        ##written by SJV for 11yrCW

        # this jump takes into account the comb-like structure of the likelihood 
        # as a function of gwphi, with sharp spikes superimposed on a smoothly-varying function
        # the width of these spikes is related to the GW wavelength
        # this jump does two things:
        #  1. jumps an integer number of GW wavelengths away from the current point
        #  2. draws a step size from a Gaussian with variance equal to half the GW wavelength, 
        #     and takes a small step from its position in a new spike
        
        q = x.copy()
        lqxy = 0
        
        # compute the GW wavelength
        sigma = const.c/self.fgw/const.kpc
        
        # now draw an integer to go to a nearby spike
        # we need to move over a very large number of spikes to move appreciably in gwphi
        # the maximum number of spikes away you can jump 
        # corresponds to moving 0.1 times the prior range
        idx = 0
        for i,p in enumerate(self.params):
            if p.name == 'gwphi':
                idx = i
        pmax = float(str(self.params[idx]).split('=')[-1].split(')')[0])
        pmin = float(str(self.params[idx]).split('=')[1].split(',')[0])
        N = int(0.1*(pmax - pmin)/sigma)

        #N = int(0.1*(self.params[idx]._pmax - self.params[idx]._pmin)/sigma)
        n = np.random.randint(-N,N)
        
        q[self.pnames.index('gwphi')] = x[self.pnames.index('gwphi')] + (sigma/2)*np.random.randn() + n*sigma

        return q, float(lqxy)


def get_global_parameters(pta):
    """Utility function for finding global parameters."""
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)

    gpars = list(set(par for par in pars if pars.count(par) > 1))
    ipars = [par for par in pars if par not in gpars]

    # gpars = np.unique(list(filter(lambda x: pars.count(x)>1, pars)))
    # ipars = np.array([p for p in pars if p not in gpars])

    return np.array(gpars), np.array(ipars)


def get_parameter_groups(pta):
    """Utility function to get parameter groupings for sampling."""
    params = pta.param_names
    ndim = len(params)
    groups = [list(np.arange(0, ndim))]

    # get global and individual parameters
    gpars, ipars = get_global_parameters(pta)
    if gpars.size:
        # add a group of all global parameters
        groups.append([params.index(gp) for gp in gpars])

    # make a group for each signal, with all non-global parameters
    for sc in pta._signalcollections:
        for signal in sc._signals:
            ind = [params.index(p) for p in signal.param_names if not gpars.size or p not in gpars]
            if ind:
                groups.append(ind)

    return groups


def get_cw_groups(pta):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    ang_pars = ['costheta', 'phi', 'cosinc', 'phase0', 'psi']
    mfdh_pars = ['log10_Mc', 'log10_fgw', 'log10_dL', 'log10_h']
    freq_pars = ['log10_Mc', 'log10_fgw', 'pdist', 'pphase']

    groups = []
    for pars in [ang_pars, mfdh_pars, freq_pars]:
        groups.append(group_from_params(pta, pars))

    return groups

def get_parameter_groups_CAW(pta):
    
    """Utility function to get parameter groups for CW sampling.
    These groups should be used instead of the usual get_parameter_groups output.
    Will also include groupings for other signal types for combination with CW signals, if included"""
    
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names

    snames = np.unique([[qq.signal_name for qq in pp._signals] 
                        for pp in pta._signalcollections])
    
    # sort parameters by signal collections
    ephempars = []
    rnpars = []
    cwpars = []
    wnpars = []

    for sc in pta._signalcollections:
        for signal in sc._signals:
            if signal.signal_name == 'red noise':
                rnpars.extend(signal.param_names)
            elif signal.signal_name == 'phys_ephem':
                ephempars.extend(signal.param_names)
            elif signal.signal_name == 'cw':
                cwpars.extend(signal.param_names)
            elif signal.signal_name == 'efac':
                wnpars.extend(signal.param_names)
            elif signal.signal_name == 'equad':
                wnpars.extend(signal.param_names)
            elif 'ecor'in signal.signal_name:
                wnpars.extend(signal.param_names)

                
    
    if 'red noise' in snames:
        
        # create parameter groups for the red noise parameters
        rnpsrs = [ p.split('_')[0] for p in params if '_log10_A' in p and 'gwb' not in p]
        b = [params.index(p) for p in params if 'alpha' in p]
        for psr in rnpsrs:
            groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')]])

        b = [params.index(p) for p in params if 'alpha' in p]
        groups.extend([b])
        
        for alpha in b:
            groups.extend([[alpha, params.index('J0613-0200_red_noise_gamma'), params.index('J0613-0200_red_noise_log10_A')]])
        
        
        for i in np.arange(0,len(b),2):
            groups.append([b[i],b[i+1]])
        
        
        groups.extend([[params.index(p) for p in rnpars]])
        a = [params.index(p) for p in rnpars]
        if 'log10_fgw' in params:
            a.append(params.index('log10_fgw'))
            groups.extend([a])
            
        a = [params.index(p) for p in rnpars]
        if 'gwb_log10_A' in params and 'gwb_gamma' in params:
            a.append(params.index('gwb_log10_A'))
            a.append(params.index('gwb_gamma'))
            if 'gwb_log10_fbend' in params:
                a.append(params.index('gwb_log10_fbend'))

            groups.extend([a])
            

    #addition for sampling wn
    #this groups efac and equad together for each pulsar
    if 'efac' in snames and 'equad' in snames:
    
        # create parameter groups for the red noise parameters
        wnpsrs = [ p.split('_')[0] for p in params if '_efac' in p]

        for psr in wnpsrs:
            groups.extend([[params.index(psr + '_efac'), params.index(psr + '_log10_equad')]])
            
        groups.extend([[params.index(p) for p in wnpars]])
        
    if 'efac' in snames and 'equad' in snames and 'red noise' in snames:
    
        # create parameter groups for the red noise parameters
        psrs = [ p.split('_')[0] for p in params if '_efac' in p and '_log10_A' in p and 'gwb' not in p]

        for psr in psrs:
            groups.extend([[params.index(psr + '_efac'), params.index(psr + '_log10_equad'),
                            params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')]])
            
                    
    # set up groups for the BayesEphem parameters
    if 'phys_ephem' in snames:
        
        ephempars = np.unique(ephempars)
        juporb = [p for p in ephempars if 'jup_orb' in p]
        groups.extend([[params.index(p) for p in ephempars if p not in juporb]])
        groups.extend([[params.index(jp) for jp in juporb]])
        for i1 in range(len(juporb)):
            for i2 in range(i1+1, len(juporb)):
                groups.extend([[params.index(p) for p in [juporb[i1], juporb[i2]]]])
        
    if 'cw' in snames:
        
    
        # divide the cgw parameters into two groups: 
        # the common parameters and the pulsar phase and distance parameters
        cw_common = np.unique(list(filter(lambda x: cwpars.count(x)>1, cwpars)))
        groups.extend([[params.index(cwc) for cwc in cw_common]])

        cw_pulsar = np.array([p for p in cwpars if p not in cw_common])
        if len(cw_pulsar) > 0:
            
            pdist_params = [ p for p in cw_pulsar if 'p_dist' in p ]
            pphase_params = [ p for p in cw_pulsar if 'p_phase' in p ]
            
            for pd,pp in zip(pdist_params,pphase_params):
                #groups.extend([[params.index(pd), params.index('cos_gwtheta'), params.index('gwphi')]])
                groups.extend([[params.index(pd), params.index('log10_mc')]])
                groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc')]])
                groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), 
                                params.index('cos_inc'), params.index('psi')]])
                groups.extend([[params.index(pd), params.index(pp), 
                                params.index('log10_mc')]])
                if 'log10_fgw' in cw_common:
                    groups.extend([[params.index(pd), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_fgw'), 
                                    params.index('cos_inc'), params.index('psi')]])
                    groups.extend([[params.index(pd), params.index(pp), 
                                    params.index('log10_fgw')]])
                    
                    groups.extend([[params.index(pd), params.index(pp), 
                                    params.index('log10_fgw'), params.index('log10_mc')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), 
                                    params.index('cos_inc'), params.index('psi'), params.index('log10_fgw')]])
            
        # now try other combinations of the common cgw parameters
        
        #adapted from get_cw_groups to simplify code
        ang_pars = ['cos_gwtheta', 'gwphi', 'cos_inc', 'phase0', 'psi']
        loc_pars = ['cos_gwtheta', 'gwphi']
        orb_pars = ['cos_inc', 'phase0', 'psi']
        mfdh_pars = ['log10_mc', 'log10_fgw', 'log10_dL', 'log10_h']
        freq_pars = ['log10_mc', 'log10_fgw', 'p_dist', 'p_phase']
        cw_pars = ang_pars.copy()
        cw_pars.extend(mfdh_pars)

        amp_pars = ['log10_mc', 'log10_h']
        
        #parameters to catch and match gwb signals - if set to constant or not included, will skip

        crn_pars = ['gwb_gamma', 'gwb_log10_A']
        crn_cw_pars = crn_pars.copy()
        crn_cw_pars.extend(cw_pars)
        bpl_pars = ['gwb_gamma', 'gwb_log10_A', 'gwb_log10_fbend']
        bpl_cw_pars = bpl_pars.copy()
        bpl_cw_pars.extend(cw_pars)
        
        groups1 = []
        
        for pars in [ang_pars, loc_pars, orb_pars, mfdh_pars, freq_pars, cw_pars, crn_pars, crn_cw_pars, bpl_pars, bpl_cw_pars]:
            if any(item in params for item in pars):
                groups1.append(group_from_params(pta, pars))

        for group in groups1:
            if any(params.index(item) in group for item in amp_pars):
                pass
            else:
                for p in amp_pars:
                    if p in params:
                        g = group.copy()
                        g.append(params.index(p))
                        groups1.append(g)

        groups.extend(groups1)
        
                
                

    if 'cw' in snames and 'phys_ephem' in snames:
        # add a group that contains the Jupiter orbital elements and the common GW parameters
        juporb = list([p for p in ephempars if 'jup_orb' in p])

        cw_common = list(np.unique(list(filter(lambda x: cwpars.count(x)>1, cwpars))))

        
        myparams = juporb + cw_common
        
        groups.extend([[params.index(p) for p in myparams]])
        
        if 'gwb_log10_A' in params and 'gwb_gamma' in params:
            myparams += ['gwb_log10_A', 'gwb_gamma']
            groups.extend([[params.index(p) for p in myparams]])
                
    for group in groups:
        if len(group) == 0:
            groups.remove(group)
    return groups


def group_from_params(pta, params):
    gr = []
    for p in params:
        for q in pta.param_names:
            if p in q:
                gr.append(pta.param_names.index(q))
    return gr


def setup_sampler(pta, outdir='chains', resume=False, empirical_distr=None):
    """
    Sets up an instance of PTMCMC sampler.

    We initialize the sampler the likelihood and prior function
    from the PTA object. We set up an initial jump covariance matrix
    with fairly small jumps as this will be adapted as the MCMC runs.

    We will setup an output directory in `outdir` that will contain
    the chain (first n columns are the samples for the n parameters
    and last 4 are log-posterior, log-likelihood, acceptance rate, and
    an indicator variable for parallel tempering but it doesn't matter
    because we aren't using parallel tempering).

    We then add several custom jump proposals to the mix based on
    whether or not certain parameters are in the model. These are
    all either draws from the prior distribution of parameters or
    draws from uniform distributions.
    """

    # dimension of parameter space
    params = pta.param_names
    ndim = len(params)

    # initial jump covariance matrix
    cov = np.diag(np.ones(ndim) * 0.1**2)

    # parameter groupings
    groups = get_parameter_groups(pta)

    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups,
                     outDir=outdir, resume=resume)
    np.savetxt(outdir+'/pars.txt',
               list(map(str, pta.param_names)), fmt='%s')
    np.savetxt(outdir+'/priors.txt',
               list(map(lambda x: str(x.__repr__()), pta.params)), fmt='%s')

    # additional jump proposals
    jp = JumpProposal(pta, empirical_distr=empirical_distr)

    # always add draw from prior
    sampler.addProposalToCycle(jp.draw_from_prior, 5)

    # try adding empirical proposals
    if empirical_distr is not None:
        print('Adding empirical proposals...\n')
        sampler.addProposalToCycle(jp.draw_from_empirical_distr, 10)

    # Red noise prior draw
    if 'red noise' in jp.snames:
        print('Adding red noise prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_red_prior, 10)

    # DM GP noise prior draw
    if 'dm_gp' in jp.snames:
        print('Adding DM GP noise prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 10)

    # DM annual prior draw
    if 'dm_s1yr' in jp.snames:
        print('Adding DM annual prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dm1yr_prior, 10)

    # DM dip prior draw
    if 'dmexp' in jp.snames:
        print('Adding DM exponential dip prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dmexpdip_prior, 10)

    # DM cusp prior draw
    if 'dm_cusp' in jp.snames:
        print('Adding DM exponential cusp prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dmexpcusp_prior, 10)

    # DMX prior draw
    if 'dmx_signal' in jp.snames:
        print('Adding DMX prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dmx_prior, 10)

    # Ephemeris prior draw
    if 'd_jupiter_mass' in pta.param_names:
        print('Adding ephemeris model prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

    # GWB uniform distribution draw
    if 'gw_log10_A' in pta.param_names:
        print('Adding GWB uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

    # Dipole uniform distribution draw
    if 'dipole_log10_A' in pta.param_names:
        print('Adding dipole uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

    # Monopole uniform distribution draw
    if 'monopole_log10_A' in pta.param_names:
        print('Adding monopole uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_monopole_log_uniform_distribution, 10)

    # Altpol uniform distribution draw
    if 'log10Apol_tt' in pta.param_names:
        print('Adding alternative GW-polarization uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_altpol_log_uniform_distribution, 10)

    # BWM prior draw
    if 'bwm_log10_A' in pta.param_names:
        print('Adding BWM prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)

    # CW prior draw
    if 'cw_log10_h' in pta.param_names:
        print('Adding CW strain prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)
    if 'cw_log10_Mc' in pta.param_names:
        print('Adding CW prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_cw_distribution, 10)

    return sampler
