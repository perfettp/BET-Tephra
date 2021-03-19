#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  BET-Tephra@OV is a Python software package to compute and visualize
  long- and short- term eruption forecasting and probabilistic tephra hazard
  assessment of Campi Flegrei caldera.

  BET_Tephra@OV was realized in the framework of the Italian Civil Protection
  Department (DPC) - INGV Research Project “OBIETTIVO 5 (V2): Implementazione
  nell’ambito delle attività di sorveglianza del vulcano Campi Flegrei di una
  procedura operativa per la stima in tempo quasi reale della probabilità di
  eruzione, probabilità di localizzazione della bocca eruttiva, e probabilità
  di accumulo delle ceneri al suolo in caso di eruzione di tipo esplosivo”,
  2014-2015.

  Copyright(C) 2021 Paolo Perfetti
  Copyright(C) 2015-2017 Paolo Perfetti, Roberto Tonini and Jacopo Selva

  This file is part of BET-Tephra@OV software.

  BET-Tephra@OV is free software: you can redistribute it and/or modify it under
  the terms of the GNU Affero General Public License as published by the
  Free Software Foundation, either version 3 of the License, or (at your
  option) any later version.

  BET-Tephra@OV is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
  more details.

  You should have received a copy of the GNU Affero General Public License
  along with BET-Tephra@OV. If not, see <http://www.gnu.org/licenses/>.

"""

import os
import requests
from sys import argv
from bet.conf import BetConf, MonitoringConf
from bet.service import ConfService
import bet.run.tephra
import bet.run.bet_ef
import logging
from bet.data import UTMPoint, HCProbPoint
import bet.messaging.celery.tasks
from datetime import datetime, timedelta
from bet.function import get_season_number, plot_vh_prob, \
    get_load_kg, export_contours, create_run_dir, log_to_file, get_logger
from bet.function.cli import opts_parser
import bet.messaging.celery.tasks
from math import exp
import numpy as np
import numpy.matlib as ml
from scipy.spatial import distance
import tempfile
import pickle
import bet.messaging.celery.tasks


# DA pyBetEF (Tonini)
# self._bet_ef_result.vent_prob_list (ESeff) contiene una lista di indici di
# Size (da 1 a 4) e Loc (da 1 a 700),per cui la  prob > 0

# Da ceneri
# self._tephra_results.hazards (hazards): lista dei modelli disponibili per t0
# [fall3d,hazmap]
# self._tephra_results.get_haz_loads_n(haz_model)
# (conf[haz_model]['n_sim']): numero di simulazioni per il modello
#   haz_model disponibili (tutti i t0 nella finestra di inferenza)
# self._tephra_results.get_haz_loads(haz_model)
# (conf[haz_model][i_sim]): n_size matrici di load in kPa per questo
#   modello e i_sim (letti da files .nc)

# Settings pyBetVH
#   self._bet_vh_conf.n_areas (nAreeOutput): numero totale aree target (1747)
#   self._bet_vh_conf.load_thresholds (soglie): soglie di carico in kPa
#       ndArray(nSoglie)
#   len(self._bet_vh_conf.load_thresholds) (nSoglie): numero di soglie di carico
#        per le curve di hazard (12)
#   self._bet_vh_conf.grid2area(iloc, iarea) (grid2Area): per ogni iloc,
#      corrispondenza tra ogni area di output ed 1 punto della griglia ceneri
#   self._bet_vh_conf.lambdas[haz_model] (conf[haz_model]['lambda']): parametro
#      corrispondente al model haz_model (10)


def beta_sampling_vh(alpha, beta):
    if alpha != 0 and beta != 0:
        return np.random.beta(alpha, beta, size=1)
    elif alpha == 0 and beta != 0:
        return 0
    elif alpha != 0 and beta == 0:
        return 1
    else:
        raise Exception("alpha and beta are both zero!")

v_beta_sampling = np.vectorize(beta_sampling_vh, cache=False,
                               otypes=[np.float])


class BetVHOut(object):
    def __init__(self, date, exp_window=None, areas_n=2050,
                 n_samples=None, n_samples_st=None,
                 n_samples_lt=None, n78_weight=None, tmp_dir=None,
                 day_prob=None):
        self._date = date
        self._exp_window = exp_window
        self._areas_n = areas_n
        self._n_samples = n_samples,
        self._n_samples_st = n_samples_st,
        self._n_samples_lt = n_samples_lt
        self._n78_weight = n78_weight
        self._tmp_dir = tmp_dir
        self._hc = list([None for i in range(areas_n)])
        self._alpha78_st_path = None
        self._beta78_st_path = None
        self._alpha78_lt_path = None
        self._beta78_lt_path = None
        self._day_prob = None

    @property
    def date(self):
        return self._date

    @property
    def exp_window(self):
        return self._exp_window

    @exp_window.setter
    def exp_window(self, ew):
        self._exp_window = ew

    @property
    def areas_n(self):
        return self._areas_n

    @property
    def hc(self):
        return self._hc

    @hc.setter
    def hc(self, val):
        self._hc = val

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, val):
        self._n_samples = val

    @property
    def n_samples_st(self):
        return self._n_samples_st

    @n_samples_st.setter
    def n_samples_st(self, val):
        self._n_samples_st = val

    @property
    def n_samples_lt(self):
        return self._n_samples_lt

    @n_samples_lt.setter
    def n_samples_lt(self, val):
        self._n_samples_lt = val

    @property
    def n78_weight(self):
        return self._n78_weight

    @n78_weight.setter
    def n78_weight(self, val):
        self._n78_weight = val

    @property
    def tmp_dir(self):
        return self._tmp_dir

    @tmp_dir.setter
    def tmp_dir(self, val):
        self._tmp_dir = val

    @property
    def day_prob(self):
        return self._day_prob

    @day_prob.setter
    def day_prob(self, val):
        self._day_prob = val

    @property
    def alpha78_st_path(self):
        return self._alpha78_st_path

    @alpha78_st_path.setter
    def alpha78_st_path(self, val):
        self._alpha78_st_path = val

    @property
    def beta78_st_path(self):
        return self._beta78_st_path

    @beta78_st_path.setter
    def beta78_st_path(self, val):
        self._beta78_st_path = val

    @property
    def alpha78_lt_path(self):
        return self._alpha78_lt_path

    @alpha78_lt_path.setter
    def alpha78_lt_path(self, val):
        self._alpha78_lt_path = val

    @property
    def beta78_lt_path(self):
        return self._beta78_lt_path

    @beta78_lt_path.setter
    def beta78_lt_path(self, val):
        self._beta78_lt_path = val


class BetVHSamplesCfg(object):
    def __init__(self, area_id, n_samples, n_samples_st, n_samples_lt,
                 thresholds_n, percentiles, sizes,  sizes_prior, n78_weight,
                 tmp_dir, day_prob, alpha78_st_area, alpha78_lt_area,
                  beta78_st_area,  beta78_lt_area, ):
        self._area_id = area_id
        self._n_samples = n_samples
        self._n_samples_st = n_samples_st
        self._n_samples_lt = n_samples_lt
        self._thresholds_n = thresholds_n
        self._percentiles = percentiles
        self._sizes = sizes
        self._sizes_prior = sizes_prior
        self._n78_weight = n78_weight
        self._day_prob = day_prob
        self._alpha78_st_area = alpha78_st_area
        self._alpha78_lt_area = alpha78_lt_area
        self._beta78_st_area = beta78_st_area
        self._beta78_lt_area = beta78_lt_area

    @property
    def area_id(self):
        return self._area_id

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_samples_st(self):
        return self._n_samples_st

    @property
    def n_samples_lt(self):
        return self._n_samples_lt

    @property
    def thresholds_n(self):
        return self._thresholds_n

    @property
    def percentiles(self):
        return self._percentiles

    @property
    def sizes(self):
        return self._sizes

    @property
    def sizes_prior(self):
        return self._sizes_prior

    @property
    def n78_weight(self):
        return self._n78_weight

    @property
    def day_prob(self):
        return self._day_prob

    @property
    def alpha78_st_area(self):
        return self._alpha78_st_area

    @property
    def beta78_st_area(self):
        return self._beta78_st_area

    @property
    def alpha78_lt_area(self):
        return self._alpha78_lt_area

    @property
    def beta78_lt_area(self):
        return self._beta78_lt_area


class BetVHModel(object):
    def __init__(self, bet_conf=None,
                 bet_ef_result=None,
                 tephra_result=None,
                 exp_window=None,
                 tmpdir=None):
        """

        """
        self._bet_conf = bet_conf
        self._bet_ef_result = bet_ef_result
        self._tephra_results = tephra_result
        self._exp_window = exp_window
        self._vent_tephra_haz_map = None
        self._rundate = None
        self._n_samples = None
        self._n_samples_st = None
        self._n_samples_lt = None
        self._load_thresholds = list()
        self._percentiles = list()
        if tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="bet-",
                                            dir="/mnt/bet-data/tmp")
        else:
            self._tmpdir = tmpdir
        self._result = BetVHOut(bet_conf.obs_time,
                                tmp_dir=self._tmpdir)

    @property
    def result(self):
        return self._result

    def create_mapping(self):
        print "Create grids mapping file %s" % \
              self._bet_conf.BET['Hazard']['grids_mapping']
        vent_tephra_haz_map = np.empty([self._bet_conf.vent_grid_n,
                                        self._bet_conf.hazard_grid_n],
                                       dtype=int)
        vent_tephra = UTMPoint(
            easting=float(self._bet_conf.BET['Tephra']['vent_easting']),
            northing=float(self._bet_conf.BET['Tephra']['vent_northing']))

        for iloc in range(self._bet_conf.vent_grid_n):
        # for iloc in range(5):
            print iloc

            shift_x = (self._bet_conf.vent_grid[iloc].point.easting -
                       vent_tephra.easting)

            shift_y = (self._bet_conf.vent_grid[iloc].point.northing -
                       vent_tephra.northing)

            griglia_ceneri_xeff = np.array(
                      np.array([tp.easting for tp in
                                self._bet_conf.tephra_grid]) - shift_x)

            griglia_ceneri_yeff = np.array(
                      np.array([tp.northing for tp in
                                self._bet_conf.tephra_grid]) - shift_y)

            # 2 x self._bet_conf.tephra_grid_n
            xa = np.array([(griglia_ceneri_xeff[i], griglia_ceneri_yeff[i])
                           for i in range(len(griglia_ceneri_xeff))])

            # 2 x self._bet_conf.hazard_grid_n
            xb = np.array([(p.easting, p.northing) for p in
                           self._bet_conf.hazard_grid])

            tmp = distance.cdist(xb, xa, 'euclidean')

            for iarea in range(self._bet_conf.hazard_grid_n):
                vent_tephra_haz_map[iloc][iarea] = np.argmin(tmp[iarea])

        np.save(self._bet_conf.BET['Hazard']['grids_mapping'],
                vent_tephra_haz_map)

    def load_mapping(self):
        try:
            tmp = np.load(self._bet_conf.BET['Hazard']['grids_mapping'])
        except IOError:
            self.create_mapping()
            tmp = np.load(self._bet_conf.BET['Hazard']['grids_mapping'])
        if (tmp.shape[0] != self._bet_conf.vent_grid_n
            or tmp.shape[1] != self._bet_conf.hazard_grid_n):
            raise Exception("Wrong grid dimensions detected! Please check the grids in use")
        self._vent_tephra_haz_map = tmp

    def grid2area(self, iloc, iarea):
        if self._vent_tephra_haz_map is None:
            self.load_mapping()
        return self._vent_tephra_haz_map[iloc][iarea]

    def run(self, logger=None):
        """

        """

        if logger is None:
            logger = get_logger(level=logging.DEBUG)

        logger.debug("Running BetVH")

        if self._bet_ef_result is None or self._bet_conf is None:
            raise Exception("BetVHModel not (completely) initialized!")

        self._rundate = self._bet_conf.obs_time
        print 'Running on rundate %s' % self._rundate

        self._load_thresholds = \
            np.array(self._bet_conf.BET['Hazard']['load_thresholds'])

        self._sizes = self._bet_conf.BET['Styles']['sizes']

        # CALCOLO PESO SHORT-TERM VS LONG-TERM
        print "CALCOLO PESO SHORT-TERM VS LONG-TERM"

        exp_wind_begin = self._exp_window['begin']
        exp_wind_end = self._exp_window['end']
        exp_wind_time = exp_wind_end - exp_wind_begin

        print "Window begin time: {}".format(exp_wind_begin)
        print "Exposure window lenght: {}".format(exp_wind_time)
        print "Window end time: {}".format(exp_wind_end)

        self._result.exp_window = self._exp_window


        # print "WARNING: aggiungere parametro per giorno EF e variarne il peso"
        day_prob_i = (exp_wind_begin - self._bet_ef_result.date).days
        day_prob = float(self._bet_conf.BET['daily_probabilities'][day_prob_i])
        print "BetVH: Daily probability: exp_wind_begin {}, bet_ef_out {}, days {}, daily_prob {}".format(
                    exp_wind_begin,
                    self._bet_ef_result.date,
                    day_prob_i,
                    day_prob)
        self._result.day_prob = day_prob

        if self._tephra_results is None:
            print "WARNING: tephra_result is None"
            n_tephra_sim = 0
        else:
            n_tephra_sim = self._tephra_results.n_sim
        print "Tephra simulation in window interval: {}".format(n_tephra_sim)

        if n_tephra_sim > 0:
            print "Forecast orig time: {}".format(self._tephra_results.forecast_orig_t)
            if self._tephra_results.forecast_orig_t:
                ddt = exp_wind_end - self._tephra_results.forecast_orig_t
            else:
                ddt = timedelta(days=100)
            dt = ddt.days + (ddt.seconds/3600)/24.

            # dt in giorni
            # dt = round(ddt.seconds/3600)
            # n_tephra_sim = 1

            qw = exp(-pow((dt/4), 2))
            qs = exp(-pow(((exp_wind_time.days * 24) +
                           (round(exp_wind_time.seconds/3600))/n_tephra_sim)/36, 2))
            n78_weight = qw * qs
            self._result.n78_weight = n78_weight
            print "Parameters: dt {}, qw {}, qs {}, n78_weight: {}".format(
                    dt, qw, qs, n78_weight)

        else:
            print "WARNING: no valid tephra data found in interval!"
            self._result.n78_weight = 0

        self._result.n_samples = int(self._bet_conf.BET['sampling'])
        self._result.n_samples_st = round(self._result.n78_weight *
                                          self._result.n_samples)
        self._result.n_samples_lt = self._result.n_samples - \
                                        self._result.n_samples_st

        print "Samples distribution: n_samples_st {}, n_samples_lt {}".format(
            self._result.n_samples_st,
            self._result.n_samples_lt
        )
        self._percentiles = [float(x)
                    for x in self._bet_conf.BET['Hazard']['percentiles']]

        # Setting paths for long term parameters
        lt_dir = self._bet_conf.BET['Hazard']['alpha_beta_lt_dir']
        season_number = str(get_season_number(self._rundate))
        self._result.alpha78_lt_path = os.path.join(
                lt_dir,
                'alpha78_lt-'+season_number+'.npy')
        self._result.beta78_lt_path = os.path.join(
                lt_dir,
                'beta78_lt-'+season_number+'.npy')


        alpha78_st = np.zeros((
            self._bet_conf.hazard_grid_n,
            self._bet_ef_result.n_locations,
            len(self._sizes),
            len(self._load_thresholds)))
        beta78_st = np.zeros((
            self._bet_conf.hazard_grid_n,
            self._bet_ef_result.n_locations,
            len(self._sizes),
            len(self._load_thresholds)))

        if self._tephra_results is not None:
            hazards_lambda = dict((haz, float(self._bet_conf.BET['Hazard']
                                             ['HazardModels'][haz]['lambda']))
                                 for haz in self._tephra_results.haz_loads.keys())

            print "Tephra model haz_load.keys: %s" % \
                  self._tephra_results.haz_loads.keys()
            for haz_model in self._tephra_results.haz_loads.keys():
                print "Working on haz_model: %s" % haz_model
                if haz_model == 'hazmap':
                    print "WARNING: skipping hazmap"
                    continue
                print "Calculating load probabilities"
                prob_model = np.zeros((
                    self._bet_conf.hazard_grid_n,
                    self._bet_ef_result.n_locations,
                    len(self._sizes),
                    len(self._load_thresholds)))

                for haz_data in self._tephra_results.get_haz_loads(haz_model):
                    for size in self._sizes:
                        i_size = self._sizes.index(size)
                        if (len(self._tephra_results.get_haz_loads(haz_model)) == 0 or
                            self._tephra_results.get_haz_loads_n(haz_model, size) == 0):
                            # Skip if no data is present for current model/size
                            print "Warning: no tephra result for model %s and " \
                                  "size %s" % (haz_model, size)
                            continue
                        haz_res = haz_data.get_model_res(size)
                        if haz_res:
                            load_orig = bet.run.tephra.load_tephra_matrix(
                                haz_model,
                                haz_data.get_model_res(size)['res_file']
                            )
                            load = load_orig.T.ravel()
                            for i_vent in self._bet_ef_result.eff_vents_i:
                            # for i_vent in range(len(self._bet_ef_result.vent_prob_list)):
                                for i_area in range(self._bet_conf.hazard_grid_n):

                                    # dato un iloc, iarea corrisponde ad un preciso
                                    # punto della matrice di simulazione ceneri
                                    isel_tephra = self.grid2area(i_vent, i_area)
                                    try:
                                        load_iarea = load[isel_tephra] * 0.00980665
                                    except:
                                        load_iarea = 0

                                    float_th = np.array([float(v)
                                                         for v in
                                                         self._load_thresholds])
                                    prob_model[i_area, i_vent, i_size, :] += \
                                        np.greater(load_iarea, float_th)

                for size in self._sizes:
                    i_size = self._sizes.index(size)
                    if (len(self._tephra_results.get_haz_loads(haz_model)) == 0 or
                        self._tephra_results.get_haz_loads_n(haz_model, size) == 0):
                        # Skip if no data is present for current model/size
                        continue
                    # prob_model[:, :, size, :] /= \
                    prob_model[:][:][i_size][:] /= \
                        self._tephra_results.get_haz_loads_n(haz_model, size)

                print "Calculating alpha/beta parameters"

                for i_area in range(self._bet_conf.hazard_grid_n):
                    for i_vent in range(len(self._bet_ef_result.vent_prob_list)):
                        for size in self._sizes:
                            i_size = self._sizes.index(size)
                            alpha78_st[i_area, i_vent, i_size, 0] += (
                                    hazards_lambda[haz_model] *
                                    prob_model[i_area, i_vent, i_size, 0])

                            beta78_st[i_area, i_vent, i_size, 0] += (
                                    hazards_lambda[haz_model] *
                                    (1-prob_model[i_area, i_vent, i_size, 0]))

                            for i_thresh in range(1, len(self._load_thresholds), 1):
                                thetai = prob_model[i_area][i_vent][i_size][i_thresh]
                                thetai1 = prob_model[i_area][i_vent][i_size][i_thresh-1]
                                if thetai1 > 0:
                                    fact = thetai/thetai1
                                    alpha78_st[i_area, i_vent, i_size, i_thresh] += \
                                        (hazards_lambda[haz_model] * fact)
                                    beta78_st[i_area, i_vent, i_size, i_thresh] += \
                                        (hazards_lambda[haz_model] * (1-fact))
                                else:
                                    alpha78_st[i_area, i_vent, i_size, i_thresh] += 0
                                    beta78_st[i_area, i_vent, i_size, i_thresh] += \
                                        hazards_lambda[haz_model]
        else:
            print "self._tephra_results is stil None!, Using just long term"

        print "Dumping alpha/beta parameters on disk"
        self._result.alpha78_st_path = os.path.join(self._tmpdir,
                                                  "alpha78_st.npy")
        np.save(self._result.alpha78_st_path, alpha78_st)
        print "alpha saved on %s" % self._result.alpha78_st_path

        self._result.beta78_st_path = os.path.join(self._tmpdir,
                                                   "beta78_st.npy")
        np.save(self._result.beta78_st_path, beta78_st)
        print "beta saved on %s"% self._result.beta78_st_path


# This is thought to run on one area per invocation
def sample_per_area_cfg(area_cfg,
                        bet_ef_res,
                        tephra_res,
                        tephra_samples,
                        sampling=True):

    if sampling:
        print("Probabilities and sampling for area {0}".format(
                area_cfg.area_id))
    else:
        print("Probabilities for area {0}".format(
                area_cfg.area_id))
    n_samples = area_cfg.n_samples
    n_st = area_cfg.n_samples_st
    n_lt = area_cfg.n_samples_lt
    n_thresh = area_cfg.thresholds_n
    percentiles = area_cfg.percentiles
    sizes = area_cfg.sizes
    sizes_prior = area_cfg.sizes_prior
    n78_weight = area_cfg.n78_weight
    day_prob = area_cfg.day_prob
    alpha78_st_area = area_cfg.alpha78_st_area
    alpha78_lt_area = area_cfg.alpha78_lt_area
    beta78_st_area = area_cfg.beta78_st_area
    beta78_lt_area = area_cfg.beta78_lt_area

    prob_mean_abs = np.zeros((n_thresh))
    prob_mean_cond = np.zeros((n_thresh))

    prob_n78st_means = np.divide(alpha78_st_area,
                                 (alpha78_st_area + beta78_st_area))

    prob_n78lt_means = np.divide(alpha78_lt_area,
                                 (alpha78_lt_area + beta78_lt_area))

    prob_n78_means_cum = np.nan_to_num(n78_weight *
                                       np.cumprod(prob_n78st_means, 2)) + \
                             ((1 - n78_weight) *
                              np.cumprod(prob_n78lt_means, 2)) # 3 => sulle soglie

    # print "prob_n78st_means: {}".format(prob_n78st_means)
    # print "prob_n78lt_means: {}".format(prob_n78lt_means)
    # print "prob_n78_means_cum: {}".format(prob_n78_means_cum)

    sample_abs = np.zeros((n_samples, n_thresh))
    sample_cond = np.zeros((n_samples, n_thresh))

    rand_perm = np.random.permutation(n_samples)

    # Loop over non-zero probability vent areas
    for i_vent in bet_ef_res.eff_vents_i:
        vent = bet_ef_res.vent_prob_list[i_vent]
        for i_size in range(len(sizes)):
            if sizes[i_size] == 'E':
                continue
            size = sizes[i_size]

            hc_mean = np.squeeze(
                prob_n78_means_cum[i_vent, i_size, :])
            # print("hc_mean: {0}".format(hc_mean))

            prob_mean_abs += \
                        bet_ef_res.unrest.mean * \
                        bet_ef_res.magmatic.mean * \
                        bet_ef_res.eruption.mean * \
                        day_prob * \
                        vent.ave.mean * \
                        vent.sizes_ave[size].mean *\
                        sizes_prior[i_size] * \
                        hc_mean

            # print("prob_mean_abs: {0}".format(prob_mean_abs))

            prob_mean_cond += \
                        hc_mean * \
                        vent.ave.mean * \
                        vent.sizes_ave[size].mean *\
                        sizes_prior[i_size]
            # print("prob_mean_cond: {0}".format(prob_mean_cond))

            if sampling:
                # shape (n_samples,)
                tmp = bet_ef_res.unrest.samples * \
                      bet_ef_res.magmatic.samples * \
                      bet_ef_res.eruption.samples * \
                      vent.ave.samples * \
                      vent.sizes_ave[size].samples * \
                      tephra_samples[i_size]
                # print("tmp: {0}".format(tmp))

                # print("tmp shape {0}\n".format(tmp.shape))

                # shape (n_samples, n_thresh)
                tmp_abs = ml.repmat(tmp, n_thresh, 1).T
                # print("tmp_abs shape {0}\n".format(tmp_abs.shape))
                # print("tmp_abs: {0}".format(tmp_abs))

                # shape (n_samples,)
                tmp = vent.ave.samples * \
                      vent.sizes_ave[size].samples * \
                      tephra_samples[i_size]
                # print("tmp shape {0}\n".format(tmp.shape))

                # shape (n_samples, n_thresh)
                tmp_cond = ml.repmat(tmp,  n_thresh, 1).T
                # print("tmp_cond shape {0}\n".format(tmp_cond.shape))

                alpha_st = np.squeeze(alpha78_st_area[i_vent, i_size, :])
                # shape (n_st, n_thresh)
                alpha_st_all = ml.repmat(alpha_st, n_st, 1)
                # print("alpha_st_all: \n\t{0}".format(alpha_st_all))

                alpha_lt = np.squeeze(alpha78_lt_area[i_vent, i_size, :])
                # shape (n_lt, n_thresh)
                alpha_lt_all = ml.repmat(alpha_lt, n_lt, 1)

                # shape (n_samples, n_thresh) con n_samples = n_st + n_lt
                alpha_all = np.concatenate((alpha_st_all, alpha_lt_all), 0)
                # print("alpha_all: \n\t{0}".format(alpha_all))

                beta_st = np.squeeze(beta78_st_area[i_vent, i_size, :])
                # print("beta_st: \n\t{0}".format(beta_st))
                # shape (n_st, n_thresh)
                beta_st_all = ml.repmat(beta_st, n_st, 1)
                # print("beta_st_all: \n\t{0}".format(beta_st_all))
                beta_lt = np.squeeze(beta78_lt_area[i_vent, i_size, :])
                # print("beta_lt: \n\t{0}".format(beta_lt))
                # shape (n_lt, n_thresh)
                beta_lt_all = ml.repmat(beta_lt, n_lt, 1)
                # print("beta_lt_all: \n\t{0}".format(beta_lt_all))
                # shape (n_samples, n_thresh) con n_samples = n_st + n_lt
                beta_all = np.concatenate((beta_st_all, beta_lt_all), 0)
                # print("beta_all: \n\t{0}".format(beta_all))

                # alpha_all = np.ones((n_samples, n_thresh))
                # beta_all = np.ones((n_samples, n_thresh))

                # shape (n_samples, n_thresh)
                # tmp8ord = np.random.beta(alpha_all, beta_all, size=1)
                tmp8ord = v_beta_sampling(alpha_all, beta_all)
                tmp8 = tmp8ord[rand_perm, :]
                tmp_new = np.cumprod(tmp8, 1)

                # print("tmp_new shape {0}\n".format(tmp_new.shape))
                # print("sample_abs shape {0}\n".format(sample_abs.shape))    # OK
                # print("sample_cond shape {0}\n".format(sample_cond.shape))  # OK
                sample_abs += np.multiply(tmp_new, tmp_abs)
                sample_cond += np.multiply(tmp_new, tmp_cond)
            else:
                sample_abs = np.empty((n_samples, n_thresh))
                sample_cond = np.empty((n_samples, n_thresh))

    hp = HCProbPoint(
        abs_mean=prob_mean_abs,
        con_mean=prob_mean_cond,
        samples_abs_mean=np.mean(sample_abs),
        samples_abs_perc=np.percentile(sample_abs,
                                       percentiles,
                                       axis=0),
        samples_con_mean=np.mean(sample_cond),
        samples_con_perc=np.percentile(sample_cond,
                                       percentiles,
                                       axis=0))
    return hp


if __name__ == "__main__":

    opts = vars(opts_parser().parse_args(argv[1:]))
    load_dir = opts['load']
    sampling = not opts['no_sampling']

    if load_dir:
        print("Loading data from {0}".format(load_dir))
        with open(os.path.join(load_dir, "bet_conf.pick")) as f:
            bet_conf = pickle.load(f)
        with open(os.path.join(load_dir, "mon_conf.pick")) as f:
            mon_conf = pickle.load(f)
        with open(os.path.join(load_dir, "bet_ef_out.pick")) as f:
            bet_ef_out = pickle.load(f)
        with open(os.path.join(load_dir, "tephra_out.pick")) as f:
            tephra_out = pickle.load(f)
        with open(os.path.join(load_dir, "bet_vh_out.pick")) as f:
            bet_vh_out = pickle.load(f)
        print("All data loaded.")

        print bet_vh_out.date

        bet_conf = BetConf(opts['conf'])
        bet_conf.load_hazard_grid()

        cond_perc_to_plot = float(bet_conf.BET['Hazard']['cond_perc_to_plot'])
        load_thresholds = np.array([float(t)
                                    for t in bet_conf.BET['Hazard']
                                    ['load_thresholds']])

        cond_p = np.array([pp.con_mean for pp in bet_vh_out.hc])

        hc_cond_p = np.array([get_load_kg(point_data,
                                          load_thresholds,
                                          cond_perc_to_plot)
                              for point_data in cond_p])



        export_contours(bet_conf.hazard_grid,
                        hc_cond_p,
                        [10, 100, 300, 500, 1000],
                        bet_conf,
                        basename="vh_cond_p",
                        rundir=load_dir,
                        plot=True)

    else:

        if opts['obs_time']:
            obs_time = datetime.strptime(opts['obs_time'], "%Y%m%d_%H%M%S")
        else:
            obs_time = datetime.now()

        bet_conf = BetConf(opts['conf'], obs_time=obs_time)

        if opts['run_dir']:
            run_dir = opts['run_dir']
        else:
            run_dir = create_run_dir(bet_conf.BET['data_dir'], obs_time)

        if run_dir is None:
            print "Cannot create run_dir, exiting"
            exit(-1)

        if opts['monitoring_file']:
            print "Parsing monitoring parameters from file %s" \
                  % opts['monitoring_file']
            mon_conf = MonitoringConf()
            with open(opts['monitoring_file'], 'r') as mon_file:
                mon_conf.from_json(mon_file.read())
        else:
            cs = ConfService(volcano_name='Campi_Flegrei', elicitation=6,
                             runmodel_class='TestModel',
                             mapmodel_name='CardinalModelTest',
                             bet_conf=bet_conf)

            mon_conf = cs.get_monitoring_conf(obs_time)
            # In case want dump monitoring conf
            # with open('examples/monitoring_dump.json', 'w') as mon_file:
            #     mon_file.write(mon_conf.to_json(indent=4 * ' '))

        bet_conf.load_vent_grid()
        bet_conf.load_style_grid()

        ef_model = bet.run.bet_ef.BetEFModel(bet_conf, mon_conf)
        ef_model.run(monitoring=False)

        bet_conf.load_hazard_grid()
        bet_conf.load_tephra_grid()

        t_result = bet.run.tephra.get_tephra_data(obs_time, bet_conf)

        vh_model = BetVHModel(
            bet_conf=bet_conf,
            bet_ef_result=ef_model.result,
            tephra_result=t_result,
            tmpdir=run_dir)

        vh_model.load_mapping()
        vh_model.run()
        sizes_prior = [float(x) for x in bet_conf.BET['Tephra']['prior']]
        sizes = bet_conf.BET['Styles']['sizes']

        tephra_samples = [np.zeros((vh_model.result.n_samples))
                          if (int(sizes_prior[i_size]) == 0)
                          else np.ones((vh_model.result.n_samples))
                          for i_size in range(len(sizes))]

        bet.messaging.celery.tasks.run_sample_per_arealist(
                        range(800, 820, 1),
                        bet_conf,
                        ef_model.result,
                        t_result,
                        tephra_samples,
                        vh_model.result,
                        sampling=sampling)
