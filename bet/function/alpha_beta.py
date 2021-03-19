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

import numpy as np
from bet.function import get_logger, param_anomaly
import logging
from math import exp, pow

LOG_LEVEL = logging.DEBUG

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(pathname)s:%(" \
             "lineno)d [%("\
             "process)d]: %(message)s"

LOG_FORMATTER = logging.Formatter(LOG_FORMAT)


__logger = None
"""


"""


def w(p1, p2, d):
    return exp(-1 * ((pow((p1.point.easting - p2.point.easting), 2)
                      + pow((p1.point.northing - p2.point.northing), 2))
                     / (2*pow(d, 2))))


def gaussian_filter(vent_grid, vals):
    # print vals
    mat = np.array([np.sum([vals[i_d] * w(vent_grid[i_p], vent_grid[i_d], 500)
                            for i_d in range(len(vent_grid))])
                    for i_p in range(len(vent_grid))])

    # print mat
    return mat


def calc_vent_spatial_ave(parameters, vent_grid):
    p_matrix = np.array([gaussian_filter(vent_grid, p.val_map)
                         for p in parameters
                         if p.val_map is not None and any(param_anomaly(p))])
    if len(p_matrix) > 0:
        return np.mean(p_matrix, axis=0)
    else:
        return np.nan



def calc_monitoring_prob(parameters, sample, nmix):
    param_weight_arr = np.array([p.weight for p in parameters])
    anomaly_degree_arr = calc_anomaly_degree_parameters(parameters)
    anomaly_degree_total = np.sum(anomaly_degree_arr * param_weight_arr)
    #probM, aveM = samplingMonitoring(anomaly_degree_total,
    #                                 sample, nmix, [0.5, 1.0, 0.0, 2.0])
    probM, aveM = sampling_monitoring(anomaly_degree_total,
                                      sample, nmix)
    return probM, aveM


def mixing(postBG, postM, sample, nmix):
    """
    Statistical mixing of long-term and monitoring for binomial
    distribution (yes/no)
    """
    tmp, nbranches = np.shape(postM)
    pp = np.zeros((sample,nbranches))
    for i in range(nbranches):
        sample1 = postM[:,i]
        sample2 = np.array(postBG[i][:sample-nmix])
        pp[:,i] = np.random.permutation(np.concatenate([sample1,sample2]))
    return pp


def sampling_monitoring(anDegTot, sample, nmix):
    """
    :param anDegTot:
    :param sample:
    :param nmix:
    :param pars:
    :return:
#    Sampling the monitored parameter with a given degree of anomaly
    """
    a = 0.9
    b = 1.0 # 0.5
    l = 1.0
    aveM = 1 - a*np.exp(-b*anDegTot)
    alpha = aveM*(l+1)
    beta = (l+1)-alpha
    probM = np.random.dirichlet([alpha,beta], nmix)
    return probM, aveM


def calc_anomaly_degree_parameters(parameters):
    logger = get_logger()
    anomaly_degree = []

    for p in parameters:

        # print "{}: val {}, th1 {}, th2 {}, rel {}".format(
        #         p.name, p.value, p.threshold_1, p.threshold_2, p.relation)
        # TODO: refactor to use param_anomaly
        if p.relation == "=":
            anomaly_degree.append(1.0 if p.value == p.threshold_1 else 0)

        else:
            tmp1 = (p.value-p.threshold_1) / (p.threshold_2-p.threshold_1)
            if p.relation == "<":
                if p.value >= p.threshold_1:
                    anomaly_degree.append(0.0)
                elif p.value <= p.threshold_2:
                    logger.info("Parameter {} full anomaly".format(p.name))
                    anomaly_degree.append(1.0)
                else:
                    logger.warning("Parameter {} anomaly".format(p.name))
                    anomaly_degree.append(0.5*(np.sin(np.pi*tmp1+0.5*np.pi)+1))

            elif p.relation == ">":
                if p.value >= p.threshold_2:
                    logger.warning("Parameter {} full anomaly ".format(p.name))
                    anomaly_degree.append(1.0)
                elif p.value <= p.threshold_1:
                    anomaly_degree.append(0.0)
                else:
                    logger.warning("Parameter {} partial anomaly".format(
                            p.name))
                    anomaly_degree.append(0.5*(np.sin(np.pi*tmp1-0.5*np.pi)+1) )
            else:
                anomaly_degree.append(float('nan'))

    return np.array(anomaly_degree)


def makeAlpha16(n, p, l, pd):
    """
    Calculate alpha values for nodes 1 to 6

    Variables:

    n: n. branches
    a: alpha
    p: prior probability
    l: equivalent number of data
    pd: past data

    """
    a = [0]*n
    a0 = l + n - 1
    for i in range(n):
        a[i] = (p[i] * a0) + pd[i]
    return a


def theoreticalAverage(a):
  """
  """
  if (np.size(a) <= 2):
    ave = a[0]/np.sum(a)
  else:
    ave = a/np.sum(a)

  return ave


def mixingAverage(ave_longterm, ave_monitoring, weight):
  """
  """

  ave_mix = (1-weight)*ave_longterm + weight*ave_monitoring
  return ave_mix
