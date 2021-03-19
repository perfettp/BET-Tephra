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
  Copyright(C) 2015 Marco Cincini

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

from metaparameter import MetaSeismic
from metaparameter import MetaUplift
from metaparameter import MetaManual
from parameter import UpliftRITEThreeMonths
from parameter import UpliftRITEMonthlyRatio
from parameter import SeismicCount, SeismicVTDailyRatio, \
    SeismicDeepVTDailyRatio,SeismicDeepLPDailyRatio, \
    SeismicVTMaxMagnitude, SeismicAllLPMonthlyRatio, SeismicLPDailyRatio, \
    SeismicVLPULPDailyRatio, SeismicDeepVLPULPMonthlyRatio, \
    ManualDeepMonthlyTremor, ManualMonthlyTremor, ManualAcidGasPresence, \
    ManualDegas, UpliftAbnormalStationsRatio, UpliftAbnormalVhorRatio, \
    UpliftMaxSpeedStation, ManualMagmaticVariation, \
    ManualGasCompositionVariation, ManualNewFractures, \
    ManualNewIdroThermalSources, ManualPhreaticActivity, ManualRSAMAcc, \
    ManualSeismicEnergyAcc, ManualSeismicStop, ManualSeismicAcc, UpliftRITEDailyRatio

from map import CardinalSectionsModel
from map import GridModel
import utm
import numpy as np
import json



class HCProbPoint(object):
    def __init__(self, abs_mean=np.array(list()), con_mean=np.array(list()),
                 samples_abs_mean=np.array(list()),
                 samples_abs_perc =np.array(list()),
                 samples_con_mean=np.array(list()),
                 samples_con_perc =np.array(list())):

        self._abs_mean = abs_mean
        self._con_mean = con_mean
        self._samples_abs_mean = samples_abs_mean
        self._samples_abs_perc = samples_abs_perc
        self._samples_con_mean = samples_con_mean
        self._samples_con_perc = samples_con_perc

    # def __dict__(self):
    #     return {'abs_mean': self._abs_mean.tolist(),
    #             'con_mean': self._con_mean.tolist(),
    #             'samples_abs_mean': self._samples_abs_mean.tolist(),
    #             'samples_abs_perc': self._samples_abs_perc.tolist(),
    #             'samples_con_mean': self._samples_con_mean.tolist(),
    #             'samples_con_perc': self._samples_con_perc.tolist()}

    @classmethod
    def from_dict(cls, d):
        hcpp = cls(abs_mean=np.array(d['abs_mean']),
                   con_mean=np.array(d['con_mean']),
                   samples_abs_mean=np.array(d['samples_abs_mean']),
                   samples_abs_perc=np.array(d['samples_abs_perc']),
                   samples_con_mean=np.array(d['samples_con_mean']),
                   samples_con_perc=np.array(d['samples_con_perc']))
        return hcpp

    @classmethod
    def from_json(cls, c_ser, **kwargs):
        return cls.from_dict(json.loads(c_ser, **kwargs))

    def to_json(self, **kwargs):
        return json.dumps(self.__dict__(),
                          ensure_ascii=False,
                          sort_keys=True,
                          **kwargs)



    @property
    def abs_mean(self):
        return self._abs_mean

    @property
    def con_mean(self):
        return self._con_mean

    @property
    def samples_abs_mean(self):
        return self._samples_abs_mean

    @samples_abs_mean.setter
    def samples_abs_mean(self, val):
        self._samples_abs_mean = val

    @property
    def samples_abs_perc(self):
        return self._samples_abs_perc

    @samples_abs_perc.setter
    def samples_abs_perc(self, val):
        self._samples_abs_perc = val

    @property
    def samples_con_mean(self):
        return self._samples_con_mean

    @samples_con_mean.setter
    def samples_con_mean(self, val):
        self._samples_con_mean= val

    @property
    def samples_con_perc(self):
        return self._samples_con_perc

    @samples_con_perc.setter
    def samples_con_perc(self, val):
        self._samples_con_perc = val


class UTMPoint(object):

    def __init__(self, easting, northing, zone_number=33, zone_letter='N'):
        self._easting = easting
        self._northing = northing
        self._zone_number = zone_number
        self._zone_letter = zone_letter

    def to_latlon(self):
        return utm.to_latlon(self._easting, self._northing,
                             self._zone_number, self._zone_letter)

    # def __dict__(self):
    #     return dict(easting=self._easting,
    #                 northing=self._northing,
    #                 zone_number=self._zone_number,
    #                 zone_letter=self._zone_letter)

    @classmethod
    def from_dict(cls, d):
        up = cls(d['easting'], d['northing'],
                 zone_number = d['zone_number'],
                 zone_letter = d['zone_letter'])
        return up

    @property
    def easting(self):
        return self._easting

    @property
    def northing(self):
        return self._northing

    @property
    def zone_number(self):
        return self._zone_number

    @property
    def zone_letter(self):
        return self._zone_letter

    def __repr__(self):
        r = "<UTMPoint: easting=%s, northing=%s, " \
            "zone_number=%s, zone_letter=%s >"  \
            % (self._easting, self._northing,
               self._zone_number, self._zone_letter)
        return r


class SamplingAve(object):

    def __init__(self, mean=float('nan'), samples=None):
        self._mean = mean
        self._samples = samples

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, val):
        self._mean = val

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, val):
        self._samples = val
    #
    # def __dict__(self):
    #     return {'mean': self._mean,
    #             'samples': self._samples.tolist()}

    @classmethod
    def from_dict(cls, d):
        sa = cls()
        sa.mean = d['mean']
        sa.samples = np.array(d['samples'])
        return sa


class VentProb(object):

    def __init__(self, loc=None, point=None,
                 prior=None, lambda_val=None, past_data=None):
        """

        :param loc: vent location index
        :type loc: integer

        :return:
        """

        self._loc = loc
        self._point = point
        self._prior = prior
        self._lambda = lambda_val
        self._past_data = past_data
        self._size = None
        self._ave = SamplingAve()
        self._sizes_ave = dict()

    # def __dict__(self):
    #     return {'loc': self._loc,
    #             'point': self._point.__dict__(),
    #             'prior': self._prior,
    #             'lambda': self._lambda,
    #             'past_data': self._past_data,
    #             'size': self._size,
    #             'ave': self._ave.__dict__(),
    #             'sizes_ave': dict((k, self._sizes_ave[k].__dict__())
    #                               for k in self._sizes_ave.keys())}

    @classmethod
    def from_dict(cls, d):
        vp = cls()
        vp.loc = d['loc']
        vp.point = UTMPoint.from_dict(d['point'])
        vp.prior = d['prior']
        vp._lambda = d['lambda']
        vp.past_data = d['past_data']
        vp.size = d['size']
        vp.ave = SamplingAve.from_dict(d['ave'])
        vp.sizes_ave = dict((k,
                             SamplingAve.from_dict(d['sizes_ave'][k]))
                                  for k in d['sizes_ave'])
        return vp

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, val):
        self._point = val

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, val):
        self._loc = val

    @property
    def size(self):
        return self._size

    @property
    def ave(self):
        return self._ave

    @ave.setter
    def ave(self, val):
        self._ave = val

    @property
    def sizes_ave(self):
        return self._sizes_ave

    @sizes_ave.setter
    def sizes_ave(self, val):
        self._sizes_ave = val

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, val):
        self._prior = val

    @property
    def past_data(self):
        return self._past_data

    @past_data.setter
    def past_data(self, val):
        self._past_data = val

    @size.setter
    def size(self, val):
        self._size = val

    def __repr__(self):
        r = "<VentProb: loc=%s, point=%s, prior=%s, past_data=%s, ave=%s>" % \
            (self._loc, self._point, self._prior, self._past_data, self._ave)
        return r


class VentProbList(list):
    def __init__(self, *args):
        list.__init__(self, *args)


class StyleProb(object):
    def __init__(self, sizes=list()):
        self._sizes = sizes
        self._prior = dict((k, list()) for k in self._sizes)
        self._past_data = dict((k, list()) for k in self._sizes)
        self._lambda = list()
        pass

    def set_prior(self, prior_arr):
        self._prior = prior_arr

    def set_past_data(self, past_data_arr):
        self._past_data = past_data_arr

    def set_lambda(self, lambda_arr):
        self._lambda = lambda_arr

    def get_prior(self, size, loc=1):
        return self._prior[size][loc-1]

    def get_past_data(self, size, loc=1):
        return self._past_data[size][loc-1]

    def get_lambda(self, loc=1):
        return self._lambda[loc-1]
