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


__author__ = 'Marco'


class AbstractEFModel(object):
    def __init__(self):
        self._input_data = None
        self._conf_file = None
        # mapModel object. It can be useful to give an interpretation to
        # _geograpicDecomposition. mapModel.gegraphicSamples indces can be
        # used to access (by index) to _geograpicDecomposition list,
        # and have spatial informations about the model.
        # It can happen to have a lazyinitializationexception when accessing
        # particular properties of the object. This happen because the
        # communicatio nwith the DB is closed. To access that data you must
        # open another session.
        self._map_model = None
        self._output = None

    def configure(self, conf_file=None, input_data=None, map_model=None):
        self._conf_file = conf_file
        self._input_data = input_data
        self._map_model = map_model

    def run(self):
        raise NotImplementedError("This class is to be considered "
                                  "not-instantiable.")

    @property
    def output(self):
        return self._output

    @property
    def map_model(self):
        return self._map_model

    @map_model.setter
    def map_model(self, map_model):
        self._map_model = map_model


class InputParameter(object):
    def __init__(self):
        self._description = None  # string
        self._value = None  # float
        self._threshold1 = None  # float
        self._threshold2 = None  # float
        self._relation = None  # string
        self._weight = None  # int
        self._nodeName = None  # string
        self._nodeIndex = None  # int
        self._geographicDecomposition = None  # list of float values.
        # To give a correct interpretation of the map structure used to
        # calculate this list you should use mapMopdel
