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

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
import datetime
from sqlalchemy import Column, Integer, String, DateTime, Binary, ForeignKey, Float


__author__ = 'Marco'

BaseTest = declarative_base()


class SimulatedObservation1(BaseTest):
    __tablename__ = 'simulatedObservation1'
    _idSimulatedObservation = Column('idsimulatedObservation', Integer, primary_key=True)
    _timeStamp = Column('timeStamp', DateTime, default=datetime.datetime.utcnow, nullable=False)
    _value = Column('value', Float, nullable=True)
    _xLocation = Column('xlocation', Float, nullable=True)
    _yLocation = Column('ylocation', Float, nullable=True)

    def __repr__(self):
        return "<SimulatedObservation1(_timeStamp='%s', _value='%.8f', _xlocation='%.4f', _ylocation='%.4f')>" % (
            self._timeStamp.strftime("%Y-%m-%d %H:%M:%S"), self._value, self._xLocation, self._yLocation)

    @hybrid_property
    def timeStamp(self):
        return self._timeStamp

    @timeStamp.setter
    def timeStamp(self, timeStamp):
        self._timeStamp = timeStamp

    @hybrid_property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @hybrid_property
    def xLocation(self):
        return self._xLocation

    @xLocation.setter
    def xLocation(self, xLocation):
        self._xLocation = xLocation

    @hybrid_property
    def yLocation(self):
        return self._yLocation

    @yLocation.setter
    def yLocation(self, yLocation):
        self._yLocation = yLocation


class SimulatedObservation2(BaseTest):
    __tablename__ = 'simulatedObservation2'
    _idSimulatedObservation = Column('idsimulatedObservation', Integer, primary_key=True)
    _timeStamp = Column('timeStamp', DateTime, default=datetime.datetime.utcnow, nullable=False)
    _value = Column('value', Integer, nullable=True)

    def __repr__(self):
        return "<SimulatedObservation2(_timeStamp='%s', _value='%d')>" % (
            self._timeStamp.strftime("%Y-%m-%d %H:%M:%S"), self._value)

    @hybrid_property
    def timeStamp(self):
        return self._timeStamp

    @timeStamp.setter
    def timeStamp(self, timeStamp):
        self._timeStamp = timeStamp

    @hybrid_property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
