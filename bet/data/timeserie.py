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

import datetime

from sqlalchemy import false, true
from orm import MapModel
from collections import MutableSequence

__author__ = 'Marco'


class Event(object):
    def __init__(self, date=None):
        self._date = date

    @property
    def date(self):
        return self._date

    def __repr__(self):
        r = "Event<"
        r += "date=" + str(self._date) + '>'
        return r


class ManualEvent(Event):
    def __init__(self, date=None, value=None):
        super(ManualEvent, self).__init__(date)
        self._value = value

    @property
    def value(self):
        return self._value


class UpliftSample(Event):
    def __init__(self, latitude=None, longitude=None,
                 date=None, start_date=None, uplift=None):
        super(UpliftSample, self).__init__(date)
        self._start_date = start_date
        self._uplift = uplift
        self._latitude = latitude
        self._longitude = longitude

    def __repr__(self):
        r = "UpliftSample<"
        r += "date=" + str(self._date) + ', '
        r += "start_date=" + str(self._start_date) + ', '
        r += "lat=" + str(self._latitude) + ', '
        r += "lon=" + str(self._longitude) + '>'
        return r

    @property
    def uplift(self):
        return self._uplift

    @property
    def start_date(self):
        return self._start_date

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude


class SeismicEvent(Event):
    def __init__(self, date=None, latitude=None, longitude=None,
                 magnitude=None, depth=None, seis_type=None):
        super(SeismicEvent, self).__init__(date)
        self._latitude = latitude
        self._longitude = longitude
        self._magnitude = magnitude
        self._depth = depth
        self._seis_type = seis_type

    def __repr__(self):
        r = "SeismicEvent<"
        r += "date=" + str(self._date) + ', '
        r += "lat=" + str(self._latitude) + ', '
        r += "lon=" + str(self._longitude) + ', '
        r += "mag=" + str(self._magnitude) + ', '
        r += "depth=" + str(self._depth) + ', '
        r += "type=" + str(self._seis_type) + '>'
        return r

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def magnitude(self):
        return self._magnitude

    @property
    def depth(self):
        return self._depth

    @property
    def seis_type(self):
        return self._seis_type


class TimeSerie(MutableSequence):

    def __init__(self, *args, **kwargs):
        self._events = list(*args, **kwargs)

    def __hash__(self):
        return self._events.__hash__()

    def __contains__(self, item):
        return self._events.__contains__(item)

    def __iter__(self):
        return self._events.__iter__()

    def __len__(self):
        return self._events.__len__()

    def __call__(self, *args, **kwargs):
        self._events = list(*args, **kwargs)

    def __getitem__(self, item):
        return self._events.__getitem__(item)

    def __setitem__(self, key, value):
        return self._events.__setitem__(key, value)

    def __delitem__(self, key):
        return self._events.__delitem__(key)

    def insert(self, index, value):
        return self._events.insert(index, value)



class GeoTimeSerie(TimeSerie):
    def __init__(self, *args, **kwargs):
        self._latitude = kwargs.pop('latitude', None)
        self._longitude = kwargs.pop('longitude', None)
        self._station = kwargs.pop('station', None)
        self._events = list(*args, **kwargs)

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def station(self):
        return self._station

# A time serie is a list where each element has two fields: "data" and "time".
# class TimeSerie(object):
#
#     def __init__(self):
#
#         self._data = []
#         self._time = []
#
#     def appendSample(self, value, timeStamp):
#         if isinstance(timeStamp, datetime.datetime):
#             self._data.append(value)
#             self._time.append(timeStamp)
#             return true
#
#         return false
#
#     def getSample(self, index):
#         return (self._data[index], self._time[index])
#
#     def getLength(self):
#         return len(self._time)
#
#     def extractSubTimeSerie(self, startDate, endDate):
#         subTs = TimeSerie()
#         if isinstance(startDate, datetime.datetime) and isinstance(endDate, datetime.datetime):
#             i = 0
#             while i < self.getLength():
#                 if self._time[i] >= startDate and self._time[i] < endDate:
#                     subTs.appendSample(self._data[i], self._time[i])
#                 i += 1
#         return subTs
#
#     def calculateMax(self):
#         res = None
#         if self.getLength() > 0:
#             i = 0
#             while i < self.getLength():
#                 if self._data[i] is not None and (res is None or res < self._data[i]):
#                     res = self._data[i]
#                 i += 1
#
#         return res
#
#     def calculateMin(self):
#         res = None
#         if self.getLength() > 0:
#             i = 0
#             while i < self.getLength():
#                 if self._data[i] is not None and (res is None or res > self._data[i]):
#                     res = self._data[i]
#                 i += 1
#
#         return res
#
#     def calculateSum(self):
#         res = None
#         if self.getLength() > 0:
#             i = 0
#             while i < self.getLength():
#                 if self._data[i] is not None:
#                     if res is None:
#                         res = self._data[i]
#                     else:
#                         res += self._data[i]
#                 i += 1
#
#         return res
#
#     def calculateAvg(self):
#         sum = self.calculateSum()
#         if self.getLength() > 0 and sum is not None:
#             return sum / self.getLength()
#         else:
#             return None
#
#     #useful to say "yes" or "no" to the presence of a value different than 0 in a dataList
#     def verifyPresence(self):
#         res = None
#         if self.getLength() > 0:
#             i = 0
#             while i < self.getLength():
#                 if self._data[i] is not None:
#                     if res is None:
#                         res = False
#                     if self._data[i] != 0:
#                         res = True
#
#                 i += 1
#
#         return res
#
#     def calculateNothing(self):
#         return None
#
# #Inherit from TimeSerie, add a (x,y) reference to each new sample.
# class TimeSerieGeolocalized(TimeSerie):
#
#     def __init__(self):
#         TimeSerie.__init__(self)
#         self._x = []
#         self._y = []
#
#     #values must be a list (also if dimension = 1)
#     def appendSample(self, value, timeStamp, x, y):
#         added = super(TimeSerieGeolocalized, self).appendSample(value, timeStamp)
#         if added == true:
#             self._x.append(x)
#             self._y.append(y)
#
#         return added
#
#     def getSample(self, index):
#         return (self._data[index], self._time[index], self._x[index], self._y[index])
#
#     def extractSubTimeSerie(self, startDate, endDate):
#         subTs = TimeSerieGeolocalized()
#         if isinstance(startDate, datetime.datetime) and isinstance(endDate, datetime.datetime):
#             i = 0
#             while i < self.getLength():
#                 if self._time[i] >= startDate and self._time[i] < endDate:
#                     subTs.appendSample(self._data[i], self._time[i], self._x[i], self._y[i])
#                 i += 1
#         return subTs
#
#     def calculateMaxOnMapModel(self, mapModel):
#         if isinstance(mapModel, MapModel):
#             res = [None] * len(mapModel.geographicSamples) #fill with none
#             if self.getLength() > 0:
#                 i = 0
#                 while i < self.getLength():
#                     index = mapModel.localize_point(self._x[i], self._y[i])  #will localize the current point in the map model
#                     if index is not None:
#                         if self._data[i] is not None and (res[index] is None or res[index] < self._data[i]):
#                             res[index] = self._data[i]
#
#                     i += 1
#
#             return res
#         return None
#
#     def calculateMinOnMapModel(self, mapModel):
#         if isinstance(mapModel, MapModel):
#             res = [None] * len(mapModel.geographicSamples) #fill with none
#             if self.getLength() > 0:
#                 i = 0
#                 while i < self.getLength():
#                     index = mapModel.localize_point(self._x[i], self._y[i])  #will localize the current point in the map model
#                     if index is not None:
#                         if self._data[i] is not None and (res[index] is None or res[index] > self._data[i]):
#                             res[index] = self._data[i]
#
#                     i += 1
#
#             return res
#         return None
#
#     def calculateSumOnMapModel(self, mapModel):
#         if isinstance(mapModel, MapModel):
#             res = [None] * len(mapModel.geographicSamples) #fill with none
#             if self.getLength() > 0:
#                 i = 0
#                 while i < self.getLength():
#                     index = mapModel.localize_point(self._x[i], self._y[i])  #will localize the current point in the map model
#                     if index is not None:
#                         if self._data[i] is not None:
#                             if res[index] is None:
#                                 res[index] = self._data[i]
#                             else:
#                                 res[index] += self._data[i]
#
#                     i += 1
#
#             return res
#         return None
#
#     def calculateSumOnLocation(self, coords):
#         res = None
#         if self.getLength() > 0:
#             i = 0
#             while i < self.getLength():
#                 if self._data[i] is not None and self._x[i] == coords[0] and \
#                         self._y[i] == coords[1]:
#                     if res is None:
#                         res = self._data[i]
#                     else:
#                         res += self._data[i]
#                 i += 1
#         return res
#
#     def calculateAvgOnMapModel(self, mapModel):
#         if isinstance(mapModel, MapModel):
#             res = [None] * len(mapModel.geographicSamples) #fill with none
#             count = [0] * len(mapModel.geographicSamples)
#             if self.getLength() > 0:
#                 i = 0
#                 while i < self.getLength():
#                     index = mapModel.localize_point(self._x[i], self._y[i])  #will localize the current point in the map model
#                     if index is not None:
#                         if self._data[i] is not None:
#                             if res[index] is None:
#                                 res[index] = self._data[i]
#                             else:
#                                 res[index] += self._data[i]
#                             count[index] += 1
#                     i += 1
#
#             for i, item in enumerate(res):
#                 if res[i] is not None:
#                     res[i] = res[i] / count[i]
#
#             return res
#         return None
#
#     def verifyPresenceOnMapModel(self, mapModel):
#         if isinstance(mapModel, MapModel):
#             res = [None] * len(mapModel.geographicSamples) #fill with none
#             if self.getLength() > 0:
#                 i = 0
#                 while i < self.getLength():
#                     index = mapModel.localize_point(self._x[i], self._y[i])  #will localize the current point in the map model
#                     if index is not None:
#                         if self._data[i] is not None:
#                             if res[index] is None:
#                                 res[index] = False
#                             if self._data[i] != 0:
#                                 res[index] = True
#
#                     i += 1
#
#             return res
#         return None
