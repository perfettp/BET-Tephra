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

# from ov_model import Metaparameter
# from ov_model_map import GridModel
# from ov_model_map import CardinalSectionsModel
# from timeserie import TimeSerie, TimeSerieGeolocalized
from abc import abstractmethod
from collections import namedtuple
from sqlalchemy import Table, Column, Integer, Float, join, \
    ForeignKey, DateTime, String
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy import inspect
from sqlalchemy.orm import mapper
from bet.data.orm import Metaparameter
from bet.fetchers.db import DbFetcher
from bet.data.timeserie import SeismicEvent, TimeSerie, ManualEvent
import datetime

__author__ = 'Marco'

# In this file the developer can adjust the rule to fetch the data for every
# metaparameter.
# Metaparameter = information related to a group of parameters that are
# calculated over the same data coming from the same remote place
# (Es: uplift: uplift cumulativo dei 3 mesi, rateo uplift,  macroscopiche
# variazioni del pattern deformativo...)
# Adjusting the fetch rule for the Uplift metaparameter will affect every
# uplift-based parameter


# To be used only as parent for each remote parameter
class RemoteMetaparameter(Metaparameter):
    pass


# To be used only as parent for each remote metaparameter coming from a DB
class RemoteDbMetaparameter(RemoteMetaparameter):

    Source = namedtuple('Source', ['type', 'host', 'port', 'user',
                                   'password', 'name'])

    data_interval = datetime.timedelta(days=120)

    def fetch_values(self, sample_date, conf=None):

        fetcher = DbFetcher(self._remoteSource)
        if fetcher is not None:
            start_date = sample_date - self.data_interval
            data = fetcher.fetch_interval(self, start_date, sample_date)
            return data
        else:
            return None

    def map_on_remote(self, db_manager):
        raise NotImplementedError("This class is to be considered "
                                  "not-instantiable.")


class MappedMetaparmeter(object):
    @property
    def date(self):
        raise None


class MetaUplift(RemoteDbMetaparameter):

    _metaparameter_identity = 'MetaUplift'
    __mapper_args__ = {
        'polymorphic_identity': _metaparameter_identity
    }

    data_interval = datetime.timedelta(days=120)

    def __init__(self):
        super(MetaUplift, self).__init__()
        self.polymorphicIdentity = self._metaparameter_identity

    # When map_on_remote method is invoked by a DbFetcher object the remote
    # data will be mapped on this class. To avoid properties, it's important
    # that Column objects utile to the rights identifier (see Column key
    # argument)
    class RemoteMapping(MappedMetaparmeter):
        def __repr__(self):
            return "<UpliftSample(date='%s', station='%s', value='%s', " \
                   "lat='%s', lon='%s')>" % (self.date, self.station,
                                             self.value, self.y_location,
                                             self.x_location)

    def map_on_remote(self, db_manager):
        metadata = db_manager.metadata
        data_table = Table("gps_data", metadata,
                           Column("sta_id", Integer,
                                  ForeignKey("gps_sta.sta_id"),
                                  primary_key=True,
                                  key='Id'),
                           Column("date", DateTime,
                                  primary_key=True,
                                  key="date"),
                           Column("Z", Float,
                                  key="value"),
                           Column("N", Float,
                                  key="value_n"),
                           Column("E", Float,
                                  key="value_e"),
                           autoload=True)

        stat_table = Table("gps_sta", metadata,
                           Column('sta_id', Integer, primary_key=True),
                           Column('name', DateTime,
                                  primary_key=True,
                                  key='station'),
                           Column('lat', Float,
                                  key="y_location"),
                           Column('lon', Float,
                                  key="x_location"),
                           autoload=True)

        data_stations_join = join(data_table, stat_table)

        try:
            inspect(self.RemoteMapping)
        except NoInspectionAvailable:
            mapper(self.RemoteMapping, data_stations_join)

    @classmethod
    def get_start_sample(cls, seq):
        try:
            return seq[0]
        except IndexError:
            return None

    @classmethod
    def get_end_sample(cls, seq):
        seq_len = len(seq)
        try:
            return seq[seq_len-1]
        except IndexError:
            return None


class MetaSeismic(RemoteDbMetaparameter):

    _metaparameter_identity = 'MetaSeismic'
    __mapper_args__ = {
        'polymorphic_identity': _metaparameter_identity
    }

    data_interval = datetime.timedelta(days=120)

    def __init__(self):
        super(MetaSeismic, self).__init__()
        self.polymorphicIdentity = self._metaparameter_identity

    class SismolabRemoteMapping(MappedMetaparmeter):
        query = "SELECT s.Origin, s.latitude, s.longitude, s.magnitude, " \
                "s.depth  "\
                "FROM revision1 as s "\
                "WHERE s.origin BETWEEN '{0}'  AND '{1}' AND " \
                "s.region='FLEGREI' " \
                "AND s.magnitude IS NOT NULL "\
                "AND s.magnitude<>-9 AND s.valid=1 ORDER BY s.origin ASC"
        pass

    class WgeovesRemoteMapping(MappedMetaparmeter):
        query = "SELECT s.Data, s.OraHH, s.OraMM, s.OraSS, " \
                "s.Lat_Epicentro, s.Long_Epicentro, s.Magnitudo_ML, "\
                "s.Profondita, s.Tipo_Sisma " \
                "FROM Sismi as s join Tipo_Area as a on s.Area=a.ID "\
                "WHERE s.Data BETWEEN '{0}' AND '{1}' AND " \
                "a.Descrizione='FLEGREI' " \
                "AND s.Magnitudo_Ml IS NOT NULL "\
                "AND s.Magnitudo_Ml<>-9 AND s.Lat_Epicentro<>1 AND " \
                "s.Profondita<>0 ORDER BY s.Data ASC"
        pass

    def fetch_values(self, sample_date, conf=None):
        events = []

        start_date = sample_date - self.data_interval
        wgeoves_fetcher = DbFetcher(
            self.Source(conf['wgeoves']['db_type'],
                        conf['wgeoves']['db_host'],
                        conf['wgeoves']['db_port'],
                        conf['wgeoves']['db_user'],
                        conf['wgeoves']['db_pwd'],
                        conf['wgeoves']['db_name']))

        wgeoves_values = wgeoves_fetcher.execute(
            self.WgeovesRemoteMapping.query.format(
                    start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    sample_date.strftime("%Y-%m-%d %H:%M:%S")))

        for ev in wgeoves_values:
            dt = ev[0].replace(hour=ev[1], minute=ev[2], second=int(ev[3]))
            seis_type = "Unknown"
            if ev[8] == 9:
                seis_type = 'VT'
            elif ev[8] == 3:
                seis_type = 'LP'
            events.append(SeismicEvent(date=dt,
                                       latitude=ev[4],
                                       longitude=ev[5],
                                       magnitude=ev[6],
                                       depth=ev[7],
                                       seis_type=seis_type))

        try:
            last_wgeoves_date = dt
        except NameError:
            last_wgeoves_date = start_date

        sismolab_fetcher = DbFetcher(
            self.Source(conf['sismolab']['db_type'],
                        conf['sismolab']['db_host'],
                        conf['sismolab']['db_port'],
                        conf['sismolab']['db_user'],
                        conf['sismolab']['db_pwd'],
                        conf['sismolab']['db_name']))

        sismolab_values = sismolab_fetcher.execute(
            self.SismolabRemoteMapping.query.format(last_wgeoves_date,
                                                    sample_date))

        for ev in sismolab_values:
            events.append(SeismicEvent(date=ev[0],
                                       latitude=ev[1],
                                       longitude=ev[2],
                                       magnitude=ev[3],
                                       depth=ev[4],
                                       seis_type='VT'))

        return TimeSerie(sorted(events, key=lambda e: e.date))

    def map_on_remote(self, db_manager):
        raise NotImplementedError("This metaparameter cannot be mapped.")


class MetaManual(RemoteDbMetaparameter):

    _metaparameter_identity = 'MetaManual'
    __mapper_args__ = {
        'polymorphic_identity': _metaparameter_identity
    }

    data_interval = datetime.timedelta(days=120)

    def __init__(self):
        super(MetaManual, self).__init__()
        self.polymorphicIdentity = self._metaparameter_identity

    # When map_on_remote method is invoked by a DbFetcher object the remote
    # data will be mapped on this class. To avoid properties, it's important
    # that Column objects utile to the rights identifier (see Column key
    # argument)
    class RemoteMapping(MappedMetaparmeter):
        pass

    def fetch_values(self, sample_date, conf=None):
        events = []

        return TimeSerie([ManualEvent(date=sample_date, value=0)])

    def map_on_remote(self, db_manager):
        raise NotImplementedError("This metaparameter cannot be mapped.")
