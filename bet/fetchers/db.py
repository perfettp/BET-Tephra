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

from generic import GenericFetcher
from bet.database import manager
from sqlalchemy.exc import OperationalError

__author__ = 'Marco'


# I can use the same fetcher for n parameters that are derived from same
# data! (es: i can have 3-4 parameters related to uplift data). But for now
# it is ok to build a new DbFetcher everytime (asking for the same number :( )
class DbFetcher(GenericFetcher):
    def __init__(self, source):
        GenericFetcher.__init__(self, source)
        # init dbms handler
        self._dbManager = manager.DbManager(
            db_type=self._source.type,
            db_host=self._source.host,
            db_port=self._source.port,
            db_user=self._source.user,
            db_password=self._source.password)

    # return a single "readElement" object
    def fetch_last(self, remote_parameter):

        elements = self.fetch_interval(remote_parameter, None, None)

        if elements is None or len(elements) == 0:
            return None
        else:
            return elements[0]

    # return query result for data interval
    def fetch_interval(self, remote_parameter, start_date, end_date):

        if remote_parameter.remotesource == self._source:
            # Fetch (will use the loaded object as query source)
            if self._dbManager.dbName != self._source.name:
                self._dbManager.use_db(self._source.name)

                remote_parameter.map_on_remote(self._dbManager)

                with self._dbManager.session_scope() as session:
                    # Short circuit to get query result
                    elements = session.query(
                        remote_parameter.RemoteMapping).filter(
                        remote_parameter.RemoteMapping.date >
                        start_date,
                        remote_parameter.RemoteMapping.date <
                        end_date)
                return elements
        return None

    def execute(self, query):
        if self._dbManager.dbName != self._source.name:
                self._dbManager.use_db(self._source.name)
                try:
                    with self._dbManager.session_scope() as session:
                        # Short circuit to get query result
                        res = session.execute(query)
                        return res.fetchall()
                except OperationalError as e:
                    print("Warning: Operational error on {}".format(self._dbManager))
                    return []

    @property
    def db_manager(self):
        return self._dbManager
