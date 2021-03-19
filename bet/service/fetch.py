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

import sys
from datetime import datetime
import tempfile
import bet.conf
from sqlalchemy.orm import joinedload
from bet.fetchers.db import DbFetcher
from bet.database import manager
from bet.data.orm import Metaparameter
from bet.data.orm import Parameter
from bet.data.orm import MapModel
from bet.data.orm import Elicitation
from bet.data.orm import Volcano
from bet.data.orm import Base


from bet.data.metaparameter import RemoteMetaparameter
from bet.function.cli import opts_parser
from sys import argv
import numpy as np
import utm

class FetchService(object):

    _supportedDbs = ["mysql", "postgres"]

    def __init__(self, dbManager, volcano, elicitationNumber, mapModelName):
        self._dbManager = dbManager
        self._volcano = volcano
        self._mapModelName = mapModelName
        self._elicitationNumber = elicitationNumber

    # STEP 1: prepare the metaparameter to query remotely. Actually i keep in
    # memory this metaparameter (expunge) in order to avoid to keep local db
    # session open for the whole fetching time (can be long?)
    def search_metaparameter(self, metaparameterId):

        with self._dbManager.session_scope() as session:

            metaparameter = session.query(Metaparameter).join(Elicitation).\
                join(Volcano).\
                options(joinedload('remotesource')).\
                options(joinedload('elicitation')).\
                options(joinedload('elicitation.volcano')).\
                filter(Metaparameter._id == metaparameterId,
                       Elicitation.elicitation_number == elicitationNumber,
                       Volcano.description == volcano).first()

            # Expunge the metaparameter to use it as readonly in the future
            if metaparameter is not None:
                if metaparameter.remotesource is not None:
                    if (self._dbManager.getObjectState(
                            metaparameter.remotesource) == "persistent"):
                        session.expunge(metaparameter.remotesource)
                    if (self._dbManager.getObjectState(
                            metaparameter.elicitation) == "persistent"):
                        session.expunge(metaparameter.elicitation)
                    if (self._dbManager.getObjectState(
                            metaparameter.elicitation.volcano) == "persistent"):
                        session.expunge(metaparameter.elicitation.volcano)
                # Before closing the session i de-attach the item from it so
                # I can use it in future queries (as immutable to avoid
                # problems)
                if (self._dbManager.getObjectState(metaparameter) ==
                        "persistent"):
                    session.expunge(metaparameter)

        return metaparameter

    # STEP 2: prepare the mapmodel to use for data calculation. Actually i
    # keep in memory this mapmodel (expunge) in order to avoid to keep local
    # db session open for the whole fetching time (can be long?)
    def loadMapModel(self):

        with self._dbManager.session_scope() as session:

            # Load the mapModel
            mapModel = session.query(MapModel).\
                options(joinedload('geographic_samples')).\
                filter(MapModel.name == self._mapModelName).first()

            # Expunge the metaparameter to use it as readonly in the future
            if mapModel is not None:
                mapModel.check_integrity()
                if mapModel.geographic_samples is not None:
                    for item in mapModel.geographic_samples:
                        if self._dbManager.getObjectState(item) == "persistent":
                            session.expunge(item)
                if self._dbManager.getObjectState(mapModel) == "persistent":
                    session.expunge(mapModel)

        return mapModel

    def estimateCorrectQueryDateRange(self, metaparameter, parameterIndex):
        return self.estimateCorrectHistoricalQueryDateRange(
            metaparameter,
            parameterIndex,
            datetime.now())

    # STEP 2: given a proposedStartDate will calculate the "real" startDate
    # based on the parameters samplingReference (and samplingDuration),
    # So, given a parameterIndex, it will calculate the reference date for
    # that parameter (es: if refresh for this par is "1Day", the ref date
    # will be 00:00 of the proposed day.
    # If parameterIndex = None, then the ref date is the minimum between all
    # the ref dates of all the parameters linked to current metaparameter
    def estimateCorrectHistoricalQueryDateRange(
            self, metaparameter, parameterId, proposedStartDate):

        startDate = proposedStartDate

        with self._dbManager.session_scope() as session:
            # query the parameter list
            if parameterId is None:
                parameters = session.query(Parameter).join(Metaparameter).\
                    filter(Parameter.metaparameter == metaparameter,
                           Elicitation.elicitation_number ==
                           self._elicitationNumber, Volcano.description ==
                           self._volcano).all()
            else:
                 parameters = session.query(Parameter).join(Metaparameter).\
                    filter(Parameter.metaparameter == metaparameter,
                           Parameter._id == parameterId,
                           Elicitation.elicitation_number ==
                           self._elicitationNumber,
                           Volcano.description == self._volcano).all()

            if parameters is not None and len(parameters) > 0:
                refDate = startDate
                for (i, item) in enumerate(parameters):
                    currRefDate = item.get_currentreferencedate(startDate)
                    if currRefDate < refDate:
                        refDate = currRefDate
            else:
                raise AttributeError("Chosen parameterIndex is not linked "
                                     "with metaparameter")

        return refDate

    # STEP 3: query the remote source asking for a list of values (in a date
    # range) for the current metaparameter. (will init a new connection with
    # a remote source).
    def fetch_values(self, metaparameter, start_date, end_date):

        if isinstance(metaparameter, RemoteMetaparameter):

            fetcher = None
            if metaparameter.remotesource.type in self._supportedDbs:
                fetcher = DbFetcher(metaparameter.remotesource)
            else:
                raise NotImplementedError("Selected remoteSourceType not "
                                          "currently supported.")

            if fetcher is not None:
                data = fetcher.fetch_interval(metaparameter,
                                              start_date, end_date)

        return data

    def search_parameters(self, metaparameter):

        with self._dbManager.session_scope() as session:
            parameters = session.query(Parameter).join(Metaparameter).\
                filter(Parameter.metaparameter == metaparameter,
                       Elicitation.elicitation_number ==
                       self._elicitationNumber,
                           Volcano.description == self._volcano).all()

            #expunge the metaparameter to use it as readonly in the future
            if parameters is not None and len(parameters) > 0:
                for param in parameters:
                    # Expunge parameters.
                    if (self._dbManager.getObjectState(param.metaparameter)
                                == "persistent"):
                        session.expunge(param.metaparameter)
                    if self._dbManager.getObjectState(param) == "persistent":
                        session.expunge(param)
                    # print "Parameter class: %s" % param.polymorphicIdentity
                    # print "Parameter volcano: %s" % param.elicitation.volcano
        return parameters

    def fetch_data_interval(self, metaparameter, start_date, end_date):
        # cut the fractional part and cast again to a datetime object
        run_timestamp = datetime.datetime.strptime(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "%Y-%m-%d %H:%M:%S")
        mapModel = self.loadMapModel()
        referenceDate = self.estimateCorrectHistoricalQueryDateRange(
            metaparameter, None, start_date)
        # data = self.fetch_values(metaparameter, referenceDate, end_date)
        data = metaparameter.fetch_values(end_date)
        return data

    def param_value(self, parameter, run_timestamp, metaparameter_values):

        for parameter in parameters:
            sample_enddate = end_date
            while sample_enddate >= start_date:
                sample_startdate = \
                    parameter.get_currentreferencedate(sample_enddate)
                # create new readElement with temp version
                total, map_values = parameter.sample_value(
                    None, metaparameter_data, sample_enddate)
                sys.exit(0)
                sample_timeserie = data.extractSubTimeSerie(
                    sample_startdate,
                    sample_enddate)

                if (sample_timeserie is not None and
                            sample_timeserie.getLength() > 0):


                    total, map_values = parameter.sample_value(map_model,
                                                               sample_timeserie)
                    if total is not None:
                        read_element = parameter.initReadElement(
                            None,
                            session=session,
                            startTime=sample_startdate,
                            endTime=sample_enddate,
                            value=total,
                            mapModel=map_model,
                            geographicDecomposition=map_values)

                        session_element_rel = ReadSession_Has_ReadElement()
                        session_element_rel.readSession = read_session
                        session_element_rel.readElement = read_element

                # go to previous time sample
                sample_enddate = (sample_startdate -
                                  datetime.timedelta(seconds=1))

        session.add(read_session)


if __name__ == "__main__":

    opts = vars(opts_parser().parse_args(argv[1:]))

    if opts['obs_time']:
        obs_time = datetime.strptime(opts['obs_time'], "%Y%m%d_%H%M%S")
    else:
        obs_time = datetime.now()



    # bet_conf = bet.conf.BetConf(opts['conf'], obs_time=obs_time)
    # bet_conf.merge_local_conf()
    if opts['run_dir']:
            bet_conf = bet.conf.BetConf(opts['conf'], run_dir=opts['run_dir'])
    else:
            bet_conf = bet.conf.BetConf(opts['conf'])


    bet_conf.obs_time = obs_time
    # Parameters in input
    elicitationNumber = '6'
    volcano = 'Campi_Flegrei'
    # mapModel = 'Grid35x20'
    mapModel = 'CardinalModelTest' #can be also none

    # init dbms handler
    ovDbManager = manager.DbManager(db_type=bet_conf.BET['Database']['db_type'],
                                    db_host=bet_conf.BET['Database']['db_host'],
                                    db_port=bet_conf.BET['Database']['db_port'],
                                    db_user=bet_conf.BET['Database']['db_user'],
                                    db_password=bet_conf.BET['Database']['db_pwd'],
                                    debug=False)

    dbName = bet_conf.BET['Database']['db_name']

    ovDbManager.init_and_use_db(dbName, Base.metadata)

    fetchService = FetchService(ovDbManager, volcano, elicitationNumber,
                                mapModel)

    metaparameter = fetchService.search_metaparameter(1)
    parameters = fetchService.search_parameters(metaparameter)
    # metaparameter_values = fetchService.fetch_data_interval(metaparameter,
    #                                                         start_date,
    #                                                         end_date)
    try:
        meta_conf = bet_conf['MetaParameters'][metaparameter.polymorphic_identity]
    except:
        print "No specific MetaParameter configuration for %s" % metaparameter.polymorphic_identity
        meta_conf = None

    metaparameter_values = metaparameter.fetch_values(bet_conf.obs_time, meta_conf)
    fetch_data = dict()
    fetch_map = dict()
    val_tmp = dict()
    val_map_tmp = dict()
    bet_conf.load_vent_grid()
    vent_latlon = np.array([utm.to_latlon(v.point.easting,
                                          v.point.northing,
                                          v.point.zone_number,
                                          v.point.zone_letter)
                            for v in bet_conf.vent_grid])

    for param in parameters:
        val, val_map = param.sample_value(bet_conf.obs_time,
                                          metaparameter_values, vent_latlon)
        val_tmp[param.polymorphic_identity] = val
        val_map_tmp[param.polymorphic_identity] = val_map

    fetch_data[metaparameter.polymorphic_identity] = val_tmp
    fetch_map[metaparameter.polymorphic_identity] = val_map_tmp
    print fetch_data
    print fetch_map
        # fetchService.param_value(metaparameter, None,
        #                          end_date, metaparameter_values)
    # stat = data.filter(metaparameter.RemoteMapping.station == 'RITE')
    # print len(stat.all())
    # for i in stat.all():
    #     print i.value
    # fetchService.runOnHistoricalData(2, 2, start, end)
    # fetchService.runOnHistoricalData(1, 2, start, end)
    # fetchService.runOnLastInterval(3, None)
