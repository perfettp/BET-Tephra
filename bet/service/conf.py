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

import logging
from bet.database import manager
from sqlalchemy.orm import joinedload, eagerload
from bet.function.cli import opts_parser
from bet.function import param_anomaly, get_logger
from bet.data.orm import Metaparameter
from bet.data.orm import Parameter
from bet.data.orm import Elicitation, ElicitationNode
from bet.data.orm import Node, NodeParameter
from datetime import datetime
import tempfile
from bet.service import FetchService
from configobj import ConfigObj
from sys import argv
import os.path
import bet.conf
import utm
import numpy as np


class ConfService(object):
    def __init__(self, volcano_name='Campi_Flegrei', elicitation=6,
                 runmodel_class='TestModel', mapmodel_name='CardinalModelTest',
                 bet_conf=None, logger=None):

        self._bet_conf = bet_conf
        db_conf = dict(db_type=bet_conf.BET['Database']['db_type'],
                       db_host=bet_conf.BET['Database']['db_host'],
                       db_port=str(bet_conf.BET['Database']['db_port']),
                       db_user=bet_conf.BET['Database']['db_user'],
                       db_password=bet_conf.BET['Database']['db_pwd'],
                       db_name=bet_conf.BET['Database']['db_name'])

        self._db_manager = manager.DbManager(**db_conf)
        connected = self._db_manager.use_db(db_conf['db_name'])
        if connected:
            self._fetch_service = FetchService(self._db_manager, volcano_name,
                                               elicitation, mapmodel_name)


    def get_monitoring_conf(self, sample_date):

        with self._db_manager.session_scope() as session:
            nodes = session.query(Node).join(NodeParameter).\
                join(Parameter).join(ElicitationNode).join(Elicitation).filter(
                Elicitation._elicitation_number == 6).order_by(
                ElicitationNode.order).all()
            node_parameter = session.query(Node, Parameter, NodeParameter).join(
                NodeParameter).\
                join(Parameter).join(ElicitationNode).join(Elicitation).filter(
                Elicitation._elicitation_number == 6).\
                order_by(ElicitationNode.order).\
                all()
            metaparameters = session.query(Metaparameter).join(
                Parameter).join(NodeParameter).\
                join(Node).join(ElicitationNode).join(Elicitation).filter(
                Elicitation._elicitation_number == 6).\
                options(joinedload(Metaparameter.remotesource)).all()
            session.expunge_all()

        parameters_values = dict()
        parameters_maps = dict()
        for meta in metaparameters:
            try:
                meta_conf = self._bet_conf['MetaParameters'][meta.polymorphic_identity]
            except:
                meta_conf = None
            metaparameter_values = meta.fetch_values(sample_date, meta_conf)
            parameters = self._fetch_service.search_parameters(meta)
            self._bet_conf.load_vent_grid()
            vent_latlon = np.array([utm.to_latlon(v.point.easting,
                                                  v.point.northing,
                                                  v.point.zone_number,
                                                  v.point.zone_letter)
                                    for v in self._bet_conf.vent_grid])
            for param in parameters:
                val, val_map = param.sample_value(
                        sample_date,
                        metaparameter_values,
                        vent_latlon)
                parameters_values[param.parameter_id] = val
                parameters_maps[param.parameter_id] = val_map

        monitoring_conf = bet.conf.MonitoringConf(date=sample_date)
        elicitation_conf = bet.conf.ElicitationConf()
        for node in nodes:
            node_conf = bet.conf.NodeConf(name=node.name)

            params = sorted([p[1:] for p in node_parameter if p[0] == node],
                            key=lambda rel: rel[1].order)

            for p in params:
                if p[1]._node_id == 4:
                    print p
                p_conf = bet.conf.ParameterConf(class_type=p[0].polymorphic_identity,
                                                value=parameters_values[p[
                                                    0].parameter_id],
                                                val_map=parameters_maps[p[
                                                    0].parameter_id],
                                                relation=p[1].relation,
                                                threshold_1=p[1].threshold1,
                                                threshold_2=p[1].threshold2,
                                                weight=p[1].weight,
                                                parameter_family=p[0].parameter_family)
                elicitation_conf.add(p_conf)
                node_conf.append(p_conf)

            monitoring_conf.append(node_conf)
            monitoring_conf.elicitation_conf = elicitation_conf

        return monitoring_conf


if __name__ == "__main__":

    opts = vars(opts_parser().parse_args(argv[1:]))
    logger = get_logger(level=logging.DEBUG, name="bet_suite")

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

    cs = ConfService(volcano_name='Campi_Flegrei', elicitation=6,
                     runmodel_class='TestModel',
                     mapmodel_name='CardinalModelTest',
                     bet_conf=bet_conf)

    logger.info("Getting conf for date for observation time {}".format(
            obs_time.strftime("%H:%M:%S %Y/%m/%d")))

    conf = cs.get_monitoring_conf(obs_time)

    # print(conf)
