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


from bet.database import manager
# from sqlalchemy.orm import joinedload
from bet.data.orm import Run, RunModel
# from bet.data.orm import Parameter
# from bet.data.orm import Elicitation


class DBService(object):
    def __init__(self, volcano_name='Campi_Flegrei', elicitation=6,
                 mapmodel_name='CardinalModelTest',
                 db_conf=None):
        if not db_conf:
            raise('db_conf needed!')
        self._db_manager = manager.DbManager(**db_conf)

        self._db_manager.use_db(db_conf['db_name'])


    def persist_run(self, timestamp, input_parameters, model_result,  user=None,
                    runmodel_class='TestModel'):
        with self._db_manager.session_scope() as session:
            run_model = session.query(RunModel).\
                filter(RunModel.class_name == runmodel_class).\
                first()
            run_obj = Run()
            run_obj.timestamp = timestamp
            run_obj.input_parameters = input_parameters
            run_obj.output = model_result
            run_obj.user = user
            run_obj.runmodel = run_model
            run_obj.mapmodel = None
            session.commit()
