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

from sys import argv
from bet.conf import BetConf, MonitoringConf
from bet.run import bet_ef
from bet.run import bet_vh
from bet.run import tephra
import datetime
from bet.service import ConfService
from bet.function.cli import opts_parser


if __name__ == "__main__":

    opts = vars(opts_parser().parse_args(argv[1:]))

    bet_conf = BetConf(opts['conf'])
    # bet_conf.merge_local_conf()

    if opts['monitoring_file']:
        print "Parsing monitoring parameters from file %s" \
              % opts['monitoring_file']
        mon_conf = MonitoringConf()
        with open(opts['monitoring_file'], 'r') as mon_file:
            mon_conf.from_json(mon_file.read())
    else:
        cs = ConfService(volcano_name='Campi_Flegrei', elicitation=6,
                         runmodel_class='TestModel',
                         mapmodel_name='CardinalModelTest',
                         bet_conf=bet_conf)

        mon_conf = cs.get_monitoring_conf(bet_conf.obs_time)
        # In case want dump monitoring conf
        # with open('examples/monitoring_dump.json', 'w') as mon_file:
        #     mon_file.write(mon_conf.to_json(indent=4 * ' '))

    bet_conf.load_vent_grid()
    bet_conf.load_style_grid()

    ef_model = bet_ef.BetEFModel(bet_conf, mon_conf)
    ef_model.run(monitoring=False)
    res = ef_model.result
    bet_conf.load_hazard_grid()
    bet_conf.load_tephra_grid()

    vh_model = bet_vh.BetVHModel(bet_conf=bet_conf,
                                 bet_ef_result=ef_model.result,
                                 tephra_result=tephra.TephraOut())
    vh_model.load_mapping()
    vh_model.run()

    # print res.vent_prob_list
