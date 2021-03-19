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

import os
import pickle
from sys import argv
from datetime import datetime
from bet.conf import BetConf
from bet.function.cli import opts_parser
import bet.messaging.celery.tasks


def main(argv):
    opts = vars(opts_parser().parse_args(argv[1:]))
    load_dir = opts['load']

    if opts['obs_time']:
        obs_time = datetime.strptime(opts['obs_time'], "%Y%m%d_%H%M%S")
    else:
        obs_time = datetime.now()

    if load_dir:
        print("Loading data from {0}".format(load_dir))
        with open(os.path.join(load_dir, "bet_conf.pick")) as f:
            bet_conf = pickle.load(f)
    else:
        bet_conf = BetConf(opts['conf'], obs_time=obs_time)

    print "Getting weather data"
    res = bet.messaging.celery.tasks.tephra_get_weather(bet_conf)
    print "tephra_get_weather: {}".format(res)

    bet.messaging.celery.tasks.tephra_run_all.delay(bet_conf, run_bet=False)


if __name__ == "__main__":
    main(argv)
    exit(0)
