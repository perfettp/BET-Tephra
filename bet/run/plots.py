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
import requests
import pickle
from sys import argv
from bet.function.cli import opts_parser
from bet.messaging.celery.tasks import save_vh_out, save_ef_out


if __name__ == "__main__":

    opts = vars(opts_parser().parse_args(argv[1:]))
    load_dir = opts['load']
    sampling = not opts['no_sampling']

    if load_dir:
        print("Loading data from {0}".format(load_dir))
        with open(os.path.join(load_dir, "bet_conf.pick")) as f:
            bet_conf = pickle.load(f)
        with open(os.path.join(load_dir, "mon_conf.pick")) as f:
            mon_conf = pickle.load(f)
        with open(os.path.join(load_dir, "bet_ef_out.pick")) as f:
            bet_ef_out = pickle.load(f)
        with open(os.path.join(load_dir, "tephra_out.pick")) as f:
            tephra_out = pickle.load(f)
        with open(os.path.join(load_dir, "bet_vh_out.pick")) as f:
            bet_vh_out = pickle.load(f)
        print("All data loaded.")

        # save_vh_out(bet_conf, bet_vh_out, run_dir=load_dir)

        save_ef_out(bet_conf, bet_ef_out, run_dir=load_dir)


        # update_data_url = bet_conf['WebInterface']['update_data_url']
        update_data_url = "http://127.0.0.1:5001/update_data"
        print "Notifying %s ... " % update_data_url
        data = {'data_dir': load_dir}

        try:
            requests.post(update_data_url, data=data)
        except Exception as e:
            print("ERROR: {0}".format(e.message))

    else:
        print("--load options needed!")
