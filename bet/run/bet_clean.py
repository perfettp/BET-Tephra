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
import logging
from functools import partial
from shutil import rmtree
from sys import argv
from datetime import datetime
from bet.function import cli, get_logger


from bet.conf import BetConf


def is_old(dir_time, orig_time=None, max_delta=1):
    return abs(dir_time - orig_time).days > max_delta


def clean_dir(dirname, days, filter_old):
    try:
        results = [os.path.join(dirname, d) for d in os.listdir(dirname)]

        old_results = [d for d in results
                if filter_old(datetime.fromtimestamp(os.stat(d).st_mtime))]

        logger.info("Removing {} entries in {}, older than {} days".format(
                len(old_results), dirname, max_days))
        logger.debug("Removing {}".format(old_results))
        for res in old_results:
            if os.path.isdir(res):
                rmtree(res)
            else:
                os.remove(res)
    except OSError as e:
        logger.warn('Cleaning dir {}, error: {}'.format(dirname, e.strerror))


if __name__ == "__main__":

    start_time = datetime.now()
    opts = vars(cli.opts_parser().parse_args(argv[1:]))
    logger = get_logger(level=logging.DEBUG, name="bet_clean")

    logger.info("Execution started at: {}".format(start_time))
    logger.debug("Getting bet_conf")
    bet_conf = BetConf(opts['conf'])

    # to become (also) a cli parameter?
    max_days = int(bet_conf['Apollo']['data_archive_days'])
    # Partial application of is_old for now() and max_days
    is_old_now = partial(is_old, orig_time=datetime.now(), max_delta=max_days)

    results_dir = bet_conf['Apollo']['fall3d']['results_dir']
    data_dir = os.path.join(bet_conf['Apollo']['home_dir'], 'Data')
    grib_dir = os.path.join(data_dir, 'arpa-grib')
    nc_dir = os.path.join(data_dir, 'arpa-nc')

    logger.info("Cleaning Apollo directories")
    clean_dir(results_dir, max_days, is_old_now)
    clean_dir(grib_dir, max_days, is_old_now)
    # clean_dir(nc_dir, max_days, is_old_now)
