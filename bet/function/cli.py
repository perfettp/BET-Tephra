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

import argparse


def opts_parser():
    parser = argparse.ArgumentParser(description="BET@OV")
    parser.add_argument('-d', '--dev', help='Dev mode', action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dump', help='Dump all data to disk',
                       action='store_true')
    group.add_argument('--load', help='Load all data from disk',
                       required=False, nargs='+')
    group.add_argument('--obs_time', required=False,
                        help='Observation Time to use (YYYYMMDD_HHMMSS)')

    parser.add_argument('-c', '--conf', required=True,
                        help='Configuration file to use')
    parser.add_argument('--run_dir', required=False,
                        help='Run directory to use')
    parser.add_argument('--no_sampling', required=False, action='store_true',
                        help='Skip sampling')
    parser.add_argument('-m', '--monitoring_file', required=False,
                        help='Monitoring parameters dump to load')

    return parser
