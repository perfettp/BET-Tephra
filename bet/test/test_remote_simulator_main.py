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

import random
import datetime

from bet.database import manager
from test_remote_model import BaseTest
from test_remote_model import SimulatedObservation1
from test_remote_model import SimulatedObservation2

__author__ = 'Marco'


def populateRandomNormalized(observationList, nObservations, startTime):
    valueSum = 0
    for i in range(0, nObservations):
        observationList.append(SimulatedObservation1())
        observationList[i].value = random.random()  # Random float x, 0.0 <= x < 1.0
        observationList[i].xLocation = random.random()  # Random float x, 0.0 <= x < 1.0
        observationList[i].yLocation = random.random()  # Random float x, 0.0 <= x < 1.0
        observationList[i].timestamp = startTime + datetime.timedelta(seconds=(90 + random.randint(-30, 30)),
                                                                        milliseconds=random.randint(0, 999))  # i am simulating a not fixed time interval
        valueSum += observationList[i].value
        startTime = observationList[i].timestamp

    # normalization (???)
    #invValueSum = 1 / valueSum
    #for i in range(0, nObservations):
    #    observationList[i].value *= invValueSum

def populateEventCounter(observationList, nObservations, startTime):
    valueSum = 0
    currentTime = datetime.datetime.utcnow() - datetime.timedelta(days=5)
    for i in range(0, nObservations):
        observationList.append(SimulatedObservation2())
        observationList[i].value = int(random.random() * 100)
        observationList[i].timestamp = startTime + datetime.timedelta(minutes=(1000 + random.randint(-100, 100)),
                                                                        milliseconds=random.randint(0, 999))  # i am simulating a not fixed time interval
        valueSum += observationList[i].value
        startTime = observationList[i].timestamp


if __name__ == "__main__":
    # init dbms handler
    dbManager = manager.DbManager(db_type='mysql', db_host='localhost', db_port=3306, db_user='root', db_password='root')

    #init db
    dbName = 'observationsTest'
    dbManager.init_and_use_db(dbName, BaseTest.metadata)

    #populate
    observationList1 = []
    nObservations = 1000
    populateRandomNormalized(observationList1, nObservations, datetime.datetime.utcnow() - datetime.timedelta(days=5))
    print observationList1

    #persist
    with dbManager.session_scope() as session:
        session.add_all(observationList1)

    #populate
    observationList2 = []
    nObservations = 1000
    populateEventCounter(observationList2, nObservations, datetime.datetime.utcnow() - datetime.timedelta(days=5))
    print observationList2

    with dbManager.session_scope() as session:
        session.add_all(observationList2)
