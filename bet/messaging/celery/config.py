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

from __future__ import absolute_import
import logging
from kombu import Exchange, Queue
from kombu.common import Broadcast
from bet.conf import BetConf
from celery.schedules import crontab
from bet.function import LOG_FORMAT, LOG_FORMATTER, LOG_LEVEL, \
    get_logger, init_logger

# TODO da sistemare meglio
b_conf = BetConf("etc/bet.cfg")

BROKER_URL = b_conf['Celery']['broker_url']
CELERY_RESULT_BACKEND = b_conf['Celery']['result_backend']
deploy_tag = b_conf['Celery']['deploy_tag']
deploy_site = b_conf['Celery']['deploy_site']


CELERY_IMPORTS = ('bet.data.metaparameter',
                  'bet.data.timeserie',
                  'bet.data',
                  'bet.data.parameter',
                  'bet.data.map',
                  'bet.data.orm',
                  'bet.run.tephra',
                  'bet.run.bet_vh',
                  'bet.run.abstract',
                  'bet.run',
                  'bet.run.test',
                  'bet.run.bet_ef',
                  # 'bet.service.tephra',
                  # 'bet.service.fetch',
                  # 'bet.service.initialization',
                  # 'bet.service.runmodel',
                  # 'bet.service.master',
                  # 'bet.service',
                  # 'bet.service.webviewer',
                  # 'bet.service.conf',
                  # 'bet.service.database',
                  # 'bet.service.suite',
                  'bet.fetchers',
                  'bet.fetchers.db',
                  'bet.fetchers.generic',
                  'bet.messaging',
                  # 'bet.test.test_remote_model',
                  # 'bet.test',
                  # 'bet.test.test_remote_simulator_main',
                  'bet.database.manager',
                  'bet.database',
                  # 'bet.interface.viewer',
                  # 'bet.interface',
                  # 'bet.interface.nav',
                  'bet.conf',
                  'bet.function.cli',
                  'bet.function.alpha_beta',
                  'bet.function',
                  'bet.function.conversions')


CELERY_TASK_SERIALIZER = 'pickle'
CELERY_RESULT_SERIALIZER = 'pickle'
CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack']
CELERY_TIMEZONE = 'Europe/Rome'
CELERY_ENABLE_UTC = True


class TasksRouter(object):
    def route_for_task(self, task, args=None, kwargs=None):
        if (task == 'bet.messaging.celery.tasks.run_vh_model' or
            task == 'bet.messaging.celery.tasks.run_ef_model' or
            task == 'bet.messaging.celery.tasks.run_sample_per_arealist'):
            return {'queue': 'sampling_jobs',
                    'routing_key': deploy_site+".sampling"}
        elif (task == 'bet.messaging.celery.tasks.get_tephra_out' or
              task == 'bet.messaging.celery.tasks.tephra_get_weather' or
              task == 'bet.messaging.celery.tasks.tephra_run_all' or
              task == 'bet.messaging.celery.tasks.tephra_run_one'):
            return {'queue': 'tephra_jobs',
                    'routing_key': deploy_site+".tephra"}
        else:
            return {'queue': 'default', 'routing_key': deploy_site+'.default'}

CELERY_CREATE_MISSING_QUEUES = True

site_exchange = Exchange(deploy_tag, type='direct')

CELERY_QUEUES = (
    Queue('default', site_exchange, routing_key=deploy_site+'.default'),
    Queue('master_jobs', site_exchange, routing_key=deploy_site+'.master'),
    Queue('tephra_jobs', site_exchange, routing_key=deploy_site+'.tephra'),
    Queue('sampling_jobs', site_exchange, routing_key=deploy_site+'.sampling')
)

CELERY_DEFAULT_QUEUE = 'default'

CELERY_DEFAULT_EXCHANGE = deploy_tag
CELERY_DEFAULT_ROUTING_KEY = deploy_site+'.default'

# Disable prefetch
CELERYD_PREFETCH_MULTIPLIER = 1

CELERY_ROUTES = (TasksRouter(), )

# CELERYBEAT_SCHEDULE = {
#     'tephra-twice-a-day': {
#         'task': 'bet.messaging.celery.tasks.tephra_models_decomposed',
#         'schedule': crontab(minute=20, hour='6,18'),
#         'args': (b_conf,)
#     }
# }

#
# logger = logging.getLogger(__name__)
# fh = logging.FileHandler(file, mode='w')
# fh.setLevel(LOG_LEVEL)
# fh.setFormatter(LOG_FORMATTER)
# logger.addHandler(fh)
# logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
