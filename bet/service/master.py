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

import sys
from celery import Celery
# from bet.messaging.celery import tasks
# from bet.run import DemoModel
#
# from celery.events import EventReceiver
# from kombu import Connection as BrokerConnection


# def my_monitor():
#     connection = BrokerConnection('amqp://')
#
#     def on_event(event):
#         print "EVENT HAPPENED: ", event
#
#     def on_task_failed(event):
#         exception = event['exception']
#         print "TASK FAILED!", event, " EXCEPTION: ", exception
#
#     print "dentro al monitor"
#     while True:
#         try:
#             with connection as conn:
#                 recv = EventReceiver(conn,
#                                      handlers={'task-failed': on_task_failed,
#                                                'task-succeeded': on_event,
#                                                'task-sent': on_event,
#                                                'task-received': on_event,
#                                                'task-revoked': on_event,
#                                                'task-started': on_event,
#                                                # OR: '*' : on_event
#                                                })
#                 recv.capture(limit=None, timeout=None)
#         except (KeyboardInterrupt, SystemExit):
#             print "EXCEPTION KEYBOARD INTERRUPT"
#             sys.exit()


def my_monitor(app):
    state = app.events.State()

    def on_event(event):
        print "EVENT HAPPENED: ", event

    def announce_failed_tasks(event):
        state.event(event)
        # task name is sent only with -received event, and state
        # will keep track of this for us.
        task = state.tasks.get(event['uuid'])

        print('TASK FAILED: %s[%s] %s' % (
            task.name, task.uuid, task.info(), ))
    try:
        with app.connection() as connection:
            recv = app.events.Receiver(connection, handlers={
                    'task-failed': announce_failed_tasks,
                    'task-succeeded': on_event,
                    'task-sent': on_event,
                    'task-received': on_event,
                    'task-revoked': on_event,
                    'task-started': on_event,
                    # '*': on_event
            })
            recv.capture(limit=None, timeout=None, wakeup=True)
    except (KeyboardInterrupt, SystemExit):
            print "EXCEPTION KEYBOARD INTERRUPT"
            sys.exit()


# print "before model"
# model = DemoModel()
#
# # res = tasks.add.delay(4,4)
# res = tasks.model.delay(model, ['uno', 2, 'tre'])
# print "after submit"
# # res = tasks.external.delay("/hades")

if __name__ == '__main__':
    app = Celery(broker='amqp://')
    my_monitor(app)
