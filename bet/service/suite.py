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
import shutil
import logging
import pickle
from sys import argv
from celery import group
import numpy as np
import requests
import json
from datetime import timedelta, datetime
from bet.data.orm import Run
import bet.messaging.celery.tasks as tasks
from bet.function import chunks, cli, create_run_dir, get_logger, log_to_file
import bet.database
import bet.run.bet_ef
import bet.run.tephra


def get_monconf(bet_conf, run_dir, logger, opts):
    logger.debug("Waiting for mon_conf")
    mon_conf = None
    if opts['monitoring_file']:
        logger.info("Parsing monitoring parameters from file {}".format(
                    opts['monitoring_file']))
        mon_conf_res = tasks.get_mon_conf.delay(
                bet_conf,
                monitoring_file=opts['monitoring_file'])
    else:
        mon_conf_res = tasks.get_mon_conf.delay(bet_conf)

    try:
        mon_conf = mon_conf_res.get()
        logger.info("Received MonConf with nodes {}".format(
            ", ".join([n.name for n in mon_conf.nodes])
        ))
    except Exception as e:
        logger.exception("Error decoding mon_conf: {}".format(e.message))
        exit(1)

    logger.debug("Dumping MonitoringConf to {}".format(
        os.path.join(run_dir, "mon_conf.pick")))
    with open(os.path.join(run_dir, "mon_conf.pick"), 'w') as f:
        pickle.dump(mon_conf, f)

    return mon_conf


def insert_update_run(logger, bet_conf, run_dir=None,
                      mon_conf=None, bet_ef_out=None, output=None):

    ovDbManager = bet.database.manager.DbManager(
            db_type=bet_conf.BET['Database']['db_type'],
            db_host=bet_conf.BET['Database']['db_host'],
            db_port=bet_conf.BET['Database']['db_port'],
            db_user=bet_conf.BET['Database']['db_user'],
            db_name=bet_conf.BET['Database']['db_name'],
            db_password=bet_conf.BET['Database']['db_pwd'],
            debug=False)
    connected = ovDbManager.use_db(bet_conf.BET['Database']['db_name'])

    if connected:
        with ovDbManager.session_scope() as session:
            if bet_conf.run_db_id is None:
                betRun = Run()
                betRun._runmodel_id = 1
                logger.info("Creating new run db entry")
                betRun.rundir = run_dir
                # betRun.input_parameters =   mon_conf ?
                betRun.timestamp = bet_conf.obs_time
                session.add(betRun)
                session.commit()
            else:
                betRun = session.query(Run).filter_by(
                        id=bet_conf.run_db_id).first()
                logger.info("Existing run db entry {}".format(betRun))

            if mon_conf is not None:
                logger.info("Updating input parameters")
                betRun.input_parameters = mon_conf.to_json()

            if bet_ef_out is not None:
                logger.info("Updating bet_ef data")
                betRun.ef_unrest = np.array([
                    bet_ef_out.unrest.mean,
                    np.percentile(bet_ef_out.unrest.samples, 16),
                    np.percentile(bet_ef_out.unrest.samples, 50),
                    np.percentile(bet_ef_out.unrest.samples, 84)])

                magmatic_abs_samples = bet_ef_out.unrest.samples * \
                                       bet_ef_out.magmatic.samples
                magmatic_abs = bet_ef_out.unrest.mean * \
                               bet_ef_out.magmatic.mean
                betRun.ef_magmatic = np.array([
                    magmatic_abs,
                    np.percentile(magmatic_abs_samples, 16),
                    np.percentile(magmatic_abs_samples, 50),
                    np.percentile(magmatic_abs_samples, 84)])

                eruption_abs_samples = magmatic_abs_samples * \
                                       bet_ef_out.eruption.samples
                eruption_abs = magmatic_abs * bet_ef_out.eruption.mean
                betRun.ef_eruption = np.array([
                    eruption_abs,
                    np.percentile(eruption_abs_samples, 16),
                    np.percentile(eruption_abs_samples, 50),
                    np.percentile(eruption_abs_samples, 84)])

                betRun.ef_vent_map = [v.ave.mean
                                      for v in bet_ef_out.vent_prob_list]

            if output is not None:
                logger.info("Updating output")
                # betRun.output = json.dumps(dict(exit_code=0))
                betRun.output = json.dumps(output)
                logger.info("Run with id:{} and rundir:{} updated with "
                            "output:{} ".format(betRun.id,
                                                betRun.rundir,
                                                betRun.output))

            session.merge(betRun)
            session.commit()
            run_id = betRun.id
        logger.debug("Run id: {}".format(run_id))
        return run_id
    else:
        raise(Exception("db error"))


def main(args):
    start_time = datetime.now()
    opts = vars(cli.opts_parser().parse_args(args[1:]))
    print opts
    logger = get_logger(level=logging.DEBUG, name="bet_suite")

    logger.info("Computation started at: {}".format(start_time))
    dev = opts['dev']
    dump = opts['dump']
    load_dir = opts['load'].pop(0) if opts['load'] else None
    sampling = not opts['no_sampling']

    # BetConf
    if load_dir:
        logger.info("Loading data from {}".format(load_dir))
        run_dir = load_dir
        update_list = opts['load']
        logger.info("Suite updating elements {}".format(update_list))

        if any([x in update_list for x in ['conf']]):
            logger.debug("Waiting for bet_conf")
            bet_conf_res = tasks.get_bet_conf.delay(opts['conf'])
            try:
                bet_conf = bet_conf_res.get()
                logger.info("BetConf received")
            except Exception as e:
                logger.exception("Error decoding bet_conf: {}".format(e.message))
                exit(1)
        else:
            try:
                with open(os.path.join(run_dir, "bet_conf.pick")) as f:
                    bet_conf = pickle.load(f)
                obs_time = bet_conf.obs_time
            except IOError as e:
                bet_conf = None
                logger.exception(e)
            else:
                logger.debug("bet_conf loaded from {}".format(os.path.join(
                        run_dir, "bet_conf.pick")))
                log_to_file(run_dir, 'bet_suite.log', level=logging.DEBUG)
                logger.info("Run dir = {}".format(run_dir))
    else:
        if opts['obs_time']:
            obs_time = datetime.strptime(opts['obs_time'], "%Y%m%d_%H%M%S")
        else:
            obs_time = datetime.now()
        logger.debug("Waiting for bet_conf")
        bet_conf_res = tasks.get_bet_conf.delay(opts['conf'])

        bet_conf = None
        try:
            bet_conf = bet_conf_res.get()
            logger.info("BetConf received")
        except Exception as e:
            logger.exception("Error decoding bet_conf: {}".format(e.message))
            exit(1)

        bet_conf.obs_time = obs_time
        logger.info("Observation time = {}".format(bet_conf.obs_time))

        if opts['run_dir']:
            run_dir = opts['run_dir']
        else:
            run_dir = create_run_dir(bet_conf.BET['data_dir'], obs_time)

        if run_dir is None:
           logger.error("Cannot create run_dir, exiting")
           exit(-1)

        log_to_file(run_dir, 'bet_suite.log', level=logging.DEBUG)
        logger.info("Run dir = {}".format(run_dir))

        logger.info("Creating entry in database")
        bet_conf.run_db_id = insert_update_run(logger, bet_conf,
                                               run_dir=run_dir)
        logger.info("Newly inserted run id = {}".format(bet_conf.run_db_id))

        logger.debug("Dumping BetConf to {}".format(
                os.path.join(run_dir, "bet_conf.pick")))
        with open(os.path.join(run_dir, "bet_conf.pick"), 'w') as f:
            pickle.dump(bet_conf, f)

    # Monitoring conf
    if ((load_dir and any([x in update_list for x in ['mon', 'all']])) or
            (not load_dir)):
        mon_conf = get_monconf(bet_conf, run_dir, logger, opts)

        logger.info("Updating mon_conf for database entry {}".format(
                bet_conf.run_db_id))
        insert_update_run(logger, bet_conf, run_dir=run_dir, mon_conf=mon_conf)
    else:
        try:
            with open(os.path.join(run_dir, "mon_conf.pick")) as f:
                mon_conf = pickle.load(f)
        except IOError as e:
            mon_conf = None
            logger.exception(e)
        else:
            logger.debug("mon_conf loaded from {}".format(os.path.join(
                    run_dir, "mon_conf.pick")))


    # BetEF model
    if ((load_dir and any([x in update_list for x in ['ef', 'all']])) or
            (not load_dir)):

        logger.debug("Preparing to run BetEF")
        bet_ef_res = tasks.run_ef_model.delay(bet_conf=bet_conf,
                                              mon_conf=mon_conf,
                                              run_dir=run_dir)
        bet_ef_out = bet_ef_res.get()
        logger.debug("Received BetEFOut")
        logger.info("BetEF Unrest: {:.6f}/{:.6f}/{:.6f}/{:.6f} (ave/16/50/84)".format(
                bet_ef_out.unrest.mean,
                np.percentile(bet_ef_out.unrest.samples, 16),
                np.percentile(bet_ef_out.unrest.samples, 50),
                np.percentile(bet_ef_out.unrest.samples, 84)
        ))

        logger.info("BetEF Magmatic: {:.6f}/{:.6f}/{:.6f}/{:.6f} (ave/16/50/84)".format(
            bet_ef_out.magmatic.mean,
            np.percentile(bet_ef_out.magmatic.samples, 16),
            np.percentile(bet_ef_out.magmatic.samples, 50),
            np.percentile(bet_ef_out.magmatic.samples, 84)
        ))

        logger.info("BetEF Eruption: {:.6f}/{:.6f}/{:.6f}/{:.6f} (ave/16/50/84)".format(
            bet_ef_out.eruption.mean,
            np.percentile(bet_ef_out.eruption.samples, 16),
            np.percentile(bet_ef_out.eruption.samples, 50),
            np.percentile(bet_ef_out.eruption.samples, 84)
        ))

        logger.info("BetEF effective vents: {}".format(bet_ef_out.eff_vents_i))

        logger.info("Updating bet_ef_out for database entry {}".format(
                bet_conf.run_db_id))
        insert_update_run(logger, bet_conf, run_dir=run_dir,
                          bet_ef_out=bet_ef_out)

        tasks.save_ef_out(bet_conf,
                          bet_ef_out,
                          mon_conf=mon_conf,
                          run_dir=run_dir)

    else:
        try:
            with open(os.path.join(run_dir, "bet_ef_out.pick")) as f:
                bet_ef_out = pickle.load(f)
        except IOError as e:
            bet_ef_out = None
            logger.exception(e)
        else:
            logger.debug("bet_ef_out loaded from {}".format(os.path.join(
                    run_dir, "bet_ef_out.pick")))

    offset_list = [int(t) for t in
                   bet_conf.BET['Hazard']['time_offset']]
    exp_time = int(bet_conf.BET['Hazard']['exposure_time'])

    for off_t in offset_list:
        run_off_dir = os.path.join(run_dir, "{:02d}".format(off_t))
        logger.info("Working on offset {}".format(off_t))
        logger.debug("run_off_dir {}".format(run_off_dir))
        if not load_dir:
            os.mkdir(run_off_dir)

        logger.debug("VH time offset: {} , VH exposure time: {}".format(
            off_t, exp_time))

        exp_window = dict()
        exp_window['begin'] = obs_time + timedelta(hours=off_t)
        exp_window['end'] = obs_time + timedelta(hours=off_t) \
                            + timedelta(hours=exp_time)

        # Tephra models
        if ((load_dir and any([x in update_list for x in ['tephra', 'all']]))
            or (not load_dir)):
            logger.info("Searching tephra output in interval {} - {} ".format(
                    exp_window['begin'].strftime("%Y%m%d%H"),
                    exp_window['end'].strftime("%Y%m%d%H")))

            tephra_res = tasks.tephra_get_output.delay(
                    exp_window,
                    bet_conf,
                    run_off_dir)
            tephra_out = tephra_res.get()
            logger.info("Received TephraOut: {}".format(tephra_out))

            logger.debug("Dumping TephraOut to {}".format(
                    os.path.join(run_off_dir, "tephra_out.pick")))
            with open(os.path.join(run_off_dir, "tephra_out.pick"), 'w') as f:
                pickle.dump(tephra_out, f)
        else:
            try:
                with open(os.path.join(run_off_dir, "tephra_out.pick")) as f:
                    tephra_out = pickle.load(f)
            except IOError as e:
                tephra_out = None
                logger.exception(e)
            else:
                logger.debug("tephra_out loaded from {}".format(
                        os.path.join(run_off_dir, "tephra_out.pick")))

        # BetVH
        if ((load_dir and any([x in update_list for x in ['vh', 'all']]))
            or (not load_dir)):

            logger.debug("Running BetVH")
            bet_vh_res = tasks.run_vh_model.delay(bet_conf,
                                                  bet_ef_out,
                                                  tephra_out,
                                                  exp_window,
                                                  run_off_dir)

            bet_vh_out = bet_vh_res.get()

            logger.info("BetVH out: {}".format(bet_vh_out))
            logger.debug("BetVHSamples: total {}, short term {}, long tem {}, "
                         "n78_weight: {}".format(
                    bet_vh_out.n_samples,
                    bet_vh_out.n_samples_st,
                    bet_vh_out.n_samples_lt,
                    bet_vh_out.n78_weight
            ))
            bet_conf.load_hazard_grid()
        else:
            try:
                with open(os.path.join(run_off_dir, "bet_vh_out.pick")) as f:
                    bet_vh_out = pickle.load(f)
            except IOError as e:
                bet_vh_out = None
                logger.exception(e)
            else:
                logger.debug("bet_vh_out loaded from {}".format(
                        os.path.join(run_off_dir, "bet_vh_out.pick")))

        # BetVH areas
        if ((load_dir and any([x in update_list for x in ['vh_areas', 'all']]))
            or (not load_dir)):
            # TODO: get the available worker number
            # chunk_size = bet_conf.hazard_grid_n / 30
            chunk_size = 100

            logger.debug("Total areas: {}, dividing in {} chunks of size {}".format(
                    bet_conf.hazard_grid_n,
                    len(list(chunks(bet_conf.hazard_grid, chunk_size))),
                    chunk_size))
            thresholds_n = len(bet_conf.BET['Hazard']['load_thresholds'])
            percentiles = [float(x) for x in bet_conf.BET['Hazard']['percentiles']]
            sizes = bet_conf.BET['Styles']['sizes']
            sizes_prior = [float(x) for x in bet_conf.BET['Tephra']['prior']]

            tephra_samples = [np.zeros((bet_vh_out.n_samples))
                              if (int(sizes_prior[i_size]) == 0)
                              else np.ones((bet_vh_out.n_samples))
                              for i_size in range(len(sizes))]

            # Divide areas in chunk to for each task
            # IMPORTANT: chunks has to be contigous!! Some code rely on this!

            areas_chunks = list(chunks(range(bet_conf.hazard_grid_n), chunk_size))
            # Build task group to execute in parallel
            task_group = group(
                    [tasks.run_sample_per_arealist.si(
                            area_interval,
                            bet_conf,
                            bet_ef_out,
                            tephra_out,
                            tephra_samples,
                            bet_vh_out,
                            sampling=sampling)
                     for area_interval in areas_chunks])

            task_result = task_group.apply_async()

            logger.debug("Applied all areas tasks, waiting for results")

            all_results = task_result.join_native()

            logger.info("Completed tasks: {0}".format(
                    task_result.completed_count()))

            logger.info("Collected results {0}".format(len(all_results)))

            for chunk_n in range(len(areas_chunks)):
                for i_area in range(len(areas_chunks[chunk_n])):
                    bet_vh_out.hc[areas_chunks[chunk_n][i_area]] = \
                        all_results[chunk_n][i_area]

        # BetVH plots
        if ((load_dir and any([x in update_list for x in ['vh_plots', 'all']]))
            or (not load_dir)):
            # Plotting hazard curve
            plot_res = tasks.save_vh_out.delay(bet_conf, bet_vh_out,
                                               run_dir=run_off_dir)
            plot_out = plot_res.get()

            logger.debug("All data dumped to {0}".format(run_off_dir))

        # Parameters dump and database output
        if ((load_dir and any([x in update_list
                               for x in ['vh', 'vh_areas', 'vh_plots', 'all']]))
            or (not load_dir)):
            logger.info("Updating output for database entry {}".format(
                bet_conf.run_db_id))
            insert_update_run(logger, bet_conf, run_dir=run_dir,
                              output=dict(exit_code=0))
            if not dump:
                logger.info("Removing short term parameters dump files ({}, {})".
                    format(bet_vh_out.alpha78_st_path, bet_vh_out.beta78_st_path))
            try:
                os.remove(bet_vh_out.alpha78_st_path)
                os.remove(bet_vh_out.beta78_st_path)
            except OSError as e:
                logger.warn("Error removing short term parameters dump files")
                logger.warn(e.strerror)

    end_time = datetime.now()
    logger.info("Computation ended at: {}, time elapsed: {}".format(
            end_time,
            end_time - start_time))
    return

if __name__ == "__main__":
    main(argv)
    exit(0)
