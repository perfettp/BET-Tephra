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

import subprocess
import time
import shutil
import datetime
import pickle
import tempfile
import requests
import random
import os
import utm
import numpy as np
import simplejson
from celery.app.task import Task
from celery import group, chord, chain
from bet.function import plot_vents, read_npy_chunk, plot_tephra, \
    plot_vh_prob, get_load_kg, plot_probabilities, export_contours, \
    create_run_dir, get_logger, plot_cumulative
import bet.database.manager
from bet.data.orm import Run
from bet.messaging.celery import app
# from bet.messaging.celery.config import logger
from bet.service import ConfService
from bet.service import DBService
import bet.data
import bet.conf
import bet.run.bet_ef
import bet.run.bet_vh
import bet.run.tephra
import glob
import bet.database.manager

class NotifierTask(Task):
    """Task that sends notification on completion."""
    abstract = True

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        url = 'http://localhost:5000/notify'
        data = { 'result': retval}
        requests.post(url, data=data)


@app.task
def sum_list(int_list):
    print "Received list {0}\n".format(int_list)
    time.sleep(random.randint(10, 30))
    s = sum(int_list)
    print "Sum is: {0}".format(s)
    return s


# Get all confs starting from conf_file
@app.task
def get_bet_conf(bet_conf_file):
    bet_conf = bet.conf.BetConf(bet_conf_file)
    return bet_conf

@app.task
def get_mon_conf(bet_conf, monitoring_file=None):

    if monitoring_file:
        print "Parsing monitoring parameters from file %s" \
              % monitoring_file
        mon_conf = bet.conf.MonitoringConf()
        with open(monitoring_file, 'r') as mon_file:
            mon_conf.from_json(mon_file.read())
    else:
        cs = ConfService(volcano_name='Campi_Flegrei', elicitation=6,
                         runmodel_class='TestModel',
                         mapmodel_name='CardinalModelTest',
                         bet_conf=bet_conf)

        mon_conf = cs.get_monitoring_conf(bet_conf.obs_time)

    return mon_conf

@app.task
def stupid_print(arg):
    print arg

@app.task
def run_ef_model(mon_conf=None, bet_conf=None,  run_dir=None):
    bet_conf.load_vent_grid()
    bet_conf.load_style_grid()
    # Questo deve invocare un service
    ef_model = bet.run.bet_ef.BetEFModel(bet_conf, mon_conf)
    ef_model.run(monitoring=True)
    return ef_model.result


@app.task
def save_ef_out(bet_conf, bet_ef_out, mon_conf=None, run_dir=None):
    if run_dir is None:
        raise ValueError("Run dir not specified")

    with open(os.path.join(run_dir, "bet_ef_out.pick"), 'w') as f:
        pickle.dump(bet_ef_out, f)

    bet_conf.load_vent_grid()
    bet_conf.load_hazard_grid()

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
        print "Connection to db established"
        # Init database objects
    else:
        raise Exception("Impossible to connect db")

    end_time = bet_conf.obs_time
    start_time = end_time - datetime.timedelta(days=30)

    data = dict()
    with ovDbManager.session_scope() as session:
        runs = session.query(Run).filter(Run.timestamp <= end_time,
                                         Run.timestamp >= start_time).\
            order_by(Run.timestamp).all()
        valid_runs = [r for r in runs
                      if (r.output and
                          simplejson.loads(r.output).get('exit_code') == 0)]

        times = np.array([r.timestamp for r in valid_runs])

        data['unrest'] = np.array([r.ef_unrest for r in valid_runs])
        data['magmatic'] = np.array([r.ef_magmatic for r in valid_runs])
        data['eruption'] = np.array([r.ef_eruption for r in valid_runs])

    plot_probabilities(times, data, os.path.join(run_dir, "bet_ef_probs.png"))

    for nk in data.keys():
        samples = getattr(getattr(bet_ef_out, nk), 'samples')
        mean = getattr(getattr(bet_ef_out, nk), 'mean')
        plot_cumulative(nk, mean, samples, os.path.join(run_dir, "bet_ef_" + nk + ".png"))

    points = [utm.to_latlon(v.easting,
                            v.northing,
                            v.zone_number,
                            v.zone_letter)
              for v in bet_conf.hazard_grid]
    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])
    llcrn = (lats.min(), lons.min())
    urcrn = (lats.max(), lons.max())

    v_mean = np.array([v.ave.mean for v in bet_ef_out.vent_prob_list])
    points = [v.point for v in bet_conf.vent_grid]
    plot_vents(points, v_mean, os.path.join(run_dir, "bet_ef.png"),
               title='(conditional to the occurence of an eruption)',
               scatter=False, basemap_res='f')

    return True


@app.task
def run_vh_model(bet_conf, bet_ef_out, tephra_out, exp_window, run_dir):
    bet_conf.load_vent_grid()
    bet_conf.load_style_grid()
    bet_conf.load_hazard_grid()
    bet_conf.load_tephra_grid()
    vh_model = bet.run.bet_vh.BetVHModel(bet_conf, bet_ef_out, tephra_out,
                                         exp_window, tmpdir=run_dir)
    vh_model.run()
    print "BetVHModel main terminated"
    return vh_model.result


@app.task
def run_sample_per_arealist(area_interval,
                            bet_conf,
                            bet_ef_out,
                            tephra_out,
                            tephra_samples,
                            bet_vh_out,
                            sampling=True):

    print("run_sample_per_arealist: {0}, {1}".format(area_interval,
                                                     len(area_interval)))
    hc_samples = list()

    thresholds_n = len(bet_conf.BET['Hazard']['load_thresholds'])
    percentiles = [float(x) for x in bet_conf.BET['Hazard']['percentiles']]
    sizes = bet_conf.BET['Styles']['sizes']
    sizes_prior = [float(x) for x in bet_conf.BET['Tephra']['prior']]

    alpha78_st_chunk = read_npy_chunk(bet_vh_out.alpha78_st_path,
                                      area_interval[0], len(area_interval))
    alpha78_lt_chunk = read_npy_chunk(bet_vh_out.alpha78_lt_path,
                                      area_interval[0], len(area_interval))
    beta78_st_chunk = read_npy_chunk(bet_vh_out.beta78_st_path,
                                     area_interval[0], len(area_interval))
    beta78_lt_chunk = read_npy_chunk(bet_vh_out.beta78_lt_path,
                                     area_interval[0], len(area_interval))

    # Deactivating warning for invalid value
    old_settings = np.geterr()
    np.seterr(invalid='ignore')

    # Loop over area_interval
    for area_index in range(len(area_interval)):
        area_id = area_interval[area_index]
        hcpp = bet.run.bet_vh.sample_per_area_cfg(
                bet.run.bet_vh.BetVHSamplesCfg(
                        area_id,
                        bet_vh_out.n_samples,
                        bet_vh_out.n_samples_st,
                        bet_vh_out.n_samples_lt,
                        thresholds_n,
                        percentiles,
                        sizes,
                        sizes_prior,
                        bet_vh_out.n78_weight,
                        bet_vh_out.tmp_dir,
                        bet_vh_out.day_prob,
                        np.squeeze(alpha78_st_chunk[area_index]),
                        np.squeeze(alpha78_lt_chunk[area_index]),
                        np.squeeze(beta78_st_chunk[area_index]),
                        np.squeeze(beta78_lt_chunk[area_index]),
                        ),
                bet_ef_out,
                tephra_out,
                tephra_samples,
                sampling=sampling)
        hc_samples.append(hcpp)

    np.seterr(**old_settings)
    print "Task: all areas done"
    return hc_samples


@app.task
def save_vh_out(bet_conf, bet_vh_out, run_dir):
    bet_conf.load_hazard_grid()
    abs_perc_to_plot = float(bet_conf.BET['Hazard']['abs_perc_to_plot'])
    cond_perc_to_plot = float(bet_conf.BET['Hazard']['cond_perc_to_plot'])
    load_thresholds = np.array([float(t)
                                for t in bet_conf.BET['Hazard']
                                ['load_thresholds']])
    abs_p = np.array([pp.abs_mean for pp in bet_vh_out.hc])
    cond_p = np.array([pp.con_mean for pp in bet_vh_out.hc])

    hc_abs_p = np.array([get_load_kg(point_data,
                                     load_thresholds,
                                     abs_perc_to_plot)
                         for point_data in abs_p])

    hc_cond_p = np.array([get_load_kg(point_data,
                                      load_thresholds,
                                      cond_perc_to_plot)
                          for point_data in cond_p])

    plot_vh_prob(bet_conf.hazard_grid, hc_abs_p,
                 os.path.join(run_dir, "bet_vh_abs.png"), title="Absolute "
                                                                "probability")

    plot_vh_prob(bet_conf.hazard_grid, hc_cond_p,
                 os.path.join(run_dir, "bet_vh_cond.png"),
                 title="Probability conditional to the occurrence of an "
                       "eruption")

    with open(os.path.join(run_dir, "bet_vh_out.pick"), 'w') as f:
            pickle.dump(bet_vh_out, f)

    export_contours(bet_conf.hazard_grid,
                    hc_cond_p,
                    [10, 100, 300, 500, 1000],
                    bet_conf,
                    basename="vh_cond_p",
                    rundir=run_dir,
                    plot=True)

    export_contours(bet_conf.hazard_grid,
                    hc_abs_p,
                    [10, 100, 300, 500, 1000],
                    bet_conf,
                    basename="vh_abs_p",
                    rundir=run_dir,
                    plot=True)


    return True


@app.task(bind=True)
def run_all(self, mconf):
    """Background task that runs a long function with progress reports."""

    print "Inside run_all"
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = random.randint(10, 50)
    time.sleep(10)
    for i in range(total):
        if not message or random.random() < 0.25:
            message = '{0} {1} {2}...'.format(random.choice(verb),
                                              random.choice(adjective),
                                              random.choice(noun))
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': total,
                                'status': message})
        time.sleep(1)
    return {'current': 100, 'total': 1}


@app.task
def tephra_get_output(exp_window, bet_conf, run_dir):
    print "Searching tephra output in interval {} - {} ".format(
        exp_window['begin'].strftime("%Y%m%d%H"),
        exp_window['end'].strftime("%Y%m%d%H")
    )
    tephra_out = bet.run.tephra.get_tephra_data(exp_window, bet_conf)
    print "Fetched TephraOut"

    bet_conf.load_tephra_grid()
    print "Loaded tephra grid"

    try:
        bet.run.tephra.copy_gmt_imgs(tephra_out, run_dir)
        print "GMT images copied"
    except Exception as e:
        print "WARNING: error copying gmt imgs {}".format(e.message)

    try:
        bet.run.tephra.all_images(bet_conf, tephra_out, run_dir)
        print("All images created")
    except Exception as e:
        print "WARNING: error creating images: {}".format(e.message)


    return tephra_out


@app.task
def tephra_models(bet_conf):
    env = os.environ.copy()
    env['PATH'] = bet_conf['Apollo']['bin_dir'] + ":" + \
                  bet_conf['Apollo']['scripts_dir'] + ":" + env['PATH']
    env['HOME'] = bet_conf['Apollo']['home_dir'] + '/'
    if bet_conf.obs_time:
        n = bet_conf.obs_time
    else:
        n = datetime.datetime.now()

    hour = "00" if n.hour < 12 else "12"
    models_out = dict()
    for model in bet_conf['Apollo']['models']:
        ext_args = [
            bet_conf['Apollo'][model]['cron_script'],
            bet_conf['Apollo'][model]['run_script'],
            n.strftime("%Y"),
            n.strftime("%m"),
            n.strftime("%d"),
            hour]
        try:
            print ' '.join(ext_args)
            exit_code = subprocess.call(ext_args, env=env)
        except OSError as e:
            print "OSError: %s" % e.strerror
            exit_code = -1
        models_out[model] = exit_code
    return models_out


@app.task
def tephra_get_weather(bet_conf):
    env = os.environ.copy()
    env['PATH'] = bet_conf['Apollo']['bin_dir'] + ":" + \
                  bet_conf['Apollo']['scripts_dir'] + ":" + env['PATH']
    env['HOME'] = bet_conf['Apollo']['home_dir'] + '/'
    if bet_conf.obs_time:
        n = bet_conf.obs_time
    else:
        n = datetime.datetime.now()

    hour = "00" if n.hour < 12 else "12"

    exit_code = 0
    ext_args = [
        bet_conf['Apollo']['weather_script'],
        n.strftime("%Y"),
        n.strftime("%m"),
        n.strftime("%d"),
        hour]
    try:
        print ' '.join(ext_args)
        exit_code = subprocess.call(ext_args, env=env)
    except OSError as e:
        print "OSError: %s" % e.strerror
        exit_code = -1
    print "{} exited: {}".format(
            bet_conf['Apollo']['weather_script'],
            exit_code)
    return exit_code


@app.task(ignore_result=True)
def tephra_run_all(bet_conf, run_bet=False):
    env = os.environ.copy()
    env['PATH'] = bet_conf['Apollo']['bin_dir'] + ":" + \
                  bet_conf['Apollo']['scripts_dir'] + ":" + env['PATH']
    env['HOME'] = bet_conf['Apollo']['home_dir'] + '/'
    if bet_conf.obs_time:
        n = bet_conf.obs_time
    else:
        n = datetime.datetime.now()

    hour = "00" if n.hour < 12 else "12"

    for model in bet_conf['Apollo']['models']:
        with open(bet_conf['Apollo'][model]['run_script']) as f:
            print "Preparing to launch simulations"
            if run_bet:
                print "Executing Tephra scenario simulations and BET suite"
                sims = chord(group(
                        tephra_run_one.s(
                                bet_conf,
                                bet_conf['Apollo'][model]['bin_script'],
                                n.strftime("%Y"),
                                n.strftime("%m"),
                                n.strftime("%d"),
                                hour,
                                line.strip("\n ").split(" ")
                        ) for line in f),
                    bet_suite.s(bet_conf))
            else:
                print "Executing just Tephra scenario simulations"
                sims = group(
                    tephra_run_one.s(
                            bet_conf,
                            bet_conf['Apollo'][model]['bin_script'],
                            bet_conf['Apollo'][model]['fall3d2gmt_script'],
                            n.strftime("%Y"),
                            n.strftime("%m"),
                            n.strftime("%d"),
                            hour,
                            line.strip("\n ").split(" ")
                    ) for line in f)
            sims()


@app.task
def tephra_run_one(bet_conf, binfile, convert_script,  year, month, day, hour,
                   scenario):
    env = os.environ.copy()
    env['PATH'] = bet_conf['Apollo']['bin_dir'] + ":" + \
                  bet_conf['Apollo']['scripts_dir'] + ":" + \
                  bet_conf['Apollo']['gmt_dir'] + ":" + env['PATH']
    env['HOME'] = bet_conf['Apollo']['home_dir'] + '/'
    ext_args = [
        binfile,
        scenario[0],
        scenario[1],
        year,
        month,
        day,
        hour]
    try:
        print "Begin {}".format(' '.join(ext_args))
        exit_code = subprocess.call(ext_args, env=env)
    except OSError as e:
        print "OSError: %s" % e.strerror
        exit_code = -1
    print "Returning {}".format(exit_code)
    if exit_code == 0:
        problem_name = bet.run.tephra.tephra_problem_name(scenario, year,
                                                          month, day, hour)
        ext_args = [convert_script, problem_name]
        try:
            print "Begin {}".format(' '.join(ext_args))
            conv_exit_code = subprocess.call(ext_args, env=env)
        except OSError as e:
            print "OSError: %s" % e.strerror
            conv_exit_code = -1
    return (ext_args, exit_code)


@app.task
def add(x, y):
    print x
    return x + y

@app.task
def tsum(numbers):
    print "TSUMTSUMTSUM"
    return sum(numbers)


@app.task
def bet_suite(res_list, bet_conf):
    print "Starting bet_suite task"

    if res_list is not None:
        print "Received res_list: {}".format(res_list)
        if all(res[1] == 0 for res in res_list):
            print "All scenario simulations were successfull!"
        else:
            for res in res_list:
                print "Scenario {} failed with exit code {}".format(res[0], res[1])
    else:
        print "Called with res_list == None"
        run_dir = create_run_dir(bet_conf.BET['data_dir'], bet_conf.obs_time)

        if run_dir is None:
            print "Cannot create run_dir, exiting"
            exit(-1)
        print "Run dir: {}".format(run_dir)

        print "Defining ef_chain"
        ef_chain = chain(get_mon_conf.s(bet_conf),
                         run_ef_model.s(bet_conf=bet_conf,
                                        run_dir=run_dir))
        # print "type ef_chain: {}".format(type(ef_chain))
        ef_chain()
        print "Launched ef_chain"


    print "Go with the suite!"
    return True

@app.task
def fetch_parameter_conf(bet_conf, start_time_str, end_time_str):

    start_time = datetime.datetime.strptime(start_time_str,
                                            "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time_str,
                                          "%Y-%m-%d %H:%M:%S")

    cs = ConfService(volcano_name='Campi_Flegrei', elicitation=6,
                     runmodel_class='TestModel',
                     mapmodel_name='CardinalModelTest',
                     bet_conf=bet_conf)

    monitoring_conf = cs.get_monitoring_conf(end_time)
    return monitoring_conf.to_json()


@app.task
def prepare_web_data(bet_conf=None):
    print "Prepare web data"
    if os.path.isdir(bet_conf.run_dir):
        return True
    else:
        return False
