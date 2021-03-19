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
from sqlalchemy import desc
from flask import Flask
from flask import render_template, jsonify, request, redirect, url_for, \
    send_from_directory
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
from werkzeug.routing import BaseConverter
from bet.data.orm import Run
import bet.database
from bet.interface.nav import nav
from bet.function import param_anomaly, find_offset_dirs
import glob
import json

from bet.messaging.celery import tasks

from flask_wtf import Form
from wtforms import StringField, TextField


class RegexConverter(BaseConverter):
    def __init__(self, url_map, *items):
        super(RegexConverter, self).__init__(url_map)
        self.regex = items[0]


class DemoBootstrap(Flask):
    def __init__(self, bet_conf=None, cur_dir=None,
                 debug=False):
        super(DemoBootstrap, self).__init__(__name__)
        Bootstrap(self)

        self._refresh_interval = 1800  # 30 minutes
        self._cur_dir = os.path.join(self.static_folder, 'values', 'last')
        self.debug = debug
        self.bet_conf = bet_conf
        self.secret_key = "fuffanta"
        self.nav = Nav()
        self.url_map.converters['regex'] = RegexConverter

        self._db_manager = bet.database.manager.DbManager(
                db_type=bet_conf.BET['Database']['db_type'],
                db_host=bet_conf.BET['Database']['db_host'],
                db_port=bet_conf.BET['Database']['db_port'],
                db_user=bet_conf.BET['Database']['db_user'],
                db_name=bet_conf.BET['Database']['db_name'],
                db_password=bet_conf.BET['Database']['db_pwd'],
                debug=False)

        connected = self._db_manager.use_db(bet_conf.BET['Database']['db_name'])
        if connected:
            print "Connection to db established"
            # Init database objects
        else:
            raise Exception("Impossible to connect db")

        self.config['BOOTSTRAP_SERVE_LOCAL'] = True
        self.nav.register_element('frontend_top', Navbar(
                View('BET-Viewer', '.index'),
                # View('Home', '.index'),
                # Subgroup(
                #     'Docs',
                #     Link('Flask-Bootstrap', 'http://pythonhosted.org/Flask-Bootstrap'),
                #     Link('Flask-AppConfig', 'https://github.com/mbr/flask-appconfig'),
                #     Link('Flask-Debug', 'https://github.com/mbr/flask-debug'),
                #     Separator(),
                #     Text('Bootstrap'),
                #     Link('Getting started', 'http://getbootstrap.com/getting-started/'),
                #     Link('CSS', 'http://getbootstrap.com/css/'),
                #     Link('Components', 'http://getbootstrap.com/components/'),
                #     Link('Javascript', 'http://getbootstrap.com/javascript/'),
                #     Link('Customize', 'http://getbootstrap.com/customize/'),
                # ),
                # Text('Using Flask-Bootstrap {}'.format(FLASK_BOOTSTRAP_VERSION)),
        ))
        self.nav.init_app(self)

        def get_link(dir, resource=None):
            if dir is not None:
                if resource is not None:
                    anchor = os.path.join(
                        bet_conf['WebInterface']['web_root'],
                        os.path.basename(dir),
                        resource) + '?' + request.query_string
                else:
                    anchor = os.path.join(
                        bet_conf['WebInterface']['web_root'],
                        os.path.basename(dir)) + '?' + request.query_string
            else:
                anchor = ""
            return anchor

        def get_static_rundirs(run_dir):
            static_run_dir = None
            prev_dir = None
            next_dir = None
            static_run_dir = None
            headers = dict()
            with self._db_manager.session_scope() as session:
                run_list_tmp = session.query(Run).\
                    order_by(desc(Run.timestamp)).all()

                run_list = [run for run in run_list_tmp
                            if (run.output is not None and
                                'exit_code' in json.loads(run.output) and
                                json.loads(run.output)['exit_code'] == 0)]

                date_found = False
                if run_dir == 'last':
                    headers['Refresh'] = str(self._refresh_interval)
                    if len(run_list) > 0:
                        run_entry = run_list[0]
                        date_found = True
                        static_run_dir = run_entry.rundir
                        run_dir = os.path.basename(static_run_dir)
                        if len(run_list) > 1:
                            prev_dir = run_list[1].rundir
                        next_dir = None
                    else:
                        return "No runs in db ", 200
                else:
                    i = 0
                    while i < len(run_list) and not date_found:
                        if os.path.basename(run_list[i].rundir) == run_dir:
                            static_run_dir = run_list[i].rundir
                            if (i + 1) < len(run_list):
                                prev_dir = run_list[i + 1].rundir
                            if (i - 1) >= 0:
                                next_dir = run_list[i - 1].rundir
                            date_found = True
                        else:
                            i += 1
            return static_run_dir, prev_dir, next_dir

        def get_anomalies(static_run_dir):
            parameters_status = dict(date=None, nodes=list())
            try:
                with open(os.path.join(static_run_dir, "mon_conf.pick")) as f:
                    mon_conf = pickle.load(f)

                    parameters_status['date'] = mon_conf.date

                    for node in mon_conf.nodes[0:3]:
                        node_element = dict(name=node.name)
                        node_element['parameters'] = list()
                        for param in node.parameters:
                            part, full = param_anomaly(param)
                            param_elem = dict(name=param.name,
                                              part_anomaly=part,
                                              full_anomaly=full,
                                              relation=param.relation,
                                              threshold_1=param.threshold_1,
                                              threshold_2=param.threshold_2,
                                              value=param.value)
                            node_element['parameters'].append(param_elem)
                        node_element['part_an'] = \
                            len([p for p in node_element['parameters']
                                 if p['part_anomaly']])
                        node_element['full_an'] = \
                            len([p for p in node_element['parameters']
                                 if p['full_anomaly']])
                        parameters_status['nodes'].append(node_element)

                        # Forcing anomalies
                        # parameters_status['nodes'][0]['parameters'][2]['part_anomaly'] = True
                        # parameters_status['nodes'][2]['parameters'][2]['full_anomaly'] = True
            except Exception as e:
                print "mon_conf.pick not found! {}".format(e.message)

            node_anomalies = []
            for node_element in parameters_status['nodes']:
                # print node_element.get('parameters', list())
                na = dict(name=node_element['name'])
                na['part_n'] = len([p for p in
                                    node_element.get('parameters', list())
                                    if p['part_anomaly']])
                na['full_n'] = len([p for p in
                                    node_element.get('parameters', list())
                                    if p['full_anomaly']])
                node_anomalies.append(na)
                if na['part_n'] == 0 and na['full_n'] == 0:
                    break

            return parameters_status, node_anomalies

        def get_probabilities(static_run_dir):
            date = None
            unrest_mean = ""
            magmatic_mean = ""
            eruption_mean = ""
            try:
                with open(os.path.join(static_run_dir, "bet_ef_out.pick")) as f:
                    bet_ef_out = pickle.load(f)
                    date = bet_ef_out.date
                    unrest_mean = bet_ef_out.unrest.mean
                    magmatic_mean = bet_ef_out.magmatic.mean
                    eruption_mean = bet_ef_out.eruption.mean
            except Exception as e:
                print "bet_ef_out.pick not found! {}".format(e.message)
            return date, unrest_mean, magmatic_mean, eruption_mean

        def get_fall3d_images(static_run_dir, offset_time='00'):
            fall3d_imgs_glob = os.path.join(static_run_dir,
                                            offset_time,
                                            "fall3d", "*.jpg")
            fall3d_imgs = glob.glob(fall3d_imgs_glob)

            fall3d_problem_img = set(
                    ["-".join(img_path.split('/')[-1].split('-')[1:]) for
                     img_path in fall3d_imgs])
            try:
                suffix = str(fall3d_problem_img.pop())
                fall3d_imgs_rel = [
                    url_for("custom_static",
                            run_dir=os.path.basename(static_run_dir),
                            filename=os.path.join(offset_time,
                                                  "fall3d",
                                                  scenario + '-' + suffix))
                    for scenario in ['CFL', 'CFM', 'CFH'] if
                    os.path.exists(os.path.join(static_run_dir,
                                                offset_time,
                                                "fall3d", scenario + '-' + suffix))]
            except KeyError:
                fall3d_imgs_rel = []
            return fall3d_imgs_rel

        @self.route('/<regex("(last|bet_[0-9]{8}_[0-9]{4}_[0-9]{2})"):run_dir>/cdn/<path:filename>')
        def custom_static(run_dir, filename):
            return send_from_directory(self.config['CUSTOM_STATIC_PATH'],
                                       os.path.join(run_dir, filename))

        @self.route('/', defaults={'run_dir': 'last', 'view': 'page', 'name': None})
        @self.route('/<regex("(last|bet_[0-9]{8}_[0-9]{4}_[0-9]{2})"):run_dir>', defaults={'view': 'page', 'name': None})
        @self.route('/<regex("(last|bet_[0-9]{8}_[0-9]{4}_[0-9]{2})"):run_dir>/<view>', defaults={'name': None})
        @self.route('/<regex("(last|bet_[0-9]{8}_[0-9]{4}_[0-9]{2})"):run_dir>/<view>/<name>')
        def show(run_dir, view, name):
            page_style = request.args.get('style', None)
            off_t = request.args.get('offset', '00')
            headers = dict()
            static_run_dir, prev_dir, next_dir = get_static_rundirs(run_dir)
            if view != 'page':
                ext_link = view
                if name is None:
                    if view =='block':
                        name = 'anomalies'
                    elif view == 'tab':
                        name = 'parameters'
                    else:
                        name = 'errore'
                else:
                    ext_link += '/'+ name
                prev_a, next_a = get_link(prev_dir, resource=ext_link), \
                                 get_link(next_dir, resource=ext_link)
            else:
                prev_a, next_a = get_link(prev_dir), \
                                 get_link(next_dir)

            if run_dir == 'last':
                headers['Refresh'] = str(self._refresh_interval)
                run_dir = os.path.basename(static_run_dir)

            if static_run_dir is None or not os.path.isdir(static_run_dir):
                return "Run not specified or not found", 404

            print "Physical run_dir: {}".format(static_run_dir)

            offset_dirs = find_offset_dirs(static_run_dir)

            date, unrest_mean, magmatic_mean, eruption_mean = get_probabilities(static_run_dir)

            parameters_status, node_anomalies = get_anomalies(static_run_dir)
            fall3d_imgs_rel = dict()
            for offset in offset_dirs:
                fall3d_imgs_rel[offset] = get_fall3d_images(static_run_dir,
                                                            offset)
            print "fall3dimages: {}".format(fall3d_imgs_rel)

            try:
                with open(os.path.join(static_run_dir, "logs",
                                       "bet_suite.log")) as f:
                    suite_log = f.read()
            except Exception as e:
                print "bet_suite.log not found! {}".format(e.message)
                suite_log = ''

            # TODO: Check if template exists!
            template = 'run_' + view + '.html'

            body = render_template(template,
                                   style=page_style,
                                   run_dir=run_dir,
                                   offset_dirs=offset_dirs,
                                   off_t=off_t,
                                   name=name,
                                   obs_time=date,
                                   next=next_a,
                                   prev=prev_a,
                                   parameters_report=parameters_status,
                                   node_anomalies=node_anomalies,
                                   ef_unrest_p=unrest_mean,
                                   ef_magmatic_p=magmatic_mean,
                                   ef_eruption_p=eruption_mean,
                                   ef_prob_plot=url_for('custom_static',
                                                        run_dir=run_dir,
                                                        filename="bet_ef_probs.png"),
                                   ef_vent_plot=url_for('custom_static',
                                                        run_dir=run_dir,
                                                        filename="bet_ef.png"),
                                   vh_cond_plot="bet_vh_cond.png",
                                   vh_abs_plot="bet_vh_abs.png",
                                   tephra_plot="tephra_fall3d_M.png",
                                   vh_cond_p_kml="vh_cond_p.kml",
                                   vh_abs_p_kml="vh_abs_p.kml",
                                   tephra_p_kml="tephra_fall3d_M.kml",
                                   fall3d_imgs=fall3d_imgs_rel,
                                   disclaimer=self.bet_conf['Misc']['disclaimer'],
                                   userguide_pdf=url_for('static',
                                     filename=self.bet_conf['WebInterface'][
                                         'userguide_pdf']),
                                   suite_log=suite_log)
            return (body, 200, headers)

        @self.route('/test')
        def test():
            return render_template("test.html")

        @self.route('/update_data', methods=['POST'])
        def update_data():
            if 'data_dir' in request.form.keys():
                tmp_dir = request.form['data_dir']
                if os.path.isdir(tmp_dir):
                    self._cur_dir = tmp_dir
                    return 'Dir updated\n', 200
                else:
                    return 'Dir not valid\n', 400
            else:
                return 'No dir specified\n', 400

                # def get_run_list(self):
                #     with self._db_manager.session_scope() as session:
                #         runs = session.query(Run).\
                #             order_by(desc(Run.timestamp)).all()
                #     return runs
