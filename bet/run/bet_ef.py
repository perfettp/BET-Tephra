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
from sys import argv
import numpy as np
import logging
from bet.data import VentProbList, VentProb, SamplingAve
from bet.function import plot_vents, save_data, to_geojson, create_run_dir, \
    init_logger, log_to_file, get_logger, plot_probabilities
from bet.data.orm import Run
import bet.function.alpha_beta as ab
from bet.function.cli import opts_parser
from datetime import datetime, timedelta
from bet.service import ConfService
from bet.conf import BetConf, MonitoringConf, ParameterConf
import utm
import bet.database
from scipy.spatial import distance

class BetEFOut(object):
    def __init__(self, date, n_locations=700, n_samples=1000):
        self._date = date
        self._n_samples = n_samples
        self._unrest = None
        self._magmatic = None
        self._eruption = None
        self._percentiles = None
        self._n_locations = n_locations
        self._vents_prob = None
        self._eff_vents_i = None

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, val):
        self._date = val

    @property
    def unrest(self):
        return self._unrest

    @unrest.setter
    def unrest(self, val):
        self._unrest = val

    @property
    def magmatic(self):
        return self._magmatic

    @magmatic.setter
    def magmatic(self, val):
        self._magmatic = val

    @property
    def eruption(self):
        return self._eruption

    @eruption.setter
    def eruption(self, val):
        self._eruption = val

    @property
    def percentiles(self):
        return self._percentiles
    @percentiles.setter
    def percentiles(self, val):
        self._percentiles = val

    @property
    def vent_prob_list(self):
        return self._vents_prob

    @vent_prob_list.setter
    def vent_prob_list(self, v_list):
        self._vents_prob = v_list

    @property
    def n_locations(self):
        return self._n_locations

    @property
    def eff_vents_i(self):
        return self._eff_vents_i

    @eff_vents_i.setter
    def eff_vents_i(self, val):
        self._eff_vents_i = val



class BetEFModel(object):

    def __init__(self, bet_conf=None, mon_conf=None):
        self._conf = bet_conf
        self._monitoring_params = mon_conf
        self._result = BetEFOut(
            self._monitoring_params.date,
            n_samples=int(self._conf.BET['sampling']),
            n_locations=self._conf.vent_grid_n)

    @property
    def result(self):
        return self._result

    def run(self, monitoring=True, logger=None):

        if logger is None:
            logger = get_logger(level=logging.DEBUG)

        logger.debug("Running BetEF")
        n_samples = int(self._conf.BET['sampling'])
        n_locations = self._conf.vent_grid_n
        logger.debug("n_samples: {}, n_locations: {}".format(n_samples,
                                                             n_locations))
        nmix = float('nan')
        aveM = 1.0

        # node unrest
        p1 = [float(self._conf.BET['Unrest']['prior_probability']),
              1.0 - float(self._conf.BET['Unrest']['prior_probability'])]
        l1 = float(self._conf.BET['Unrest']['lambda'])
        d1 = [float(self._conf.BET['Unrest']['past_data_suc']),
              float(self._conf.BET['Unrest']['past_data_tot']) -
              float(self._conf.BET['Unrest']['past_data_suc'])]

        # node magmatic unrest #2
        p2 = [float(self._conf.BET['Magmatic']['prior_probability']),
              1.0 - float(self._conf.BET['Magmatic']['prior_probability'])]
        l2 = float(self._conf.BET['Magmatic']['lambda'])
        d2 = [float(self._conf.BET['Magmatic']['past_data_suc']),
              float(self._conf.BET['Magmatic']['past_data_tot']) -
              float(self._conf.BET['Magmatic']['past_data_suc'])]

        # node magmatic eruption #3
        p3 = [float(self._conf.BET['Eruption']['prior_probability']),
              1.0 - float(self._conf.BET['Eruption']['prior_probability'])]
        l3 = float(self._conf.BET['Eruption']['lambda'])
        d3 = [float(self._conf.BET['Eruption']['past_data_suc']),
              float(self._conf.BET['Eruption']['past_data_tot']) -
              float(self._conf.BET['Eruption']['past_data_suc'])]

        # node vent location #4
        p4 = [vent.prior for vent in self._conf.vent_grid]
        l4 = float(self._conf.BET['Vents']['lambda'])
        d4 = [vent.past_data for vent in self._conf.vent_grid]

        # node eruptive style #5
        node4_dependence = self._conf.BET['Styles']['node_4_dependence'].upper()\
                           in ['TRUE']
        if node4_dependence:
            raise Exception("Not implemented! TODO! (maybe...)")
        else:
            nstyles = len(self._conf.BET['Styles']['sizes'])
            p5 = np.array([self._conf.styles_grid.get_prior(s)
                  for s in self._conf.BET['Styles']['sizes']])

            l5 = self._conf.styles_grid.get_lambda()
            d5 = np.array([self._conf.styles_grid.get_past_data(s)
                  for s in self._conf.BET['Styles']['sizes']])

        # tree selection
        nodes_flag = [0, 0, 0, 0, 0]

        pp_con = []
        pp_con_ave = []

        # Node Unrest
        # TODO: select on node name instead index
        node = self._monitoring_params.nodes[0]
        alpha_unrest = ab.makeAlpha16(2, p1, l1, d1)
        posterior = np.random.dirichlet(alpha_unrest, n_samples).transpose()
        aveLT = ab.theoreticalAverage(alpha_unrest)

        if monitoring:
            anomaly_degree_arr = ab.calc_anomaly_degree_parameters(
                    node.parameters)
            deg_unrest = 1 - np.prod(1.0 - anomaly_degree_arr)
            logger.info("Degree of Unrest: {0}".format(deg_unrest))
            nmix = int(deg_unrest*n_samples)
            sample1 = np.ones((nmix))
            sample2 = np.array(posterior[0][: n_samples-nmix])
            pp_yes = np.concatenate([sample1, sample2])
            sample1 = np.zeros((nmix))
            sample2 = np.array(posterior[1][:n_samples-nmix])
            pp_no = np.concatenate([sample1,sample2])
            tmp = np.transpose(np.vstack((pp_yes, pp_no)))
            pp_unrest = np.random.permutation(tmp)
            pp_con_ave.append(ab.mixingAverage(aveLT, aveM, deg_unrest))
        else:
            deg_unrest = 0
            pp_con_ave.append(aveLT)
            pp_unrest = posterior.transpose()
        pp_con.append(pp_unrest[:, nodes_flag[0]])
        logger.debug("Node unrest: aveLT:{} aveM:{}".format(aveLT, aveM))

        # Node Magmatic
        node = self._monitoring_params.nodes[1]
        alpha_magma = ab.makeAlpha16(2, p2, l2, d2)
        posterior = np.random.dirichlet(alpha_magma, n_samples).transpose()
        aveLT = ab.theoreticalAverage(alpha_magma)
        if monitoring:
            probM, aveM2 = ab.calc_monitoring_prob(node.parameters,
                                                   n_samples,
                                                   nmix)
            pp_magma = ab.mixing(posterior, probM, n_samples, nmix)
            pp_con_ave.append(ab.mixingAverage(aveLT, aveM2, deg_unrest))
            logger.debug("Node magmatic: aveLT:{} aveM2:{}".format(aveLT,
                                                                   aveM2))
        else:
            pp_con_ave.append(aveLT)
            pp_magma = posterior.transpose()
            logger.debug("Node magmatic: aveLT:{}".format(aveLT))
        pp_con.append(pp_magma[:, nodes_flag[1]])


        # Node Eruption
        node = self._monitoring_params.nodes[2]
        alpha_magma_eru = ab.makeAlpha16(2, p3, l3, d3)
        posterior = np.random.dirichlet(alpha_magma_eru, n_samples).transpose()
        aveLT = ab.theoreticalAverage(alpha_magma_eru)
        if monitoring :
            probM, aveM3 = ab.calc_monitoring_prob(node.parameters,
                                                   n_samples,
                                                   nmix)
            pp_magma_eru = ab.mixing(posterior, probM, n_samples, nmix)
            pp_con_ave.append(ab.mixingAverage(aveLT, aveM3, deg_unrest))
            logger.debug("Node eruption: aveLT:{} aveM3:{}".format(aveLT,
                                                                   aveM3))
        else:
            pp_con_ave.append(aveLT)
            pp_magma_eru = posterior.transpose()
            logger.debug("Node eruption: aveLT:{}".format(aveLT))

        pp_con.append(pp_magma_eru[:, nodes_flag[2]])

        # Node Vent
        alpha4 = ab.makeAlpha16(n_locations, p4, l4, d4)
        posterior = np.random.dirichlet(alpha4, n_samples).transpose()
        aveLT = ab.theoreticalAverage(alpha4)
        if monitoring:
            try:
                node = self._monitoring_params.nodes[3]
                vent_probabilities = ab.calc_vent_spatial_ave(
                        node.parameters,
                        self._conf.vent_grid
                )
            except IndexError:
                logger.warn("WARNING: Node parameters not defined!")
                vent_probabilities = np.zeros((n_locations))

            if (isinstance(vent_probabilities, float) and
                    np.isnan(vent_probabilities)) or \
                            np.sum(vent_probabilities) == 0:
                logger.warn("WARNING: Sum of Monitoring Vent = 0")
                pp_con_ave.append(aveLT)
                pp_vent = posterior
            else:
                nmix4 = int(min(deg_unrest, 0.5) * n_samples)
                alphaM4 = n_locations * vent_probabilities
                probM = np.random.dirichlet(alphaM4, nmix4)
                aveM4 = alphaM4 / np.sum(alphaM4)
                pp_vent = ab.mixing(posterior, probM, n_samples, nmix4)
                pp_vent = pp_vent.transpose()
                pp_con_ave.append(ab.mixingAverage(aveLT,
                                                   aveM4,
                                                   nmix4/float(n_samples)))
                logger.debug("Node vent: aveLT:{} aveM4:{}".format(aveLT,
                                                                   aveM4))
        else:
            pp_con_ave.append(aveLT)
            pp_vent = posterior
            logger.debug("Node eruption: aveLT:{}".format(aveLT))

        pp4 = pp_vent[range(n_locations), :]     # posterior n4 conditional
        pp_con.append(pp_vent)


        # Node 5 Magmatic Sizes/Styles
        tmp5 = np.zeros((n_locations, n_samples, nstyles))

        alpha5 = [0] * (n_locations)
        if node4_dependence:
            for i in range(n_locations):
                alpha5[i] = ab.makeAlpha16(nstyles, p5[:,i],
                                           l5[i], d5[:,i])
                posterior = np.random.dirichlet(alpha5[i], n_samples)
                tmp5[i,:,:] = posterior
        else:
            for i in range(n_locations):
                alpha5[i] = ab.makeAlpha16(nstyles, p5, l5, d5)
                posterior = np.random.dirichlet(alpha5[i], n_samples)
                tmp5[i,:,:] = posterior

        if (int(nodes_flag[4])%2==0):
            tmp = int(nodes_flag[4])/2 - 1
            ind5 = range(tmp,nstyles)
        else:
            ind5 = [int(nodes_flag[4]+1)/2 - 1]

        pp5 = np.zeros((n_locations, n_samples, len(ind5)))
        for i in range(n_locations):
            pp5[i,:,:] = np.transpose(tmp5[i,:,ind5])

        aveLT = np.zeros((n_locations, nstyles))
        for i in range(n_locations):
          aveLT[i, :] = ab.theoreticalAverage(alpha5[i])

        pp5cond = np.zeros((nstyles, n_samples))
        for j in range(nstyles):
            tmp0 = 0
            tmp1 = 0
            for i in range(n_locations):
                tmp0 += pp4[i, :]
                tmp1 += pp4[i, :] * tmp5[i, :, j]
            # p cond each size / selected vents
            pp5cond[j, :] = tmp1/tmp0

        pp_con.append(pp5cond)
        pp_con_ave.append(np.mean(aveLT, axis=0))

        logger.debug("Node size: aveLT:{}".format(aveLT))

        self._result.unrest = SamplingAve(
            mean=pp_con_ave[0],
            samples=pp_con[0])
        self._result.magmatic = SamplingAve(
            mean=pp_con_ave[1],
            samples=pp_con[1])
        self._result.eruption = SamplingAve(
            mean=pp_con_ave[2],
            samples=pp_con[2])

        self._result.vent_prob_list = VentProbList(self._conf.vent_grid)

        for i_vent in range(len(self._result.vent_prob_list)):
            self._result.vent_prob_list[i_vent].ave = \
                SamplingAve(
                    mean=pp_con_ave[3][i_vent],
                    samples=pp_con[3][i_vent])

            s_ave = dict((self._conf.BET['Styles']['sizes'][s],
                          SamplingAve(
                                  mean=pp_con_ave[4][s],
                                  samples=pp_con[4][s]))
                         for s in range(nstyles))
            self._result.vent_prob_list[i_vent].sizes_ave = s_ave

        eff_vents_i = []
        for i_v in range(self._result.n_locations):
            if self._result.vent_prob_list[i_v].ave.mean > 0:
                eff_vents_i.append(i_v)
        self._result.eff_vents_i = eff_vents_i

        logger.info("Unrest: {:.6f}/{:.6f}/{:.6f}/{:.6f} (ave/10/50/90)".format(
            self._result.unrest.mean,
            np.percentile(self._result.unrest.samples, 10),
            np.percentile(self._result.unrest.samples, 50),
            np.percentile(self._result.unrest.samples, 90)
        ))

        logger.info("Magmatic: {:.6f}/{:.6f}/{:.6f}/{:.6f} (ave/10/50/90)".format(
            self._result.magmatic.mean,
            np.percentile(self._result.magmatic.samples, 10),
            np.percentile(self._result.magmatic.samples, 50),
            np.percentile(self._result.magmatic.samples, 90)
        ))

        logger.info("Eruption: {:.6f}/{:.6f}/{:.6f}/{:.6f} (ave/10/50/90)".format(
            self._result.eruption.mean,
            np.percentile(self._result.eruption.samples, 10),
            np.percentile(self._result.eruption.samples, 50),
            np.percentile(self._result.eruption.samples, 90)
        ))

        iprint = 100
        logger.info("Vent n.{}: {:.6f}/{:.6f}/{:.6f}/{:.6f} (ave/10/50/90)".format(
            iprint + 1,
            self._result.vent_prob_list[iprint].ave.mean,
            np.percentile(self._result.vent_prob_list[iprint].ave.samples, 10),
            np.percentile(self._result.vent_prob_list[iprint].ave.samples, 50),
            np.percentile(self._result.vent_prob_list[iprint].ave.samples, 90)
        ))

        for s in self._conf.BET['Styles']['sizes']:
            logger.info("Style {}: {:.6f}/{:.6f}/{:.6f}/{:.6f} "
                        "(ave/10/50/90)".format(
                    s,
                    self._result.vent_prob_list[iprint].sizes_ave[s].mean,
                    np.percentile(self._result.vent_prob_list[iprint].sizes_ave[
                                      s].samples, 10),
                    np.percentile(self._result.vent_prob_list[iprint].sizes_ave[
                                      s].samples, 50),
                    np.percentile(self._result.vent_prob_list[iprint].sizes_ave[
                                      s].samples, 90)
        ))


if __name__ == "__main__":

    opts = vars(opts_parser().parse_args(argv[1:]))

    logger = get_logger(level=logging.DEBUG, name="bet_ef")
    # logger = logging.getLogger("bet_ef")

    if opts['obs_time']:
        obs_time = datetime.strptime(opts['obs_time'], "%Y%m%d_%H%M%S")
    else:
        obs_time = datetime.now()

    bet_conf = BetConf(opts['conf'], obs_time=obs_time)

    if opts['run_dir']:
        run_dir = opts['run_dir']
    else:
        run_dir = create_run_dir(bet_conf.BET['data_dir'], obs_time)

    log_to_file(run_dir, 'bet_ef.log', level=logging.DEBUG)

    if opts['monitoring_file']:
        logger.info("Parsing monitoring parameters from file {}".format(
                opts['monitoring_file']))
        mon_conf = MonitoringConf()
        with open(opts['monitoring_file'], 'r') as mon_file:
            mon_conf.from_json(mon_file.read())
    else:
        cs = ConfService(volcano_name='Campi_Flegrei', elicitation=6,
                         runmodel_class='TestModel',
                         mapmodel_name='CardinalModelTest',
                         bet_conf=bet_conf, logger=logger)
        mon_conf = cs.get_monitoring_conf(obs_time)

    if run_dir is None:
        logger.critical("Cannot create run_dir, exiting")
        exit(-1)

    logger.debug("Using run_dir {}".format(run_dir))

    bet_conf.load_vent_grid()
    bet_conf.load_style_grid()

    # From monitoring calculate probabilities over vent map

    latlon_grid = np.array([utm.to_latlon(v.point.easting,
                                          v.point.northing,
                                          v.point.zone_number,
                                          v.point.zone_letter)
                            for v in bet_conf.vent_grid])

    logger.debug("Creating BetEFModel object")
    ef_model = BetEFModel(bet_conf, mon_conf)
    ef_model.run(monitoring=True)

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
        logger.info("Connection to db established")

        with ovDbManager.session_scope() as session:
            betRun = Run()
            betRun.rundir = run_dir
            # betRun.input_parameters =   mon_conf ?
            betRun.timestamp = bet_conf.obs_time
            betRun.ef_unrest = np.array([
                ef_model.result.unrest.mean,
                np.percentile(ef_model.result.unrest.samples, 16),
                np.percentile(ef_model.result.unrest.samples, 50),
                np.percentile(ef_model.result.unrest.samples, 84)])

            magmatic_abs_samples = ef_model.result.unrest.samples * \
                                   ef_model.result.magmatic.samples
            magmatic_abs = ef_model.result.unrest.mean * \
                           ef_model.result.magmatic.mean
            betRun.ef_magmatic = np.array([
                magmatic_abs,
                np.percentile(magmatic_abs_samples, 16),
                np.percentile(magmatic_abs_samples, 50),
                np.percentile(magmatic_abs_samples, 84)])

            eruption_abs_samples = magmatic_abs_samples * \
                                   ef_model.result.eruption.samples
            eruption_abs = magmatic_abs * ef_model.result.eruption.mean
            betRun.ef_eruption = np.array([
                eruption_abs,
                np.percentile(eruption_abs_samples, 16),
                np.percentile(eruption_abs_samples, 50),
                np.percentile(eruption_abs_samples, 84)])

            betRun.ef_vent_map = [v.ave.mean
                                  for v in ef_model.result.vent_prob_list]
            betRun.input_parameters = mon_conf.to_json()

            betRun._runmodel_id = 1
            session.add(betRun)
            logger.info("New run inserted in db")

    logger.debug("Saving EF outputs")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)

    with ovDbManager.session_scope() as session:
        runs = session.query(Run).filter(Run.timestamp <= end_time,
                                         Run.timestamp >= start_time).\
            order_by(Run.timestamp).all()

        times = np.array([r.timestamp for r in runs])
        data = dict()
        data['unrest'] = np.array([r.ef_unrest for r in runs])
        data['magmatic'] = np.array([r.ef_magmatic for r in runs])
        data['eruption'] = np.array([r.ef_eruption for r in runs])

    plot_probabilities(times, data, os.path.join(run_dir, "bet_ef_probs.png"))

    v_mean = np.array([v.ave.mean for v in ef_model.result.vent_prob_list])
    points = [v.point for v in bet_conf.vent_grid]

    # save_data(points, v_mean, os.path.join(run_dir, "bet_ef_data.txt"))

    # to_geojson(points, v_mean, os.path.join(run_dir,
    #                                        "bet_ef_data.geojson"))

    plot_vents(points, v_mean, os.path.join(run_dir, "bet_ef.png"),
               title='BetEF',
               scatter=True, basemap_res='f')


    logger.info("BetEF terminated")
