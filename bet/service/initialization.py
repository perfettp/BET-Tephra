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

from datetime import datetime
from bet.function.cli import opts_parser
from sys import argv
import tempfile
import bet.conf
import numpy as np
from bet.database import manager
from bet.data import CardinalSectionsModel
from bet.data import GridModel
from bet.data.orm import Volcano, Elicitation, RemoteSource, Node, RunModel
from bet.data.orm import ElicitationNode, NodeParameter
from bet.data.orm import Run
from bet.data.orm import MapModel
from bet.data.orm import Base
from bet.data import MetaUplift
from bet.data import MetaSeismic
from bet.data import MetaManual
from bet.data import SeismicVTDailyRatio, UpliftRITEThreeMonths, \
    UpliftRITEMonthlyRatio, ManualAcidGasPresence, ManualDegas, \
    SeismicDeepVTDailyRatio, SeismicAllLPMonthlyRatio, SeismicVTMaxMagnitude,\
    SeismicLPDailyRatio, SeismicDeepLPDailyRatio, SeismicVLPULPDailyRatio, \
    SeismicDeepVLPULPMonthlyRatio, ManualDeepMonthlyTremor, \
    ManualMonthlyTremor, UpliftAbnormalStationsRatio, \
    UpliftAbnormalVhorRatio, UpliftMaxSpeedStation, ManualMagmaticVariation, \
    ManualGasCompositionVariation, ManualNewFractures, \
    ManualNewIdroThermalSources, ManualPhreaticActivity, ManualRSAMAcc, \
    ManualSeismicEnergyAcc, ManualSeismicStop, ManualSeismicAcc, \
    UpliftRITEDailyRatio


from bet.data import map
from bet.test import populateRandomNormalized
from bet.test import populateEventCounter
from bet.test.test_remote_model import BaseTest

__author__ = 'Marco'


def add_node_param(sess, param, node, order, relation='=', threshold_1=1.,
                   threshold_2=1., weight=1.):
        rel = NodeParameter(order=node_param_index,
                            relation=relation,
                            threshold1=threshold_1,
                            threshold2=threshold_2,
                            weight=weight)
        rel.parameter = param
        node.parameters.append(rel)
        sess.add(rel)


def add_elic_node(node, elic, order, sess):
    rel = ElicitationNode(order=order)
    rel.node = node
    rel.order = order
    elic.nodes.append(rel)
    sess.add(rel)


class MapModelCreatorService(object):

    _supportedDbs = ["postgres"]

    def __init__(self, dbManager, **kwargs):
        self._dbManager = dbManager
        self._volcanoName = kwargs.pop('volcanoName', None)
        self._mapModelName = kwargs.pop('mapModelName', None)
        self._mapModelIndex = kwargs.pop('mapModelIndex', None)
        self._params = kwargs

    def run(self):

        with self._dbManager.session_scope() as session:

            # 1: check if not exists already
            mapModel = session.query(MapModel).join(Volcano).filter(
                Volcano.description == self._volcanoName,
                MapModel.name == self._mapModelName).first()
            if mapModel is not None:
                raise AttributeError("This mapModel already exists! Change "
                                      "name and/or volcano reference")

            # 2: build the mapmodel
            if self._mapModelIndex == 1:
                mapModel = GridModel(**self._params)
            elif self._mapModelIndex == 2:
                mapModel = CardinalSectionsModel(**self._params)
            elif self._mapModelIndex == 3:
                mapModel = map.ShapeFileModel(**self._params)
            else:
                raise AttributeError("Unknown model type! If you implement a new model class, please add it in the run method inside MapModelCreatorService class.")

            mapModel.volcano = session.query(Volcano).filter(
                Volcano.description == self._volcanoName).first()
            if mapModel.volcano is not None:
                mapModel.name = self._mapModelName
                mapModel.index = self._mapModelIndex

                session.add(mapModel)
            else:
                raise AttributeError("Volcano not found!")


if __name__ == "__main__":

    opts = vars(opts_parser().parse_args(argv[1:]))

    if opts['obs_time']:
        obs_time = datetime.strptime(opts['obs_time'], "%Y%m%d_%H%M%S")
    else:
        obs_time = datetime.now()

    # bet_conf = bet.conf.BetConf(opts['conf'], obs_time=obs_time)
    # bet_conf.merge_local_conf()

    bet_conf = bet.conf.BetConf(opts['conf'])

    # init dbms handler
    ovDbManager = manager.DbManager(db_type=bet_conf.BET['Database']['db_type'],
                                    db_host=bet_conf.BET['Database']['db_host'],
                                    db_port=bet_conf.BET['Database']['db_port'],
                                    db_user=bet_conf.BET['Database']['db_user'],
                                    db_password=bet_conf.BET['Database']['db_pwd'],
                                    debug=False)

    dbName = bet_conf.BET['Database']['db_name']
    # Initialization works only if DB not exists still!

    if not ovDbManager.exists_db(dbName):
        ovDbManager.init_and_use_db(dbName, Base.metadata)

        # Init database objects
        with ovDbManager.session_scope() as session:

            # Populate volcanos
            volcano = Volcano()
            volcano.description = "Campi_Flegrei"
            session.add(volcano)

            # Populate elicitation
            elicitation = Elicitation()
            elicitation.elicitation_number = 6
            elicitation.description = "testSEI"
            elicitation.volcano = volcano
            session.add(elicitation)

            # Populate remote sources
            sismolabSource = RemoteSource()
            sismolabSource.host = bet_conf['DataSources']['sismolab']['db_host']
            sismolabSource.port = bet_conf['DataSources']['sismolab']['db_port']
            sismolabSource.type = bet_conf['DataSources']['sismolab']['db_type']
            sismolabSource.user = bet_conf['DataSources']['sismolab']['db_user']
            sismolabSource.password = bet_conf['DataSources']['sismolab']['db_pwd']
            sismolabSource.name = bet_conf['DataSources']['sismolab']['db_name']

            speedSource = RemoteSource()
            speedSource.host = bet_conf['DataSources']['speed']['db_host']
            speedSource.port = bet_conf['DataSources']['speed']['db_port']
            speedSource.type = bet_conf['DataSources']['speed']['db_type']
            speedSource.user = bet_conf['DataSources']['speed']['db_user']
            speedSource.password = bet_conf['DataSources']['speed']['db_pwd']
            speedSource.name = bet_conf['DataSources']['speed']['db_name']

            # Populate metaparameters
            metaUplift = MetaUplift()
            metaUplift.description = "metauplift"
            metaUplift.elicitation = elicitation
            metaUplift.remotesource = speedSource

            metaSeismic = MetaSeismic()
            metaSeismic.description = "metaseismic"
            metaSeismic.elicitation = elicitation
            metaSeismic.remotesource = sismolabSource

            metaManual = MetaManual()
            metaManual.description = "MetaManual"
            metaManual.elicitation = elicitation

            # Populate nodes
            elic_node_index = 0

            node1 = Node()
            node1.name = "Unrest"
            session.add(node1)
            add_elic_node(node1, elicitation, elic_node_index, session)

            elic_node_index += 10
            node2 = Node()
            node2.name = "Magmatic"
            session.add(node2)
            add_elic_node(node2, elicitation, elic_node_index, session)

            elic_node_index += 10
            node3 = Node()
            node3.name = "Eruption"
            session.add(node3)
            add_elic_node(node3, elicitation, elic_node_index, session)

            elic_node_index += 10
            node4 = Node()
            node4.name = "Vent"
            session.add(node4)
            add_elic_node(node4, elicitation, elic_node_index, session)

            # Parameters definitions
            seisVTDailyRatio = SeismicVTDailyRatio()
            seisVTDailyRatio.description = "Numero di VT (M > 0.8), [ev/giorno]"
            seisVTDailyRatio.metaparameter = metaSeismic
            seisVTDailyRatio.parameter_family = 'Seismic'
            session.add(seisVTDailyRatio)

            seisDeepVTRatio = SeismicDeepVTDailyRatio()
            seisDeepVTRatio.description = "Numero di VT a prof > 3.5 km " \
                                          "(M > 0.8), [ev/giorno]"
            seisDeepVTRatio.metaparameter = metaSeismic
            seisDeepVTRatio.parameter_family = 'Seismic'
            session.add(seisDeepVTRatio)

            seisVTMaxMagnitude = SeismicVTMaxMagnitude()
            seisVTMaxMagnitude.description = "Massima magnitudo (ultimo mese)"
            seisVTMaxMagnitude.metaparameter = metaSeismic
            seisVTMaxMagnitude.parameter_family = 'Seismic'
            session.add(seisVTMaxMagnitude)

            seisAllLPMonthlyRatio = SeismicAllLPMonthlyRatio()
            seisAllLPMonthlyRatio.description = "Numero(*)(**) di LP/VLP/ULP, [ev/mese]"
            seisAllLPMonthlyRatio.metaparameter = metaSeismic
            seisAllLPMonthlyRatio.parameter_family = 'Seismic'
            session.add(seisAllLPMonthlyRatio)

            seisLPDailyRatio = SeismicLPDailyRatio()
            seisLPDailyRatio.description = "Numero(*)(**) di LP, [ev/giorno]"
            seisLPDailyRatio.metaparameter = metaSeismic
            seisLPDailyRatio.parameter_family = 'Seismic'
            session.add(seisLPDailyRatio)

            seisDeepLPDailyRatio = SeismicDeepLPDailyRatio()
            seisDeepLPDailyRatio.description = "Numero(*)(**)(***) di LP a " \
                                               "prof > 2.0 km, [ev/giorno]"
            seisDeepLPDailyRatio.metaparameter = metaSeismic
            seisDeepLPDailyRatio.parameter_family = 'Seismic'
            session.add(seisDeepLPDailyRatio)

            seisVLPULPDailyRatio = SeismicVLPULPDailyRatio()
            seisVLPULPDailyRatio.description = "Numero(*)(**) di VLP/ULP, " \
                                               "[ev/giorno]"
            seisVLPULPDailyRatio.metaparameter = metaSeismic
            seisVLPULPDailyRatio.parameter_family = 'Seismic'
            session.add(seisVLPULPDailyRatio)

            seisDeepVLPULPMonthlyRatio = SeismicDeepVLPULPMonthlyRatio()
            seisDeepVLPULPMonthlyRatio.description = "Numero(*)(**)(***) di " \
                                                     "VLP/ULP a prof > 2.0 " \
                                                     "km,  [ev/mese]"
            seisDeepVLPULPMonthlyRatio.metaparameter = metaSeismic
            seisDeepVLPULPMonthlyRatio.parameter_family = 'Seismic'
            session.add(seisDeepVLPULPMonthlyRatio)

            manualMonthlyTremor = ManualMonthlyTremor()
            manualMonthlyTremor.description = "Tremore (ultimo mese), YES/NO"
            manualMonthlyTremor.metaparameter = metaManual
            manualMonthlyTremor.parameter_family = 'Seismic'
            session.add(manualMonthlyTremor)

            manualDeepMonthlyTremor = ManualDeepMonthlyTremor()
            manualDeepMonthlyTremor.description = "Tremore profondo (>3.5 " \
                                                   "km) (ultimo mese), YES/NO"
            manualDeepMonthlyTremor.metaparameter = metaManual
            manualDeepMonthlyTremor.parameter_family = 'Seismic'
            session.add(manualDeepMonthlyTremor)

            # upliftThreeMonths = UpliftThreeMonths()
            # upliftThreeMonths.description = "Uplift (cumulativo negli " \
            #                                 "ultimi 3 mesi) [cm]"
            # upliftThreeMonths.metaparameter = metaUplift
            # upliftThreeMonths.parameter_family = 'Deformation'
            # session.add(upliftThreeMonths)

            upliftRiteThreeMonths = UpliftRITEThreeMonths()
            upliftRiteThreeMonths.description = "Uplift on RITE (cumulativo negli " \
                                            "ultimi 3 mesi) [cm]"
            upliftRiteThreeMonths.metaparameter = metaUplift
            upliftRiteThreeMonths.parameter_family = 'Deformation'
            session.add(upliftRiteThreeMonths)

            upliftRiteMonthlyRatio = UpliftRITEMonthlyRatio()
            upliftRiteMonthlyRatio.description = "Rateo uplift (ultimi 3 mesi) " \
                                             "[cm/mese]"
            upliftRiteMonthlyRatio.metaparameter = metaUplift
            upliftRiteMonthlyRatio.parameter_family = 'Deformation'
            session.add(upliftRiteMonthlyRatio)

            upliftRiteDailyRatio = UpliftRITEDailyRatio()
            upliftRiteDailyRatio.description = "Rateo uplift (ultimi 3 mesi) " \
                                             "[cm/giorno]"
            upliftRiteDailyRatio.metaparameter = metaUplift
            upliftRiteDailyRatio.parameter_family = 'Deformation'
            session.add(upliftRiteDailyRatio)

            manualDegas = ManualDegas()
            manualDegas.description = "Estensione strutture degassamento e/o " \
                                      "aumento flussi (ultimo mese) YES/NO"
            manualDegas.metaparameter = metaManual
            manualDegas.parameter_family = 'Geochemical'
            session.add(manualDegas)

            manualAcidGasPresence = ManualAcidGasPresence()
            manualAcidGasPresence.description = "Presenza gas acidi: HF - " \
                                                "HCl - SO2 (ultima settimana)" \
                                                ", YES/NO"
            manualAcidGasPresence.metaparameter = metaManual
            manualAcidGasPresence.parameter_family = 'Geochemical'
            session.add(manualAcidGasPresence)

            upliftAbnormalStationsRatio = UpliftAbnormalStationsRatio()
            upliftAbnormalStationsRatio.description = "Variazioni significative del " \
                                              "rapporto tra Vup a RITE e " \
                                              "Vup ad altre stazioni (x)(xx)" \
                                              "(ultimo mese), YES/NO"
            upliftAbnormalStationsRatio.metaparameter = metaUplift
            upliftAbnormalStationsRatio.parameter_family = 'Deformation'
            session.add(upliftAbnormalStationsRatio)

            upliftAbnormalVhorRatio = UpliftAbnormalVhorRatio()
            upliftAbnormalVhorRatio.description = "Variazioni significative " \
                                                  "del rapporto a Vhor/Vup a " \
                                                  "qualsiasi stazione (x)(xx)" \
                                                  "(ultimo mese), YES/NO"
            upliftAbnormalVhorRatio.metaparameter = metaUplift
            upliftAbnormalVhorRatio.parameter_family = 'Deformation'
            session.add(upliftAbnormalVhorRatio)

            upliftMaxSpeedStation = UpliftMaxSpeedStation()
            upliftMaxSpeedStation.description = "Massimo di velocita' di " \
                                                "uplift in stazione diversa " \
                                                "da RITE (x) (ultimo mese), " \
                                                "YES/NO"
            upliftMaxSpeedStation.metaparameter = metaUplift
            upliftMaxSpeedStation.parameter_family = 'Deformation'
            session.add(upliftMaxSpeedStation)

            manualMagmaticVariation = ManualMagmaticVariation()
            manualMagmaticVariation.description = "Variazione della frazione " \
                                                  "della componente " \
                                                  "magmatica (ultimo mese), " \
                                                  "YES/NO"
            manualMagmaticVariation.metaparameter = metaManual
            manualMagmaticVariation.parameter_family = 'Geochemical'
            session.add(manualMagmaticVariation)

            manualGasCompositionVariation = ManualGasCompositionVariation()
            manualGasCompositionVariation.description = "Variazione " \
                                                        "composizione dei " \
                                                        "gas (ultimo mese), " \
                                                        "YES/NO"
            manualGasCompositionVariation.metaparameter = metaManual
            manualGasCompositionVariation.parameter_family = 'Geochemical'
            session.add(manualGasCompositionVariation)

            manualRSAMAcc = ManualRSAMAcc()
            manualRSAMAcc.description = "Accelerazione RSAM (ultima " \
                                        "settimana), YES/NO"
            manualRSAMAcc.metaparameter = metaManual
            manualRSAMAcc.parameter_family = 'Seismic'
            session.add(manualRSAMAcc)

            manualSeismicAcc = ManualSeismicAcc()
            manualSeismicAcc.description = "Accelerazione del numero di " \
                                           "eventi sismici (ultima " \
                                           "settimana), YES/NO"
            manualSeismicAcc.metaparameter = metaManual
            manualSeismicAcc.parameter_family = 'Seismic'
            session.add(manualSeismicAcc)

            manualSeismicEnergyAcc = ManualSeismicEnergyAcc()
            manualSeismicEnergyAcc.description = "Accelerazione del energia " \
                                                 "sismica rilasciata " \
                                                 "(ultima settimana), YES/NO"
            manualSeismicEnergyAcc.metaparameter = metaManual
            manualSeismicEnergyAcc.parameter_family = 'Seismic'
            session.add(manualSeismicEnergyAcc)

            manualNewFractures = ManualNewFractures()
            manualNewFractures.description = "Nuove fratture (significative) " \
                                             "(ultimi 3 mesi), YES/NO"
            manualNewFractures.metaparameter = metaManual
            manualNewFractures.parameter_family = 'Seismic'
            session.add(manualNewFractures)

            manualNewIdroThermalSources = ManualNewIdroThermalSources()
            manualNewIdroThermalSources.description = "Nuove sorgenti " \
                                                      "(idrotermali) (ultima " \
                                                      "settimana), YES/NO"
            manualNewIdroThermalSources.metaparameter = metaManual
            manualNewIdroThermalSources.parameter_family = 'Geochemical'
            session.add(manualNewIdroThermalSources)

            manualPhreaticActivity = ManualPhreaticActivity()
            manualPhreaticActivity.description = "Attivita freatica " \
                                                 "(principale) (ultima " \
                                                 "settimana), YES/NO"
            manualPhreaticActivity.metaparameter = metaManual
            manualPhreaticActivity.parameter_family = 'Misc'
            session.add(manualPhreaticActivity)

            manualSeismicStop = ManualSeismicStop()
            manualSeismicStop.description = "Improvviso stop sismicita' e/o " \
                                            "deformazione (ultima settimana), "\
                                            "YES/NO"
            manualSeismicStop.metaparameter = metaManual
            manualSeismicStop.parameter_family = 'Misc'
            session.add(manualSeismicStop)

            # Node construction
            # Node 1 - Unrest
            node_param_index = 0
            add_node_param(session, seisVTDailyRatio, node1, node_param_index,
                           relation='>', threshold_1=5, threshold_2=20)

            node_param_index += 10
            add_node_param(session, seisDeepVTRatio, node1, node_param_index,
                           relation='>', threshold_1=1, threshold_2=3)

            node_param_index += 10
            add_node_param(session, seisVTMaxMagnitude, node1, node_param_index,
                           relation='>', threshold_1=2, threshold_2=2.5)

            node_param_index += 10
            add_node_param(session, seisAllLPMonthlyRatio, node1, node_param_index,
                           relation='>', threshold_1=3, threshold_2=10)

            node_param_index += 10
            add_node_param(session, manualMonthlyTremor, node1, node_param_index,
                           relation='=', threshold_1=1, threshold_2=1)

            node_param_index += 10
            add_node_param(session, upliftRiteThreeMonths, node1, node_param_index,
                           relation='>', threshold_1=2, threshold_2=4.5)

            node_param_index += 10
            add_node_param(session, upliftRiteMonthlyRatio, node1, node_param_index,
                           relation='>', threshold_1=0.85, threshold_2=1.4)

            node_param_index += 10
            add_node_param(session, manualDegas, node1, node_param_index,
                           relation='=', threshold_1=1, threshold_2=1)

            node_param_index += 10
            add_node_param(session, manualAcidGasPresence, node1, node_param_index,
                           relation='=', threshold_1=1, threshold_2=1)

            # Node 2 - Magmatic
            node_param_index = 0
            add_node_param(session, seisAllLPMonthlyRatio, node2, node_param_index,
                           relation='>', threshold_1=10, threshold_2=50,
                           weight=1)

            node_param_index += 10
            add_node_param(session, seisDeepVTRatio, node2, node_param_index,
                           relation='>', threshold_1=2, threshold_2=10,
                           weight=1)

            node_param_index += 10
            add_node_param(session, seisVTMaxMagnitude, node2, node_param_index,
                           relation='>', threshold_1=2.5, threshold_2=3,
                           weight=0.32)

            node_param_index += 10
            add_node_param(session, seisLPDailyRatio, node2, node_param_index,
                           relation='>', threshold_1=2., threshold_2=20,
                           weight=1)

            node_param_index += 10
            add_node_param(session, seisDeepLPDailyRatio, node2, node_param_index,
                           relation='>', threshold_1=3., threshold_2=10,
                           weight=0.68)

            node_param_index += 10
            add_node_param(session, seisVLPULPDailyRatio, node2, node_param_index,
                           relation='>', threshold_1=1., threshold_2=5,
                           weight=0.92)

            node_param_index += 10
            add_node_param(session, seisDeepVLPULPMonthlyRatio, node2,
                           node_param_index,
                           relation='>', threshold_1=1., threshold_2=5.,
                           weight=0.32)

            node_param_index += 10
            add_node_param(session, manualMonthlyTremor, node2,
                           node_param_index,
                           relation='=', threshold_1=1., threshold_2=1.,
                           weight=0.08)

            node_param_index += 10
            add_node_param(session, manualDeepMonthlyTremor, node2,
                           node_param_index,
                           relation='=', threshold_1=1., threshold_2=1.,
                           weight=1)

            node_param_index += 10
            add_node_param(session, upliftRiteThreeMonths, node2, node_param_index,
                           relation='>', threshold_1=10, threshold_2=30,
                           weight=1)

            node_param_index += 10
            add_node_param(session, upliftRiteMonthlyRatio, node2, node_param_index,
                           relation='>', threshold_1=3, threshold_2=10,
                           weight=0.88)

            node_param_index += 10
            add_node_param(session, upliftAbnormalStationsRatio, node2,
                           node_param_index, relation='=', threshold_1=1,
                           threshold_2=1, weight=0.4)

            node_param_index += 10
            add_node_param(session, upliftAbnormalVhorRatio, node2,
                           node_param_index, relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, upliftMaxSpeedStation, node2,
                           node_param_index, relation='=', threshold_1=1,
                           threshold_2=1, weight=0.16)

            node_param_index += 10
            add_node_param(session, manualDegas, node2, node_param_index,
                           relation='=', threshold_1=1, threshold_2=1,
                           weight=0.16)

            node_param_index += 10
            add_node_param(session, manualAcidGasPresence, node2,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.76)

            node_param_index += 10
            add_node_param(session, manualMagmaticVariation, node2,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.52)

            node_param_index += 10
            add_node_param(session, manualGasCompositionVariation, node2,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.40)

            # Node 3 - Eruption
            node_param_index = 0
            add_node_param(session, manualRSAMAcc, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, manualSeismicAcc, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.48)

            node_param_index += 10
            add_node_param(session, manualSeismicEnergyAcc, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, manualMonthlyTremor, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, manualDeepMonthlyTremor, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.28)

            node_param_index += 10
            add_node_param(session, upliftRiteThreeMonths, node3,
                           node_param_index, relation='>', threshold_1=20,
                           threshold_2=100, weight=0.44)

            node_param_index += 10
            add_node_param(session, upliftRiteDailyRatio, node3,
                           node_param_index, relation='>', threshold_1=5,
                           threshold_2=20, weight=0.44)

            node_param_index += 10
            add_node_param(session, upliftAbnormalStationsRatio, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.44)

            node_param_index += 10
            add_node_param(session, upliftAbnormalVhorRatio, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, upliftMaxSpeedStation, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.92)

            node_param_index += 10
            add_node_param(session, manualNewFractures, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, manualAcidGasPresence, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, manualNewIdroThermalSources, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.92)

            node_param_index += 10
            add_node_param(session, manualPhreaticActivity, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, manualSeismicStop, node3,
                           node_param_index,relation='=', threshold_1=1,
                           threshold_2=1, weight=0.36)

            # Node 4 - Vent
            node_param_index = 0
            add_node_param(session, upliftAbnormalStationsRatio, node4,
                           node_param_index, relation='=', threshold_1=1,
                           threshold_2=1, weight=1)

            node_param_index += 10
            add_node_param(session, seisVTDailyRatio, node4,
                           node_param_index, relation='>', threshold_1=5,
                           threshold_2=20, weight=1)

            # Populate run models
            runModel = RunModel()
            runModel.class_name = "TestModel"


            session.add(volcano)
            session.add(elicitation)
            session.add(node1)
            session.add(node2)
            session.add(node3)
            session.add(node4)
            session.add(speedSource)
            session.add(metaUplift)
            session.add(metaSeismic)

            session.add(runModel)


        # Init MapModel and GeographicSample table
        mapModelIndex = 1  # from MapModel subclasses
        mapModelName = 'Grid35x20'
        volcanoName = 'Campi_Flegrei'
        # params for index = 1
        xResolution = 1
        yResolution = 1
        xSize = 35
        ySize = 20
        mapModelCreatorService = MapModelCreatorService(
            ovDbManager,
            mapModelIndex=mapModelIndex,
            mapModelName=mapModelName,
            volcanoName=volcanoName,
            xResolution=xResolution,
            yResolution=yResolution,
            xSize=xSize,
            ySize=ySize)
        mapModelCreatorService.run()

        mapModelIndex = 2   # from MapModel subclasses
        mapModelName = "CardinalModelTest"
        volcanoName = 'Campi_Flegrei'
        # params for index = 2
        innerCircleRadius = 10
        outerCircleRadius = 20
        mapModelCreatorService = MapModelCreatorService(
            ovDbManager,
            mapModelIndex=mapModelIndex,
            mapModelName=mapModelName,
            volcanoName=volcanoName,
            innerCircleRadius=innerCircleRadius,
            outerCircleRadius=outerCircleRadius )
        mapModelCreatorService.run()

    else:
        print "DB exists!"
    print "Done!"
