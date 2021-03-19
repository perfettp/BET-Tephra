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

from bet.data.orm import Parameter
from bet.data.orm import Period
from bet.data import MetaUplift
from bet.data import MetaSeismic
from bet.function import get_logger, \
    inertiated_seis_events, normalized_val, \
    max_monthly_delta_15, select_samples_interval
from sqlalchemy.sql import func
import datetime
from collections import namedtuple
from math import pow, sqrt, fabs
import utm
from scipy.spatial import distance
import numpy as np
import logging

LOG_LEVEL = logging.DEBUG

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(pathname)s:%(" \
             "lineno)d [%("\
             "process)d]: %(message)s"

LOG_FORMATTER = logging.Formatter(LOG_FORMAT)

__logger = None


# Uplift (cumulativo negli ultimi 3 mesi), [cm]
class UpliftRITEThreeMonths(Parameter):

    _parameter_identity = 'UpliftRITEThreeMonths'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    station_name = 'RITE'

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=90)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        logger = get_logger()
        sample_startdate = self.get_currentreferencedate(sample_time)
        uplift_samples = data.filter(
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > sample_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())

        uplift_stations = list((set([sample.station
                                     for sample in uplift_samples.all()])))

        rite_samples = data.filter(
            MetaUplift.RemoteMapping.station == self.station_name,
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > sample_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())


        r_start_sample = MetaUplift.get_start_sample(rite_samples.all())
        r_end_sample = MetaUplift.get_end_sample(rite_samples.all())
        # normalize value and convert in cm
        # val = (r_end_sample.value - r_start_sample.value) / 10.
        val = normalized_val(r_start_sample, r_end_sample,
                             self.inertia_duration)/10

        logger.debug("UpliftRITEThreeMonths: {}cm".format(val))
        logger.debug("Starting sample: {}".format(r_start_sample))
        logger.debug("Ending sample: {}".format(r_end_sample))
        # print "UpliftRITEThreeMonths: {}".format(val)

        uplift_map = []
        for stat_name in uplift_stations:
            stat_samples = uplift_samples.filter(
                MetaUplift.RemoteMapping.station == stat_name)
            x_location = stat_samples.first().x_location
            y_location = stat_samples.first().y_location
            s_start_sample = MetaUplift.get_start_sample(stat_samples.all())
            s_end_sample = MetaUplift.get_end_sample(stat_samples.all())
            stat_vup = normalized_val(s_start_sample, s_end_sample,
                                      self.inertia_duration)/10
            stat_val = dict(station=stat_name, lat=y_location,
                            lon=x_location, val=stat_vup)
            uplift_map.append(stat_val)
            if stat_name == self.station_name:
                val = stat_vup

        return val, uplift_map


# Rateo uplift (ultimi 3 mesi), [cm/mese]
class UpliftRITEMonthlyRatio(Parameter):

    _parameter_identity = 'UpliftRITEMonthlyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    station_name = 'RITE'

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=90)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        logger = get_logger()

        sample_startdate = self.get_currentreferencedate(sample_time)
        rite_samples = data.filter(
            MetaUplift.RemoteMapping.station == self.station_name,
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > sample_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())

        val, r1, r2 = max_monthly_delta_15(rite_samples.all())
        logger.debug("UpliftRITEMonthlyRatio: {}cm".format(val/10))
        logger.debug("Starting sample {}".format(r1))
        logger.debug("Ending sample {}".format(r2))

        logger.info("UpliftRITEMonthlyRatio: {}".format(abs(val/10)))

        return abs(val/10), None


# Rateo uplift (ultimi 3 mesi), [cm/giorno]
class UpliftRITEDailyRatio(Parameter):

    _parameter_identity = 'UpliftRITEDailyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    station_name = 'RITE'

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=90)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        logger = get_logger()

        sample_startdate = self.get_currentreferencedate(sample_time)
        rite_samples = data.filter(
            MetaUplift.RemoteMapping.station == self.station_name,
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > sample_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())

        val, r1, r2 = max_monthly_delta_15(rite_samples.all())
        daily_ratio_cm = fabs(val/10/30)
        logger.info("UpliftRITEDailyRatio: {}cm".format(daily_ratio_cm))

        return daily_ratio_cm, None


# Variazioni significative del rapporto tra Vup a RITE e Vup ad altre
# stazioni (x)(xx) (ultimo mese), YES/NO
class UpliftAbnormalStationsRatio(Parameter):

    _parameter_identity = 'UpliftAbnormalStationsRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    station_ratio = dict(ACAE=(1.1, 1.2),
                         ARFE=(2.0, 2.3),
                         IPPO=(3.7, 8.4),
                         SOLO=(1.2, 1.4),
                         STRZ=(1.6, 1.7))
    station_name = 'RITE'

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        logger = get_logger()

        rite_ratio_startdate = self.get_currentreferencedate(
            sample_time, diff_days=datetime.timedelta(days=90))

        rite_ratio_samples = data.filter(
            MetaUplift.RemoteMapping.station == self.station_name,
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > rite_ratio_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())

        rite_ratio, r1, r2 = max_monthly_delta_15(rite_ratio_samples.all())

        if abs(rite_ratio) < 30:
            logger.debug("UpliftAbnormalStationsRatio: RITE monthly uplift is "
                         "less than 3cm ({}). Returning 0".format(
                rite_ratio/10))
            return 0, None

        # RITE uplift > 3cm/month, I really have to check anomalies
        logger.info("UpliftAbnormalStationsRatio: RITE monthly uplift is "
                    "greater than 3cm ({}), going on "
                    "checking for anomalies in other stations uplift's".format(
                rite_ratio/10))

        sample_startdate = self.get_currentreferencedate(sample_time)
        anomalies = []
        ratio_anomaly = False

        rite_samples = data.filter(
            MetaUplift.RemoteMapping.station == self.station_name,
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > sample_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())

        for (stat_key, stat_ratio_limits) in self.station_ratio.iteritems():
            stat_anomaly = False
            stat_samples = data.filter(
                MetaUplift.RemoteMapping.station == stat_key,
                MetaUplift.RemoteMapping.date <= sample_time,
                MetaUplift.RemoteMapping.date > sample_startdate,
                MetaUplift.RemoteMapping.length == 7).\
                order_by(MetaUplift.RemoteMapping.date.asc())

            if len(stat_samples.all()) > 0:
                stat_x_location = stat_samples.first().x_location
                stat_y_location = stat_samples.first().y_location

                for rite_ref_sample in rite_samples.all():
                    if stat_anomaly:
                        # In this station an anomaly has already been detected
                        continue
                    # For every sample, I search for the other samples within
                    # defined time boundaries (eg, between 10-18 days)
                    samples_to_check = select_samples_interval(
                        rite_ref_sample.date,
                        rite_samples.all())
                    for rite_diff_sample in samples_to_check:
                        if stat_anomaly:
                            # In this station an anomaly has already been detected
                            continue
                        rite_uplift = ((rite_ref_sample.value -
                                        rite_diff_sample.value) /
                                       (rite_ref_sample.date -
                                        rite_diff_sample.date).days) * 14
                        try:
                            stat_ref_sample = stat_samples.filter(
                                MetaUplift.RemoteMapping.date ==
                                rite_ref_sample.date).first()
                            stat_diff_sample = stat_samples.filter(
                                MetaUplift.RemoteMapping.date ==
                                rite_diff_sample.date).first()
                            stat_uplift = ((stat_ref_sample.value -
                                            stat_diff_sample.value) /
                                           (stat_ref_sample.date -
                                            stat_diff_sample.date).days) * 14
                        except Exception as e:
                            logger.warn("Exception getting samples: {}".format(e))

                        try:
                            stat_ratio = fabs(rite_uplift/stat_uplift)
                        except ZeroDivisionError:
                            continue
                        if (stat_ratio < stat_ratio_limits[0]
                            or stat_ratio > stat_ratio_limits[1]):
                            logger.info("Vup ratio on {} anomaly: {} ".
                                        format(stat_key, str(stat_ratio)))
                            logger.debug("In interval {} {}, {} uplift was {} cm"
                                         " and RITE uplift was {} cm, "
                                         "with ratio "
                                         "{} (ratio normally between {}-{})".
                                         format(
                                rite_ref_sample.date.date(),
                                rite_diff_sample.date.date(),
                                stat_key, stat_uplift/10,
                                rite_uplift/10, stat_ratio,
                                stat_ratio_limits[0],
                                stat_ratio_limits[1]))
                            stat_anomaly = True
                            ratio_anomaly = True
                            anomaly_dict = dict(
                                station=stat_key,
                                lat=stat_y_location,
                                lon=stat_x_location,
                                val=1.)
                            anomalies.append(anomaly_dict)

        if ratio_anomaly:
            latlon_vals = np.array([(v['lat'], v['lon']) for v in anomalies])
            freq_sum = np.sum(np.array([abs(v['val']) for v in anomalies]))
            tmp_distance = distance.cdist(map_model, latlon_vals)
            tmp_argmin = tmp_distance.argmin(0)
            tmp_freq_vents = np.zeros((len(map_model)))
            for i_sample in range(len(tmp_argmin)):
                tmp_freq_vents[tmp_argmin[i_sample]] += \
                    abs(anomalies[i_sample]['val'])/freq_sum
            geovalues_norm = tmp_freq_vents

            return 1, geovalues_norm
        else:
            return 0, None


# Variazioni significative del rapporto a Vhor/Vup a qualsiasi
# stazione (x)(xx) (ultimo mese), YES/NO
class UpliftAbnormalVhorRatio(Parameter):

    _parameter_identity = 'UpliftAbnormalVhorRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    station_name = 'RITE'

    station_ratio = dict(ACAE=(0.5, 0.6),
                         ARFE=(0.8, 1.1),
                         IPPO=(1.0, 2.6),
                         RITE=(0.2, 0.3),
                         SOLO=(0.5, 0.7),
                         STRZ=(0.6, 0.8))

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        logger = get_logger()

        rite_ratio_startdate = self.get_currentreferencedate(
            sample_time, diff_days=datetime.timedelta(days=90))

        rite_ratio_samples = data.filter(
            MetaUplift.RemoteMapping.station == self.station_name,
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > rite_ratio_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())

        rite_ratio, r1, r2 = max_monthly_delta_15(rite_ratio_samples.all())

        if abs(rite_ratio) < 30:
            logger.debug("UpliftAbnormalVhorRatio: RITE monthly uplift is "
                         "less than 3cm ({}). Returning 0".format(
                rite_ratio/10))
            return 0, None

        # RITE uplift > 3cm/month, I really have to check anomalies
        logger.info("UpliftAbnormalVhorRatio: RITE monthly uplift is "
                    "greater than 3cm ({}), going on "
                    "checking for anomalies in other stations uplift's".format(
                rite_ratio/10))


        sample_startdate = self.get_currentreferencedate(sample_time)
        anomalies = []
        ratio_anomaly = False

        for (stat_key, stat_ratio_limits) in self.station_ratio.iteritems():
            stat_anomaly = False
            stat_samples = data.filter(
                MetaUplift.RemoteMapping.station == stat_key,
                MetaUplift.RemoteMapping.date <= sample_time,
                MetaUplift.RemoteMapping.date > sample_startdate,
                MetaUplift.RemoteMapping.length == 7).\
                order_by(MetaUplift.RemoteMapping.date.asc())

            if len(stat_samples.all()) > 0:
                stat_x_location = stat_samples.first().x_location
                stat_y_location = stat_samples.first().y_location

                for stat_ref_sample in stat_samples.all():
                    if stat_anomaly:
                        # In this station an anomaly has already been detected
                        continue

                    # For every sample, I search for the other samples within
                    # defined time boundaries (eg, between 10-18 days)
                    samples_to_check = select_samples_interval(
                        stat_ref_sample.date,
                        stat_samples.all())
                    for stat_diff_sample in samples_to_check:
                        if stat_anomaly:
                            # In this station an anomaly has already been detected
                            continue
                        stat_uplift = ((stat_ref_sample.value -
                                        stat_diff_sample.value) /
                                       (stat_ref_sample.date -
                                        stat_diff_sample.date).days) * 14
                        stat_vhor = \
                            ((sqrt(pow((stat_ref_sample.value_n -
                                        stat_diff_sample.value_n), 2)
                                   + pow((stat_ref_sample.value_e -
                                          stat_diff_sample.value_e), 2))) /
                             (stat_ref_sample.date -
                              stat_diff_sample.date).days) * 14
                        try:
                            stat_ratio = fabs(stat_vhor/stat_uplift)
                        except ZeroDivisionError:
                            continue
                        if (stat_ratio < stat_ratio_limits[0] or
                                    stat_ratio > stat_ratio_limits[1]):
                            logger.info("Vhor/Vup ratio on {} anomaly: {} ".
                                        format(stat_key, str(stat_ratio)))
                            logger.debug("In interval {} {}, {} Vhor was {} cm"
                                         " and Vup was {} cm, "
                                         "with ratio "
                                         "{} (ratio normally between {}-{})".
                                         format(
                                stat_ref_sample.date.date(),
                                stat_diff_sample.date.date(),
                                stat_key, stat_vhor/10,
                                stat_uplift/10, stat_ratio,
                                stat_ratio_limits[0],
                                stat_ratio_limits[1]))
                            stat_anomaly = True
                            ratio_anomaly = True
                            anomaly_dict = dict(
                                station=stat_key,
                                lat=stat_y_location,
                                lon=stat_x_location,
                                val=1.)
                            anomalies.append(anomaly_dict)

        if ratio_anomaly:
            latlon_vals = np.array([(v['lat'], v['lon']) for v in anomalies])
            freq_sum = np.sum(np.array([abs(v['val']) for v in anomalies]))
            tmp_distance = distance.cdist(map_model, latlon_vals)
            tmp_argmin = tmp_distance.argmin(0)
            tmp_freq_vents = np.zeros((len(map_model)))
            for i_sample in range(len(tmp_argmin)):
                tmp_freq_vents[tmp_argmin[i_sample]] += \
                    abs(anomalies[i_sample]['val'])/freq_sum
            geovalues_norm = tmp_freq_vents

            return 1, geovalues_norm
        else:
            return 0, None


# Massimo di velocita' di uplift in stazione diversa da RITE (x)
# (ultimo mese), YES/NO
class UpliftMaxSpeedStation(Parameter):

    _parameter_identity = 'UpliftMaxSpeedStation'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    station_name = 'RITE'
    stations = ['ACAE', 'ARFE', 'IPPO', 'SOLO', 'STRZ']

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        logger = get_logger()

        rite_ratio_startdate = self.get_currentreferencedate(
            sample_time, diff_days=datetime.timedelta(days=90))

        rite_ratio_samples = data.filter(
            MetaUplift.RemoteMapping.station == self.station_name,
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > rite_ratio_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())

        rite_ratio, r1, r2 = max_monthly_delta_15(rite_ratio_samples.all())

        if abs(rite_ratio) < 30:
            logger.debug("UpliftMaxSpeedStation: RITE monthly uplift is "
                         "less than 3cm ({}). Returning 0".format(
                rite_ratio/10))
            return 0, None

        # RITE uplift > 3cm/month, I really have to check anomalies
        logger.info("UpliftMaxSpeedStation: RITE monthly uplift is "
                    "greater than 3cm ({}), going on "
                    "checking for anomalies in other stations uplift's".format(
                rite_ratio/10))

        sample_startdate = self.get_currentreferencedate(sample_time)
        anomalies = []
        speed_anomaly = False

        rite_samples = data.filter(
            MetaUplift.RemoteMapping.station == self.station_name,
            MetaUplift.RemoteMapping.date <= sample_time,
            MetaUplift.RemoteMapping.date > sample_startdate,
            MetaUplift.RemoteMapping.length == 7).order_by(
            MetaUplift.RemoteMapping.date.asc())

        for stat_key in self.stations:
            stat_anomaly = False
            stat_samples = data.filter(
                MetaUplift.RemoteMapping.station == stat_key,
                MetaUplift.RemoteMapping.date <= sample_time,
                MetaUplift.RemoteMapping.date > sample_startdate,
                MetaUplift.RemoteMapping.length == 7).\
                order_by(MetaUplift.RemoteMapping.date.asc())

            if len(stat_samples.all()) > 0:
                stat_x_location = stat_samples.first().x_location
                stat_y_location = stat_samples.first().y_location

                for rite_ref_sample in rite_samples.all():
                    if stat_anomaly:
                        # In this station an anomaly has already been detected
                        continue
                    # For every sample, I search for the other samples within
                    # defined time boundaries (eg, between 10-18 days)
                    samples_to_check = select_samples_interval(
                        rite_ref_sample.date,
                        rite_samples.all())
                    for rite_diff_sample in samples_to_check:
                        if stat_anomaly:
                            # In this station an anomaly has already been detected
                            continue
                        rite_uplift = ((rite_ref_sample.value -
                                        rite_diff_sample.value) /
                                       (rite_ref_sample.date -
                                        rite_diff_sample.date).days) * 14
                        try:
                            stat_ref_sample = stat_samples.filter(
                                MetaUplift.RemoteMapping.date ==
                                rite_ref_sample.date).first()
                            stat_diff_sample = stat_samples.filter(
                                MetaUplift.RemoteMapping.date ==
                                rite_diff_sample.date).first()
                            stat_uplift = ((stat_ref_sample.value -
                                            stat_diff_sample.value) /
                                           (stat_ref_sample.date -
                                            stat_diff_sample.date).days) * 14
                        except Exception as e:
                            logger.warn("Exception getting samples: {}".format(e))

                        logger.debug("In interval {} {}, {} uplift was {} cm"
                                         " and RITE uplift was {} cm".
                                         format(
                                rite_ref_sample.date.date(),
                                rite_diff_sample.date.date(),
                                stat_key, stat_uplift/10,
                                rite_uplift/10))

                        if fabs(stat_uplift) > fabs(rite_uplift):
                            logger.info("Max speed anomaly on {} ".
                                        format(stat_key))

                            stat_anomaly = True
                            speed_anomaly = True
                            anomaly_dict = dict(
                                station=stat_key,
                                lat=stat_y_location,
                                lon=stat_x_location,
                                val=1.)
                            anomalies.append(anomaly_dict)

        if speed_anomaly:
            latlon_vals = np.array([(v['lat'], v['lon']) for v in anomalies])
            freq_sum = np.sum(np.array([abs(v['val']) for v in anomalies]))
            tmp_distance = distance.cdist(map_model, latlon_vals)
            tmp_argmin = tmp_distance.argmin(0)
            tmp_freq_vents = np.zeros((len(map_model)))
            for i_sample in range(len(tmp_argmin)):
                tmp_freq_vents[tmp_argmin[i_sample]] += \
                    abs(anomalies[i_sample]['val'])/freq_sum
            geovalues_norm = tmp_freq_vents

            return 1, geovalues_norm
        else:
            return 0, None


class SeismicCount(Parameter):

    _parameter_identity = 'SeismicCount'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=90)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        sample_startdate = self.get_currentreferencedate(sample_time)

        events = [ev for ev in data
                  if (sample_startdate < ev.date <= sample_time)]
        seismic_events = len(events)
        return seismic_events, events


# Numero(*) di VT (M > 0.8), [ev/giorno]
class SeismicVTDailyRatio(Parameter):

    _parameter_identity = 'SeismicVTDailyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    def sample_value(self, sample_time, data, map_model):
        sample_startdate = self.get_currentreferencedate(sample_time)

        events = [ev for ev in data
                  if ((sample_startdate < ev.date <= sample_time) and
                      ev.magnitude > .8 and ev.seis_type == 'VT')]

        # print "Seismic events: {}".format(events)

        if len(events) > 0:
            val = inertiated_seis_events(events, sample_time, self.inertia_duration)
            # Tutte le coppie di coordinate in cui e' stato registrato un evento
            latlon_vals = np.array([(v.latitude, v.longitude) for v in events])
            freq_sum = val
            tmp_distance = distance.cdist(map_model, latlon_vals)
            tmp_argmin = tmp_distance.argmin(0)
            tmp_freq_vents = np.zeros((len(map_model)))
            for i_sample in range(len(tmp_argmin)):
                tmp_freq_vents[tmp_argmin[i_sample]] += \
                    inertiated_seis_events([events[i_sample]],
                                           sample_time,
                                           self.inertia_duration)/freq_sum
            geovalues_norm = tmp_freq_vents
            return val, geovalues_norm
        else:
            return 0, None


# Numero(*) di VT a prof > 3.5 km (M > 0.8), [ev/giorno]
class SeismicDeepVTDailyRatio(Parameter):
    _parameter_identity = 'SeismicDeepVTDailyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        sample_startdate = self.get_currentreferencedate(sample_time)

        events = [ev for ev in data
                  if ((sample_startdate < ev.date <= sample_time) and
                  ev.magnitude > .8 and ev.depth > 3.5 and ev.seis_type == 'VT')]

        val = inertiated_seis_events(events, sample_time, self.inertia_duration)

        return val, None


# Massima magnitudo (ultimo mese), -
class SeismicVTMaxMagnitude(Parameter):

    _parameter_identity = 'SeismicVTMaxMagnitude'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    # returns total and geographic decomposition (or None)
    def sample_value(self, sample_time, data, map_model):
        sample_startdate = self.get_currentreferencedate(sample_time)

        mags = [ev.magnitude for ev in data
                if ((sample_startdate < ev.date <= sample_time) and
                    ev.seis_type == 'VT')]
        max_seis = max(mags) if len(mags)>0 else 0
        return max_seis, None


# Numero(*)(**) di LP/VLP/ULP, [ev/mese]
class SeismicAllLPMonthlyRatio(Parameter):

    _parameter_identity = 'SeismicAllLPMonthlyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    def sample_value(self, sample_time, data, map_model):
        sample_startdate = self.get_currentreferencedate(sample_time)

        events = [ev for ev in data
                  if ((sample_startdate < ev.date <= sample_time) and
                  ev.magnitude > .8 and ev.depth > 3.5 and ev.seis_type == 'LP')]

        val = inertiated_seis_events(events, sample_time, self.inertia_duration)
        return val, None


# Numero(*)(**) di LP, [ev/giorno]
class SeismicLPDailyRatio(Parameter):

    _parameter_identity = 'SeismicLPDailyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    def sample_value(self, sample_time, data, map_model):
        sample_startdate = self.get_currentreferencedate(sample_time)
        events = [ev for ev in data
                  if ((sample_startdate < ev.date <= sample_time)
                      and ev.seis_type == 'LP')]
        val = inertiated_seis_events(events, sample_time, self.inertia_duration)
        return val, None


# Numero(*)(**)(***) di LP a prof > 2.0 km, [ev/giorno]
class SeismicDeepLPDailyRatio(Parameter):

    _parameter_identity = 'SeismicDeepLPDailyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    def sample_value(self, sample_time, data, map_model):
        sample_startdate = self.get_currentreferencedate(sample_time)

        events = [ev for ev in data
                  if ((sample_startdate < ev.date <= sample_time) and
                      ev.depth > 2. and ev.seis_type == 'LP')]
        val = inertiated_seis_events(events, sample_time, self.inertia_duration)
        return val, None


# Numero(*)(**) di VLP/ULP, [ev/giorno]
# By now not in the databases! Always return 0
class SeismicVLPULPDailyRatio(Parameter):

    _parameter_identity = 'SeismicVLPULPDailyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    def sample_value(self, sample_time, data, map_model):
        # Still not in origin database
        return 0, None


# Numero(*)(**)(***) di VLP/ULP a prof > 2.0 km, [ev/mese]
# By now not in the databases! Always return 0
class SeismicDeepVLPULPMonthlyRatio(Parameter):

    _parameter_identity = 'SeismicDeepVLPULPMonthlyRatio'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity

    def sample_value(self, sample_time, data, map_model):
        # Still not in origin database
        return 0, None


# Superclass for most manual parameters, maybe useless but just ready
class ManualParameter(Parameter):

    def __init__(self):
        Parameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=30)
        self.polymorphicIdentity = self._parameter_identity


# Tremore (ultimo mese), YES/NO
class ManualMonthlyTremor(ManualParameter):

    _parameter_identity = 'ManualMonthlyTremor'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }


# Tremore profondo (>3.5 km) (ultimo mese), YES/NO
class ManualDeepMonthlyTremor(ManualParameter):

    _parameter_identity = 'ManualDeepMonthlyTremor'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }


# Estensione strutture degassamento e/o aumento flussi (ultimo mese), YES/NO
class ManualDegas(ManualParameter):

    _parameter_identity = 'ManualDegas'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }


# Presenza gas acidi: HF - HCl - SO2 (ultima settimana), YES/NO
class ManualAcidGasPresence(ManualParameter):

    _parameter_identity = 'ManualAcidGasPresence'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        ManualParameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=7)


# Variazione della frazione della componente magmatica (ultimo mese), YES/NO
class ManualMagmaticVariation(ManualParameter):

    _parameter_identity = 'ManualMagmaticVariation'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }


# Variazione composizione dei gas (ultimo mese), YES/NO
class ManualGasCompositionVariation(ManualParameter):

    _parameter_identity = 'ManualGasCompositionVariation'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }


# Accelerazione RSAM (ultima settimana), YES/NO
class ManualRSAMAcc(ManualParameter):

    _parameter_identity = 'ManualRSAMAcc'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        ManualParameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=7)


# Accelerazione del numero di eventi sismici (ultima settimana), YES/NO
class ManualSeismicAcc(ManualParameter):

    _parameter_identity = 'ManualSeismicAcc'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        ManualParameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=7)


# Accelerazione del energia sismica rilasciata (ultima settimana), YES/NO
class ManualSeismicEnergyAcc(ManualParameter):

    _parameter_identity = 'ManualSeismicEnergyAcc'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        ManualParameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=7)


# Nuove fratture (significative) (ultimi 3 mesi), YES/NO
class ManualNewFractures(ManualParameter):

    _parameter_identity = 'ManualNewFractures'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        ManualParameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=90)
        self.polymorphicIdentity = self._parameter_identity


# Nuove sorgenti (idrotermali) (ultima settimana), YES/NO
class ManualNewIdroThermalSources(ManualParameter):
    # Maybe I could avoid this duplication and directly use class name
    _parameter_identity = 'ManualNewIdroThermalSources'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        ManualParameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=7)


# Attivita freatica (principale) (ultima settimana), YES/NO
class ManualPhreaticActivity(ManualParameter):

    _parameter_identity = 'ManualPhreaticActivity'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        ManualParameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=7)


# Improvviso stop sismicita' e/o deformazione (ultima settimana), YES/NO
class ManualSeismicStop(ManualParameter):

    _parameter_identity = 'ManualSeismicStop'
    __mapper_args__ = {
        'polymorphic_identity': _parameter_identity
    }

    def __init__(self):
        ManualParameter.__init__(self)
        self._inertia_duration = datetime.timedelta(days=7)
