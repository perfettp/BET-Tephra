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

from bet.function import get_logger
from collections import MutableSequence
import simplejson as json
from datetime import datetime
from configobj import ConfigObj
from bet.data import VentProb, VentProbList, UTMPoint, StyleProb
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding('utf8')


class BetConf(ConfigObj):

    def __init__(self, *args, **kwargs):
        logger = get_logger()
        logger.debug("Declaring a new BetConf object")
        self._obs_time = kwargs.pop('obs_time', None)
        super(BetConf, self).__init__(*args, **kwargs)
        self._vent_grid = None
        self._vent_grid_n = None
        self._hazard_grid = None
        self._hazard_grid_n = None
        self._tephra_grid = None
        self._tephra_grid_n = None
        self._styles_grid = None
        self._local_conf = None
        self._run_db_id = None
        self.merge_local_conf()


    def to_json(self, **kwargs):
        return json.dumps(self.main, ensure_ascii=False,
                          sort_keys=True, **kwargs)

    @classmethod
    def from_json(cls, j_dump, **kwargs):
        a = cls()
        a.main.update(json.loads(j_dump, **kwargs))
        return a

    @classmethod
    def from_dict(cls, o_dict):
        a = cls()
        a.main.update(o_dict)
        return a

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(self).__name__, name))

    def merge_local_conf(self):
        self._local_conf = ConfigObj(self.Files['include_conf'])
        self.merge(self._local_conf)

    def load_vent_grid(self):
        self._vent_grid = VentProbList()
        with open(self.BET['Vents']['vents_grid_prob_file'], 'r') \
                as grid_file:
            i_loc = 1
            for l in grid_file:
                east, north, prior, past_data = \
                    [x for x in l.strip(' \t\n\r').split(' ')
                               if x is not ""]
                p = VentProb(loc=i_loc,
                             point=UTMPoint(easting=float(east.strip()),
                                            northing=float(north.strip())),
                             prior=float(prior.strip()),
                             past_data=float(past_data.strip()))
                self._vent_grid.append(p)
                i_loc += 1

        self._vent_grid_n = len(self._vent_grid)

    def load_style_grid(self):
        self._styles_grid = StyleProb(sizes=self.BET['Styles']['sizes'])
        sizes = self.BET['Styles']['sizes']
        with open(self.BET['Styles']['size_prob_file'], 'r') \
                as style_file:
            prior_arr = dict((k, list()) for k in sizes)
            past_data_arr = dict((k, list()) for k in sizes)
            lambda_arr = list()
            for l in style_file:
                i_element = 0
                l_arr = l.split(" ")
                for s in sizes:
                    prior_arr[s].append(float(l_arr[i_element].strip()))
                    i_element += 1
                lambda_arr.append(float(l_arr[i_element].strip()))
                i_element += 1
                for s in sizes:
                    past_data_arr[s].append(float(l_arr[i_element].strip()))
                    i_element += 1
            self._styles_grid.set_lambda(lambda_arr)
            self._styles_grid.set_past_data(past_data_arr)
            self._styles_grid.set_prior(prior_arr)

    def load_hazard_grid(self):
        self._hazard_grid = list()
        with open(self.BET['Hazard']['grid_file_utm'], 'r') \
                as grid_file:
            for l in grid_file:
                east, north = [x for x in l.strip(' \t\n\r').split(',')
                               if x is not ""]
                self._hazard_grid.append(
                    UTMPoint(easting=float(east.strip()),
                             northing=float(north.strip())))
        self._hazard_grid_n = len(self._hazard_grid)

    def load_tephra_grid(self):
        self._tephra_grid = list()
        with open(self.BET['Tephra']['grid_file_utm'], 'r') \
                as grid_file:
            for l in grid_file:
                east, north = [x for x in l.strip(' \t\n\r').split(' ')
                               if x is not ""]
                self._tephra_grid.append(
                    UTMPoint(easting=float(east.strip()),
                             northing=float(north.strip())))
        self._tephra_grid_n = len(self._tephra_grid)

    @property
    def run_db_id(self):
        return self._run_db_id

    @run_db_id.setter
    def run_db_id(self, val):
        self._run_db_id = val

    @property
    def obs_time(self):
        return self._obs_time

    @obs_time.setter
    def obs_time(self, val):
        self._obs_time = val

    @property
    def vent_grid(self):
        return self._vent_grid

    @property
    def vent_grid_n(self):
        return self._vent_grid_n

    @property
    def styles_grid(self):
        return self._styles_grid

    @property
    def hazard_grid(self):
        return self._hazard_grid

    @property
    def hazard_grid_n(self):
        return self._hazard_grid_n

    @property
    def tephra_grid(self):
        return self._tephra_grid

    @property
    def tephra_grid_n(self):
        return self._tephra_grid_n


class MonitoringConf(MutableSequence):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', None)
        self._elicitation_conf = kwargs.pop('elicitation_conf', None)
        self._date = kwargs.pop('date', None)
        self._nodes = list(*args, **kwargs)

    def __hash__(self):
        return self._nodes.__hash__()

    def __contains__(self, item):
        return self._nodes.__contains__(item)

    def __iter__(self):
        return self._nodes.__iter__()

    def __len__(self):
        return self._nodes.__len__()

    def __call__(self, *args, **kwargs):
        self._nodes = list(*args, **kwargs)

    def __getitem__(self, item):
        return self._nodes.__getitem__(item)

    def __setitem__(self, key, value):
        return self._nodes.__setitem__(key, value)

    def __delitem__(self, key):
        return self._nodes.__delitem__(key)

    def insert(self, index, value):
        return self._nodes.insert(index, value)

    def __repr__(self):
        s = "MonitoringConf<"
        s += self.elicitation_conf.__repr__()
        for node in self._nodes:
            s += node.__repr__()
        s += ">"
        return s

    def to_json(self, **kwargs):
        return json.dumps(self.dict_ser(), ensure_ascii=False, sort_keys=True, **kwargs)

    def from_json(self, c_ser, **kwargs):
        self.from_dict(json.loads(c_ser, **kwargs))

    def dict_ser(self):

        node_list = []
        for node in self._nodes:
            node_list.append(node.dict_ser())

        c_dict = {'name': self._name,
                  'elicitation_conf': self._elicitation_conf.dict_ser(),
                  'date': self._date.strftime("%Y-%m-%d %H:%M:%S"),
                  'nodes': node_list}
        return c_dict

    def from_dict(self, d):
        self._name = d.get('name', "")
        self._date = datetime.strptime(d.get('date',""),
                                       "%Y-%m-%d %H:%M:%S")
        self._elicitation_conf = ElicitationConf()
        el_dict = d.get('elicitation_conf', {})
        self._elicitation_conf.from_dict(el_dict)
        self._nodes = []
        for n_dict in d.get('nodes', []):
            n = NodeConf()
            n.from_dict(n_dict)
            self._nodes.append(n)

    @property
    def elicitation_conf(self):
        return self._elicitation_conf

    @elicitation_conf.setter
    def elicitation_conf(self, conf):
        self._elicitation_conf = conf

    @property
    def nodes(self):
        return self._nodes

    @property
    def date(self):
        return self._date


class NodeConf(MutableSequence):

    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', None)
        self._parameters = list(*args, **kwargs)

    def __hash__(self):
        return self._parameters.__hash__()

    def __contains__(self, item):
        return self._parameters.__contains__(item)

    def __iter__(self):
        return self._parameters.__iter__()

    def __len__(self):
        return self._parameters.__len__()

    def __call__(self, *args, **kwargs):
        self._parameters = list(*args, **kwargs)

    def __getitem__(self, item):
        return self._parameters.__getitem__(item)

    def __setitem__(self, key, value):
        return self._parameters.__setitem__(key, value)

    def __delitem__(self, key):
        return self._parameters.__delitem__(key)

    def insert(self, index, value):
        return self._parameters.insert(index, value)

    @property
    def name(self):
        return self._name

    def __repr__(self):
        s = "NodeConf<"
        s += "_name=%s, _parameters: " % self.name
        for param in self._parameters:
            s += param.__repr__()
            s += ", "
        s += ">"
        return s

    def to_json(self, **kwargs):
        return json.dumps(self.dict_ser(), ensure_ascii=False, sort_keys=True, **kwargs)

    def from_json(self, n_ser, **kwargs):
        self.from_dict(json.loads(n_ser, **kwargs))

    def dict_ser(self):
        param_list = []
        for param in self._parameters:
            param_list.append(param.dict_ser())

        n_dict = {'name': self._name,
                  'parameters': param_list}
        return n_dict

    def from_dict(self, d):
        self._name = d.get('name', "")
        self._parameters = []
        for p_dict in d.get('parameters', []):
            p = ParameterConf()
            p.from_dict(p_dict)
            self._parameters.append(p)

    @property
    def parameters(self):
        return self._parameters


class ParameterConf(object):
    def __init__(self, class_type=None, value=None, val_map=None,
                 relation=None, vent_maps_norm = None,
                 threshold_1=None, threshold_2=None,
                 weight=None, parameter_family=""):
        self._name = class_type
        self._relation = relation
        self._thresh1 = float(threshold_1) if threshold_1 is not None \
            else threshold_1
        self._thresh2 = float(threshold_2) if threshold_2 is not None \
            else threshold_2
        self._weight = float(weight) if weight is not None else weight
        self._parameter_family = parameter_family
        self._value = float(value) if value is not None else value
        self._val_map = val_map
        self._vent_maps_norm = vent_maps_norm

    @property
    def name(self):
        return self._name

    @property
    def relation(self):
        return self._relation

    @property
    def value(self):
        return self._value

    @property
    def val_map(self):
        return self._val_map

    @property
    def weight(self):
        return self._weight

    @property
    def threshold_1(self):
        return self._thresh1

    @property
    def threshold_2(self):
        return self._thresh2

    @property
    def parameter_family(self):
        return self._parameter_family

    @property
    def vent_maps_norm(self):
        return self._vent_maps_norm

    @vent_maps_norm.setter
    def vent_maps_norm(self, vals):
        self._vent_maps_norm = vals

    def __repr__(self):
        s = "ParameterConf<_class_type=%s, value=%s, val_map=%s, _thresh1=%s, " \
            "_thresh2=%s, _weight=%s, _relation='%s', _family=%s>" %(
            self.name, self.value, self.val_map, self.threshold_1,
            self.threshold_2, self._weight,
            self._relation, self.parameter_family)
        return s

    def to_json(self, **kwargs):
        return json.dumps(self.dict_ser(), ensure_ascii=False, sort_keys=True, **kwargs)

    def from_json(self, p_ser, **kwargs):
        self.from_dict(json.loads(p_ser, **kwargs))

    def dict_ser(self):

        val_map_list = self._val_map.tolist() \
            if isinstance(self._val_map, np.ndarray) else self._val_map

        return {'name': self._name,
                'value': self._value,
                'val_map': val_map_list,
                'relation': self._relation,
                'threshold1': self._thresh1,
                'threshold2': self._thresh2,
                'weight': self._weight,
                'parameter_family': self.parameter_family}

    def from_dict(self, d):
        self._name = d.get('name', "")
        self._value = d.get('value', float('nan'))
        self._val_map = d.get('val_map', "")
        self._relation = d.get('relation', "")
        self._weight = d.get('weight', "")
        self._thresh1 = d.get('threshold1', "")
        self._thresh2 = d.get('threshold2', "")
        self._parameter_family = d.get('parameter_family', "")


class ElicitationConf(object):

    def __init__(self, **kwargs):
        self._name = kwargs.pop('name', "")
        self._parameters = dict(Seismic=dict(),
                                Deformation=dict(),
                                Geochemical=dict(),
                                Misc=dict())

    def add(self, param):
        self._parameters[param.parameter_family][param.name] = param

    @property
    def name(self):
        return self._name

    def __repr__(self):
        s = "ElicitationConf<"
        s += "_name=%s, _parameters: " % self.name
        for family in self._parameters.keys():
            s += family.__repr__()
            s += ", "
        s += ">"
        return s

    def to_json(self, **kwargs):
        return json.dumps(self.dict_ser(), ensure_ascii=False, sort_keys=True, **kwargs)

    def from_json(self, n_ser, **kwargs):
        self.from_dict(json.loads(n_ser, **kwargs))

    def dict_ser(self):
        name_dict = dict(Seismic=dict(),
                         Deformation=dict(),
                         Geochemical=dict(),
                         Misc=dict(),
                         name=self._name)
        for fam_key in self._parameters.keys():
            for par_key in self._parameters[fam_key].keys():
                name_dict[fam_key][par_key] = self._parameters[fam_key][par_key].dict_ser()
        return name_dict

    def from_dict(self, d):
        self._name = d.pop('name', "")
        self._parameters = dict(Seismic=dict(),
                                Deformation=dict(),
                                Geochemical=dict(),
                                Misc=dict())
        for fam_key in d.keys():
            for pam_key in d[fam_key].keys():
                p = ParameterConf()
                p.from_dict(d[fam_key][pam_key])
                self._parameters[fam_key][pam_key] = p

    @property
    def parameters(self):
        return self._parameters
