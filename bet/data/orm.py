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

import datetime
import math
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, \
    Float, orm, Numeric, Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property

__author__ = 'Marco'

# Numeric precision for Numeric/Decimal types

NUM_PREC = 15
NUM_SCALE = 5
Base = declarative_base()


class BETTable(object):
    __table_args__ = {'mysql_charset': 'utf8mb4',
                      'mysql_collate': 'utf8mb4_unicode_ci'}
    pass


# utility for enum creation (see parameter class)
class Period(object):
    MINUTE = 1
    HOUR = 2
    DAY = 3
    MONTH = 4


class Elicitation(Base, BETTable):
    __tablename__ = 'elicitation'
    _id = Column('id', Integer, primary_key=True)
    _elicitation_number = Column('elicitation_number', Integer, nullable=False,
                                unique=True)
    _description = Column('description', String(100),
                          nullable=True)
    _volcano_id = Column('volcano_id', Integer,
                                ForeignKey('volcano.id'),
                                nullable=False)

    # TODO: (_elicitationNumber, _volcano_idVolcano) should be unique

    # _parameters = relationship("Parameter",
    #                            backref="elicitation")
    #  child = relationship("Child", backref="parent_assocs")

    _nodes = relationship("ElicitationNode",  backref="elicitations")

    _volcano = relationship("Volcano", backref=backref('elicitation'))

    def __repr__(self):
        return "<Elicitation(_elicitationNumber='%s', _description='%s')>" % (
            self._elicitation_number, self._description)

    @hybrid_property
    def elicitation_number(self):
        return self._elicitation_number

    @elicitation_number.setter
    def elicitation_number(self, number):
        self._elicitation_number = number

    @hybrid_property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @hybrid_property
    def volcano(self):
        return self._volcano

    @volcano.setter
    def volcano(self, volcano):
        self._volcano = volcano

    @hybrid_property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes


class ElicitationNode(Base):
    __tablename__ = 'elicitation_node'
    _elicitation_id = Column('elicitation_id', Integer,
                             ForeignKey('elicitation.id'), primary_key=True)
    _node_id = Column('node_id', Integer, ForeignKey('node.id'),
                      primary_key=True)
    _order = Column('order', Integer)
    node = relationship("Node", backref="elicitation_assoc")

    @hybrid_property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = order


class Volcano(Base, BETTable):
    __tablename__ = 'volcano'
    _id = Column('id', Integer, primary_key=True)
    _description = Column('description', String(100),
                          nullable=False)
    # TODO: (_description) should be unique

    _elicitations = relationship("Elicitation",
                                 backref="volcano")

    def __repr__(self):
        return "<Volcano(_description='%s')>" % (self._description)

    @hybrid_property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @hybrid_property
    def elicitations(self):
        return self._elicitations

    @elicitations.setter
    def elicitations(self, elicitations):
        self._elicitations = elicitations


class Metaparameter(Base, BETTable):
    __tablename__ = 'metaparameter'
    _id = Column('id', Integer, primary_key=True)
    _description = Column('description', String(100), nullable=False)
    _metaparameter_class = Column('metaparameter_class', String(64),
                                  nullable=False)
    _remotesource_id = Column('remotesource_id',
                              Integer, ForeignKey('remoteSource.id'),
                              nullable=True)
    _elicitation_id = Column('elicitation_id', Integer,
                             ForeignKey('elicitation.id'),
                             nullable=False)

    _remoteSource = relationship("RemoteSource",
                                 backref=backref('metaparameter'))
    _parameters = relationship("Parameter", backref="metaparameter")
    _elicitation = relationship("Elicitation", backref=backref('metaparameter'))


    # i do not need a list of the "BackupElements", it would be too big
    # TODO: (_polymorphicIdentity, _elicitation_idElicitation) should be unique

    # sqlalchemy single table inheritance
    __mapper_args__ = {
        'polymorphic_on': _metaparameter_class
    }

    def __repr__(self):
        return """<Metaparameter(_description='%s', _metaparameter_class='%s',
        _remotesource_id='%s', _elicitation_id='%s')>""" % (
            self._description, str(self._metaparameter_class),
            str(self._remotesource_id),
            str(self._elicitation_id))

    @hybrid_property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @hybrid_property
    def polymorphic_identity(self):
        return self._metaparameter_class

    @polymorphic_identity.setter
    def polymorphic_identity(self, identity):
        self._metaparameter_class = identity

    @hybrid_property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    @hybrid_property
    def elicitation(self):
        return self._elicitation

    @elicitation.setter
    def elicitation(self, elicitation):
        self._elicitation = elicitation

    @hybrid_property
    def remotesource(self):
        return self._remoteSource

    @remotesource.setter
    def remotesource(self, remote):
        self._remoteSource = remote


class Parameter(Base, BETTable):
    __tablename__ = 'parameter'
    _id = Column('id', Integer, primary_key=True)
    _description = Column('description', String(255), nullable=False)
    _parameter_class = Column('parameter_class', String(64), nullable=False)
    _parameter_family = Column('parameter_family', String(64), nullable=True)
    _metaparameter_id = Column('metaparameter_id',
                               Integer, ForeignKey('metaparameter.id'),
                               nullable=True)
    _metaparameter = relationship("Metaparameter", backref=backref('parameter'))

    __mapper_args__ = {
        'polymorphic_on': _parameter_class
    }

    def __init__(self):
        self._inertia_duration = None

    # sqlalchemy will not call init when constructing objects loaded from
    # the DB. You need this tag in order to perform a init-similar operation
    @orm.reconstructor
    def init_on_load(self):
        self.__init__()

    def __repr__(self):
        return """<Parameter(_description='%s', _parameter_class='%s',
        _metaparameter_id='%s')>""" % (
            self._description, str(self._parameter_class),
            str(self._metaparameter_id))

    def get_currentreferencedate(self, query_date, diff_days=None):
        if diff_days:
            return query_date - diff_days
        else:
            return query_date - self.inertia_duration

    def sample_value(self, map_model, data, sample_time):
        return float('nan'), None
        # raise NotImplementedError("This class is to be considered
        # not-instantiable.")

    @property
    def inertia_duration(self):
        return self._inertia_duration

    @hybrid_property
    def description(self):
        return self._description

    @hybrid_property
    def parameter_id(self):
        return self._id

    @description.setter
    def description(self, description):
        self._description = description

    @hybrid_property
    def polymorphic_identity(self):
        return self._parameter_class

    @polymorphic_identity.setter
    def polymorphic_identity(self, identity):
        self._parameter_class = identity

    @hybrid_property
    def parameter_family(self):
        return self._parameter_family

    @parameter_family.setter
    def parameter_family(self, parameter_family):
        self._parameter_family = parameter_family

    @hybrid_property
    def metaparameter(self):
        return self._metaparameter

    @metaparameter.setter
    def metaparameter(self, metaparameter):
        self._metaparameter = metaparameter


class Node(Base, BETTable):
    __tablename__ = 'node'
    _id = Column('id', Integer, primary_key=True)
    _name = Column('name', String(50), nullable=False)

    _parameters = relationship("NodeParameter",  backref="nodes")

    def __repr__(self):
        return "<Node(_name='%s')>" % \
               (self._name)

    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @hybrid_property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters


class NodeParameter(Base):
    __tablename__ = 'node_parameter'
    _node_id = Column('node_id', Integer,
                  ForeignKey('node.id'), primary_key=True)
    _parameter_id = Column('parameter_id', Integer, ForeignKey('parameter.id'),
                       primary_key=True)
    _order = Column('order', Integer)
    _threshold1 = Column('threshold1', Numeric(precision=NUM_PREC,
                                               scale=NUM_SCALE), nullable=True)
    _threshold2 = Column('threshold2', Numeric(precision=NUM_PREC,
                                               scale=NUM_SCALE), nullable=True)
    _relation = Column('relation', String(10), nullable=True)
    _weight = Column('weight', Numeric(precision=NUM_PREC, scale=NUM_SCALE),
                     nullable=True)
    parameter = relationship("Parameter", backref="parameter_assoc")

    def __repr__(self):
        return """<NodeParameter(_node_id='%s', _parameter_id='%s', _order='%s',
        _relation='%s', _threshold1='%s', _threshold2='%s',
        _weight='%s')>""" % (
            self._node_id, self._parameter_id, self._order,
            self._relation, self._threshold1, self._threshold2, self._weight,
            )

    @hybrid_property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = order

    @hybrid_property
    def threshold1(self):
        return self._threshold1

    @threshold1.setter
    def threshold1(self, threshold1):
        self._threshold1 = threshold1

    @hybrid_property
    def threshold2(self):
        return self._threshold2

    @threshold2.setter
    def threshold2(self, threshold2):
        self._threshold2 = threshold2

    @hybrid_property
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, relation):
        self._relation = relation

    @hybrid_property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight



class RemoteSource(Base, BETTable):
    __tablename__ = 'remoteSource'
    _id = Column('id', Integer, primary_key=True)
    _name = Column('name', String(50), nullable=False)
    _description = Column('description', String(100), nullable=True)
    _host = Column('host', String(100), nullable=False)
    _port = Column('port', Integer, nullable=True)
    _type = Column('type', String(50), nullable=False)
    _user = Column('user', String(50), nullable=True)
    _password = Column('password', String(50), nullable=True)
    # TODO: (_remoteHost, _remoteUser, _remoteSourceName) should be unique,
    # but maybe not convenient (inefficient)

    _metaparameters = relationship("Metaparameter", backref="remoteSource")

    def __repr__(self):
        return "<Source(_host='%s', _type='%s',_port='%s', " \
               "_user='%s', _password='%s', _name='%s', " \
               "_description='%s')>" % ( self._host, self._type,
                                         str(self._port), self._user,
                                         self._password, self._name,
                                         self._description)

    @hybrid_property
    def host(self):
        return self._host

    @host.setter
    def host(self, remote_host):
        self._host = remote_host

    @hybrid_property
    def type(self):
        return self._type

    @type.setter
    def type(self, source_type):
        self._type = source_type

    @hybrid_property
    def port(self):
        return self._port

    @port.setter
    def port(self, remote_port):
        self._port = remote_port

    @hybrid_property
    def user(self):
        return self._user

    @user.setter
    def user(self, remote_user):
        self._user = remote_user

    @hybrid_property
    def password(self):
        return self._password

    @password.setter
    def password(self, remote_password):
        self._password = remote_password

    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name(self, source_name):
        self._name = source_name

    @hybrid_property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @hybrid_property
    def metaparameters(self):
        return self._metaparameters

    @metaparameters.setter
    def metaparameters(self, metaparameters):
        self._metaparameters = metaparameters


class MapModel(Base, BETTable):
    __tablename__ = 'mapmodel'
    _id = Column('id', Integer, primary_key=True)
    _name = Column('name', String(50), nullable=False)
    _map_class = Column('map_class', String(64), nullable=False)
    _map_parameters = Column('map_parameters', postgresql.ARRAY(Float),
                             nullable=True)
    _volcano_id = Column('volcano_id', Integer,
                         ForeignKey('volcano.id'),
                         nullable=False)

    # TODO: (_name, _volcano_idVolcano ) should be unique
    _volcano = relationship("Volcano", backref=backref('mapmodel'))
    _geographic_samples = relationship("GeographicSample",
                                      backref="mapmodel")

    # sqlalchemy single table inheritance
    __mapper_args__ = {
        'polymorphic_on': _map_class
    }

    def __repr__(self):
        return "<MapModel(_name='%s', _map_class='%s', _map_parameters='%s', " \
               "_volcano_idVolcano='%s')>" % (self._name,
                                              str(self._map_class),
                                              self._map_parameters,
                                              str(self._volcano_id))

    # useful for subclasses
    def __init__(self):
        pass

    # sqlalchemy will not call init when constructing objects loaded from
    # the DB. You need this tag in order to perform a init-similar operation
    @orm.reconstructor
    def init_on_load(self):
        self.__init__()

    # localize function = x,y must be in a normalized reference system,
    # x and y between [0,1] (x growing left to right, y growing bottom to top)
    def localize_point(self, x, y):
        raise NotImplementedError(
            "This class is to be considered not-instantiable.")

    # private method. Convert a point the current mapModel reference system.
    # x,y must be in a normalized reference system, x and y between [0,1]
    # (x growing left to right, y growing bottom to top)
    def convert_to_refsystem(self, x, y):
        raise NotImplementedError(
            "This class is to be considered not-instantiable.")

    # will build a new model (with new samples calculated basing on kwargs
    # parameters (defined by the model itself).
    def build_samples(self, **kwargs):
        raise NotImplementedError(
            "This class is to be considered not-instantiable.")

    # check the consistency of the model (gemoteric checks). Used internally
    # to be safe about the points and parameters loaded from the database.
    def check_integrity(self):
        raise NotImplementedError(
            "This class is to be considered not-instantiable.")

    @staticmethod
    def distance(x1, x2, y1, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    # protected
    @staticmethod
    def seq(start, stop, step=1):
        n = int(round((stop - start) / float(step)))
        if n > 1:
            return ([start + step * i for i in range(n + 1)])
        else:
            return ([])

    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @hybrid_property
    def polymorphic_identity(self):
        return self._map_class

    @polymorphic_identity.setter
    def polymorphic_identity(self, identity):
        self._map_class = identity

    @hybrid_property
    def volcano(self):
        return self._volcano

    @volcano.setter
    def volcano(self, volcano):
        self._volcano = volcano

    @hybrid_property
    def map_parameters(self):
        return self._map_parameters

    @map_parameters.setter
    def map_parameters(self, parameters):
        self._map_parameters = parameters

    @hybrid_property
    def geographic_samples(self):
        return self._geographic_samples

    @geographic_samples.setter
    def geographic_samples(self, samples):
        self._geographic_samples = samples


class GeographicSample(Base, BETTable):
    __tablename__ = 'geographicsample'
    _id = Column('id', Integer, primary_key=True)
    _index = Column('index', Integer, nullable=False)
    _x_reference = Column('x_reference',
                          Numeric(precision=NUM_PREC, scale=NUM_SCALE),
                          nullable=False)
    _y_reference = Column('y_reference',
                          Numeric(precision=NUM_PREC, scale=NUM_SCALE),
                          nullable=False)
    _sample_parameters = Column('sample_parameters', String(1000),
                                nullable=True)
    _mapmodel_id = Column('mapmodel_id', Integer,
                          ForeignKey('mapmodel.id'),
                          nullable=False)

    _mapmodel = relationship("MapModel", backref=backref('geographicsample'))

    # TODO: (_index, _mapModel_idMapModel) should be unique

    def __repr__(self):
        return "<GeographicSample(_index='%s', _mapmodel_id='%s', " \
               "_x_reference='%s', _y_reference='%s', " \
               "_sample_parameters='%s')>" % (
        str(self._index), str(self._mapmodel_id), str(self._x_reference),
        str(self._y_reference), self._sample_parameters)

    @hybrid_property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @hybrid_property
    def x_reference(self):
        return self._x_reference

    @x_reference.setter
    def x_reference(self, x):
        self._x_reference = x

    @hybrid_property
    def y_reference(self):
        return self._y_reference

    @y_reference.setter
    def y_reference(self, y):
        self._y_reference = y

    @hybrid_property
    def mapmodel(self):
        return self._mapmodel

    @mapmodel.setter
    def mapmodel(self, map):
        self._mapmodel = map

    @hybrid_property
    def sample_parameters(self):
        return self._sample_parameters

    @sample_parameters.setter
    def sample_parameters(self, parameters):
        self._sample_parameters = parameters


class User(Base, BETTable):
    __tablename__ = 'user'
    _id = Column('id', Integer, primary_key=True)
    _first_name = Column('first_name', String(50), nullable=False)
    _last_name = Column('last_name', String(50), nullable=False)
    _email = Column('email', String(50), nullable=False)
    _password = Column('password', String(50), nullable=False)
    # TODO: (_email) should be unique, but maybe not convenient (inefficient)

    def __repr__(self):
        return "<User(first name='%s', last name='%s', _email='%s', " \
               "_password='%s')>" % (
            self._first_name, self._last_name, self._email, self._password)

    @hybrid_property
    def first_name(self):
        return self._first_name

    @first_name.setter
    def first_name(self, name):
        self._first_name = name

    @hybrid_property
    def last_name(self):
        return self._last_name

    @last_name.setter
    def last_name(self, name):
        self._last_name = name

    @hybrid_property
    def email(self):
        return self._email

    @email.setter
    def email(self, email):
        self._email = email

    @hybrid_property
    def password(self):
        return self._password

    @password.setter
    def password(self, password):
        self._password = password


class Run(Base, BETTable):
    __tablename__ = 'run'
    _id = Column('id', Integer, primary_key=True)
    _rundir = Column('rundir', String(255), nullable=False)
    _timestamp = Column('timestamp', DateTime(timezone=True),
                        default=datetime.datetime.now,
                        nullable=False)
    _input_parameters = Column('input_parameters', postgresql.JSON,
                              nullable=False)
    _output = Column('output', postgresql.JSON, nullable=True)
    _ef_unrest = Column('ef_unrest', postgresql.ARRAY(Float),
                               nullable=True)
    _ef_magmatic = Column('ef_magmatic', postgresql.ARRAY(Float),
                               nullable=True)
    _ef_eruption = Column('ef_eruption', postgresql.ARRAY(Float),
                               nullable=True)
    _ef_vent_map = Column('ef_vent_map', postgresql.ARRAY(Float),
                          nullable=True)
    _user_id = Column('user_id', Integer, ForeignKey('user.id'),
                          nullable=True)
    _runmodel_id = Column('runmodel_id', Integer,
                          ForeignKey('runmodel.id'),
                          nullable=False)
    _mapmodel_id = Column('mapModel_id', Integer,
                          ForeignKey('mapmodel.id'),
                          nullable=True)

    _user = relationship("User", backref=backref('run'))
    _runmodel = relationship("RunModel", backref=backref('runmodel'))
    _mapmodel = relationship("MapModel", backref=backref('mapmodel'))

    def __repr__(self):
        return "<Run(_timeStamp='%s', _rundir = %s, _output='%s', " \
               "_user_idUser='%s', " \
               "_runModel_idRunModel='%s', _mapModel_idMapModel='%s')>" % \
               (self._timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                self._rundir, self._output,
                str(self._user_id), str(self._runmodel_id),
                str(self._mapmodel_id))

    @hybrid_property
    def id(self):
        return self._id

    @hybrid_property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, time):
        self._timestamp = time

    @hybrid_property
    def rundir(self):
        return self._rundir

    @rundir.setter
    def rundir(self, rundir):
        self._rundir = rundir

    @hybrid_property
    def output(self):
        return self._output

    @output.setter
    def output(self, output):
        self._output = output

    @hybrid_property
    def user(self):
        return self._user

    @user.setter
    def user(self, user):
        self._user = user

    @hybrid_property
    def runmodel(self):
        return self._runmodel

    @runmodel.setter
    def runmodel(self, model):
        self._runmodel = model

    @hybrid_property
    def mapmodel(self):
        return self._mapmodel

    @mapmodel.setter
    def mapmodel(self, model):
        self._mapmodel = model

    @hybrid_property
    def input_parameters(self):
        return self._input_parameters

    @input_parameters.setter
    def input_parameters(self, parameters):
        self._input_parameters = parameters

    @hybrid_property
    def ef_unrest(self):
        return self._ef_unrest

    @ef_unrest.setter
    def ef_unrest(self, ef_unrest):
        self._ef_unrest = ef_unrest

    @hybrid_property
    def ef_magmatic(self):
        return self._ef_magmatic

    @ef_magmatic.setter
    def ef_magmatic(self, ef_magmatic):
        self._ef_magmatic = ef_magmatic

    @hybrid_property
    def ef_eruption(self):
        return self._ef_eruption

    @ef_eruption.setter
    def ef_eruption(self, ef_eruption):
        self._ef_eruption = ef_eruption

    @hybrid_property
    def ef_vent_map(self):
        return self._ef_vent_map

    @ef_vent_map.setter
    def ef_vent_map(self, ef_vent_map):
        self._ef_vent_map = ef_vent_map


class RunModel(Base, BETTable):
    __tablename__ = 'runmodel'
    _id = Column('id', Integer, primary_key=True)
    _run_class = Column('run_class', String(100), nullable=False)

    _runs = relationship("Run", backref="runmodel")

    # TODO: className should be unique

    def __repr__(self):
        return "<Model(_className='%s'>" % (self._run_class)

    @hybrid_property
    def class_name(self):
        return self._run_class

    @class_name.setter
    def class_name(self, name):
        self._run_class = name

    @hybrid_property
    def runs(self):
        return self._runs

    @runs.setter
    def runs(self, runs):
        self._runs = runs
