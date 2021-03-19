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


from bet.function import get_logger
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, \
    Float, join, DateTime, ForeignKey
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, object_session
from sqlalchemy.orm.util import has_identity
from contextlib import contextmanager

__author__ = 'Marco'


def filter_conn_string(conn_string):
    return ":".join(conn_string.split('@')[0].split(':')[0:-1]) + \
           ":***@" + conn_string.split('@')[1]


class DbManager(object):
    def __init__(self, db_type='mysql', db_host=None, db_port=None,
                 db_user=None, db_password=None, db_name=None, debug=False):
        """
        Connecting to server
        """

        self._dialect = db_type
        self._host = db_host
        self._port = str(db_port)
        self._user = db_user
        self._passwd = db_password
        self._debug = debug
        self._engine = None
        self._name = db_name
        self._Session = None
        self._connect_args = dict()

        if self._dialect == 'mysql':
            self._connect_args['charset']='utf8'
        elif self._dialect == 'postgres':
            self._connect_args['client_encoding']='utf8'
        elif self._dialect == 'mssql+pymssql':
            pass
        else:
            raise NotImplementedError("DB dialect is not supported")


    def connect(self, dbname=None):
        logger = get_logger(__name__)
        try:
            if dbname is not None:
                connect_string = (self._dialect + '://' + self._user + ':' +
                                  self._passwd + '@' + self._host + ':' +
                                  self._port + '/' + dbname)
            else:
                connect_string = (self._dialect + '://' + self._user + ':' +
                                  self._passwd + '@' + self._host + ':' +
                                  self._port)

            logger.debug("Connecting: {}".format(
                    filter_conn_string(connect_string)))
            self._engine = create_engine(connect_string,
                                         connect_args=self._connect_args,
                                         echo=self._debug)
            # factory for Session objects
            self._Session = sessionmaker(bind=self._engine)
            return True
        except:
            raise

    def exists_db(self, dbname):
        logger = get_logger()
        if dbname is not None:
            connect_string = (self._dialect + '://' + self._user + ':' +
                              self._passwd + '@' + self._host + ':' +
                              self._port + '/' + dbname)
            try:
                logger.debug("Connecting: {}".format(
                        filter_conn_string(connect_string)))
                _engine = create_engine(connect_string,
                                        connect_args=self._connect_args,
                                        echo=self._debug)
                tmp_conn = _engine.connect()
                tmp_conn.close()
                return True
            except OperationalError:
                return False
        return False

    def use_db(self, dbname):
        if dbname is not None:
            if self._dialect == 'mysql':
                if self._engine is None:
                    # Return False if object cannot connect to db
                    if not self.connect():
                        return False
                try:
                    sqlquery = "USE %s"
                    sqlquery %= dbname
                    self._engine.execute(sqlquery)
                    self._name = dbname
                    return True
                except:
                    raise
            # To connect a postgres dialect, object must set up a new engine
            elif self._dialect == 'postgres':
                if not self.connect(dbname=dbname):
                    return False
                self._name = dbname
                return True
            elif self._dialect == 'mssql+pymssql':
                if not self.connect(dbname=dbname):
                    return False
                self._dbName = dbname
                return True
            else:
                raise NotImplementedError("DB dialect is not supported")
        else:
            raise Exception("dbname is not defined")

    # TODO: maybe init should be limited to postgres?
    def init_and_use_db(self, dbname, metadata=None):
        if dbname is not None:
            if self._dialect == 'postgres':
                try:
                    if not self.exists_db(dbname):
                        connect_string = (self._dialect + '://' +
                                          self._user + ':' +
                                          self._passwd + '@' +
                                          self._host + ':' +
                                          self._port + '/postgres')
                        tmp_engine = create_engine(
                            connect_string,
                            connect_args=self._connect_args,
                            echo=self._debug)
                        tmp_conn = tmp_engine.connect()
                        tmp_conn.execute("commit")
                        tmp_conn.execute("CREATE DATABASE %s" % dbname)
                        tmp_conn.close()
                    connect_string = (self._dialect + '://' + self._user + ':' +
                                      self._passwd + '@' + self._host + ':' +
                                      self._port + '/' + dbname)
                    self._engine = create_engine(
                        connect_string,
                        connect_args=self._connect_args,
                        echo=self._debug)

                    self._Session = sessionmaker(bind=self._engine)
                    self._name = dbname
                    if metadata is not None:
                        # Create all tables in the engine. This is equivalent to
                        # "Create Table" statements in raw SQL.
                        metadata.create_all(self._engine)
                    return True
                except Exception as e:
                    raise e
            else:
                raise NotImplementedError("Only postgres is supported as "
                                          "main database")

    @property
    def dbName(self):
        return self._name

    @property
    def metadata(self):
        return MetaData(self._engine)

    #use for transactional queries!
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self._Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def getObjectState(self, obj):

        # transient:
        if object_session(obj) is None and not has_identity(obj):
            objectState = "transient"
        # pending:
        elif object_session(obj) is not None and not has_identity(obj):
            objectState = "pending"
        # detached:
        elif object_session(obj) is None and has_identity(obj):
            objectState = "detached"
        # persistent:
        elif object_session(obj) is not None and has_identity(obj):
            objectState = "persistent"
        else:
            objectState = "unknown"

        return objectState

#http://sqlalchemy.narkive.com/1odMujBg/multiple-calls-to-create-engine-with-the-same-connection-string
