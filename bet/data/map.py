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
import collections
import math
from orm import MapModel, GeographicSample
import shapefile

__author__ = 'Marco'


# logic: rectangular grid where x resolution can also be different from y
# resolution (xresolution and yresoluton between points must be constant).
# Very useful (and fast) when you have a lot of samples.Thanks to the grid
# hypotesis it is very easy to localize a query point, there is no need to
# loop all over the samples. the coordinate system seems to be not fixed,
# but the same as the one where the query points come from (is this possible??)
# To build this model from scratch with user defined parameters use
# "buildSamples" method
class GridModel(MapModel):

    _map_identity = 'GridModel'
    __mapper_args__ = {
        'polymorphic_identity': _map_identity
    }

    # will load the samples from the DB. Will load the GridModel structure (
    # from mapmodel table).
    def __init__(self, **kwargs):
        super(GridModel, self).__init__()
        self.polymorphicIdentity = self._map_identity
        if not kwargs:
            self._xResolution = None
            self._yResolution = None
            self._xSize = None
            self._ySize = None
            self._initialized = True
            #i can t call checkintegrity here since it will use a list (geographicSamples) that will be loaded later in a query call..you  may want to do it manually afer the query (it is not needed, it is just a layer of security check)
        else:
            self.buildModel(**kwargs)

    #will build a new model (with new samples calculated basing on kwargs parameters (defined by the model itself).
    def buildModel(self, **kwargs):
        xResolution = kwargs.pop('xResolution', None)
        yResolution = kwargs.pop('yResolution', None)
        xSize = kwargs.pop('xSize', None)
        ySize = kwargs.pop('ySize', None)
        self._initialized = False

        if (self.geographic_samples is None or len(self.geographic_samples) ==
             0) and self._map_parameters is None:
            if xResolution > 0 and yResolution > 0 and xSize > 0 and ySize > 0 and xResolution <= xSize and yResolution <= ySize:

                self.modelParameters = [xResolution, yResolution, xSize, ySize]

                xList = MapModel.seq(0, xSize - 1, xResolution)
                yList = MapModel.seq(0, ySize - 1, yResolution)

                index = 0
                self.geographic_samples = []
                for y in yList:
                    for x in xList:
                        sample = GeographicSample()
                        sample.index = index
                        sample.mapModel = self  #will auto append in the list
                        sample.xReference = x
                        sample.yReference = y

                        index += 1

                self.check_integrity()
            else:
                raise AttributeError("Invalid attributes.")
        else:
            raise AttributeError("You must use an empty model.")

    #check the consistency of the model (gemoteric checks). Used internally to be safe about the points and parameters loaded from the database.
    def check_integrity(self):
        self._initialized = True
        if self.geographic_samples is None or len(self.geographic_samples) == 0:
            self._initialized = False
        else:
            p = self._map_parameters
            if len(p) >= 4:
                xResolution = float(p[0])
                yResolution = float(p[1])
                xSize = float(p[2])
                ySize = float(p[3])
                if xResolution > 0 and yResolution > 0 and xSize > 0 and ySize > 0 and xResolution <= xSize and yResolution <= ySize:
                    self._xResolution = xResolution
                    self._yResolution = yResolution
                    self._xSize = xSize
                    self._ySize = ySize
                else:
                    self._initialized = False
            else:
                self._initialized = False

    #localize function = x,y must be in a normalized reference system, x and y between [0,1] (x growing left to right, y growing bottom to top)
    def localize_point(self, x, y):
        if self._initialized :

            (x, y) = self.convert_to_refsystem(x, y)

            if x >= 0 and x < self._xSize and y >= 0 and y < self._ySize:
                xIndex = math.floor(x / self._xResolution)
                yIndex = math.floor(y / self._yResolution)
                return int(yIndex * self._xSize + xIndex)

        return None

    #private method. Convert a point the current mapModel reference system. x,y must be in a normalized reference system, x and y between [0,1] (x growing left to right, y growing bottom to top)
    def convert_to_refsystem(self, x, y):

        #i only need to rescale from the standard normalized reference system to obtain the one for this mapModel
        mapModelSizeX = float(self._mapModel.map_parameters[2])
        mapModelSizeY = float(self._mapModel.map_parameters[3])
        x =  x * mapModelSizeX
        y =  y * mapModelSizeY
        return (x, y)


#logic: two nested circles; the inner one represent the volcano mouth, the outer one is subdivided in 4 parts (north,east,west,south) by diagonal segments.
#So here you can have only 5 reference points: one is the center (coordinates 0,0; must have index=0), the other 4 are located halfway between the inner and the outer circonferences (in 4 different directions: N S W E)
#0,0 is located in the center; x grows left to right, y grows bottom to top.
#This section subidivision is fixed, to build this model from scratch with user defined parameters use "buildSamples" method
class CardinalSectionsModel(MapModel):

    _map_identity = 'CardinalSectionsModel'
    __mapper_args__ = {
        'polymorphic_identity': _map_identity
    }

    #will load the samples from the DB. Will check that samples agree with CardinalSectionsModel structure. Will estimate current CardinalSectionsModel parameters
    def __init__(self, **kwargs):
        super(CardinalSectionsModel, self).__init__()
        self.polymorphicIdentity = self._map_identity
        if not kwargs:
            self._innerCircleRadius = None
            self._outerCircleRadius = None
            self._center = None
            self._initialized = True
            #i can t call checkintegrity here since it will use a list (geographic_samples) that will be loaded later in a query call..you  may want to do it manually afer the query (it is not needed, it is just a layer of security check)
        else:
            self.buildModel(**kwargs)

    #will build a new model (with new samples calculated basing on kwargs parameters (defined by the model itself).
    def buildModel(self, **kwargs):
        innerCircleRadius = kwargs.pop('innerCircleRadius', None)
        outerCircleRadius = kwargs.pop('outerCircleRadius', None)
        self._initialized = False

        if (self.geographic_samples is None or len(self.geographic_samples) == 0) and self._map_parameters is None:
            if innerCircleRadius > 0 and outerCircleRadius > 0 and outerCircleRadius > innerCircleRadius:

                self.modelParameters = [innerCircleRadius, outerCircleRadius]

                #mouth
                sample = GeographicSample()
                sample.xReference = 0
                sample.yReference = 0
                sample.mapModel = self   #will auto append in the list
                sample.index = 0

                #east
                sample = GeographicSample()
                sample.xReference = + innerCircleRadius + (outerCircleRadius - innerCircleRadius) / 2.0
                sample.yReference = 0
                sample.mapModel = self
                sample.index = 1

                #north
                sample = GeographicSample()
                sample.xReference = 0
                sample.yReference = - innerCircleRadius - (outerCircleRadius - innerCircleRadius) / 2.0
                sample.mapModel = self
                sample.index = 2

                #west
                sample = GeographicSample()
                sample.xReference = - innerCircleRadius - (outerCircleRadius - innerCircleRadius) / 2.0
                sample.yReference = 0
                sample.mapModel = self
                sample.index = 3

                #south
                sample = GeographicSample()
                sample.xReference = 0
                sample.yReference = innerCircleRadius + (outerCircleRadius - innerCircleRadius) / 2.0
                sample.mapModel = self
                sample.index = 4

                self.check_integrity()
            else:
                raise AttributeError("Invalid attributes.")
        else:
            raise AttributeError("You must use an empty model.")

    #check the consistency of the model (gemoteric checks). Used internally to be safe about the points and parameters loaded from the database.
    def check_integrity(self):
        self._initialized = True
        if self.geographic_samples is None or len(self.geographic_samples) != 5:
            self._initialized = False
        else:
            p = self._map_parameters
            if len(p) >= 2:
                innerCircleRadius = float(p[0])
                outerCircleRadius = float(p[1])

                for item in self.geographic_samples:
                    if item.index == 0:
                        center = item
                        break

                if innerCircleRadius > 0 and outerCircleRadius > innerCircleRadius and center is not None and center.x_reference == 0 and center.y_reference == 0:
                    self._innerCircleRadius = innerCircleRadius
                    self._outerCircleRadius = outerCircleRadius
                    self._center = center
                else:
                    self._initialized = False
            else:
                self._initialized = False

    #localize function = x,y must be in a normalized reference system, x and y between [0,1] (x growing left to right, y growing bottom to top)
    def localize_point(self, x, y):
        if self._initialized:

            (x, y) = self.convert_to_refsystem(x, y)

            dist = MapModel.distance(x, self._center.x_reference, y, self._center.y_reference)
            if dist < self._innerCircleRadius:
                return 0
            elif dist > self._outerCircleRadius:
                return None
            else:
                angle = math.atan2(y, x)
                piDividedFour = math.pi / 4
                if angle > -piDividedFour and angle < piDividedFour:  #east   (same indices as in the "buildSamples" method
                    return 1
                elif angle > piDividedFour and angle < 3 * piDividedFour:  #north
                    return 2
                elif angle < -piDividedFour and angle > -3 * piDividedFour:  #south
                    return 4
                else:  #west
                    return 3
        return None

    #private method. Convert a point the current mapModel reference system. x,y must be in a normalized reference system, x and y between [0,1] (x growing left to right, y growing bottom to top)
    def convert_to_refsystem(self, x, y):

        #i make the total model radius outerCircleRadius as the max circle inside the "squared" original normalized reference system.
        #it can also be made larger or smaller.
        #i then move the origin to the center of the circle (to be compatible with this mapModel)
        p = self._map_parameters
        outerCircleRadius = float(p[1])
        x = x * (outerCircleRadius * 2) - outerCircleRadius
        y = y * (outerCircleRadius * 2) - outerCircleRadius
        return (x, y)

#logic: while other mapmodels are built from geometric constraints and rules, this model is imported from a shapefile.
#You only need to define how to build the model (import),  check if the model is not broken, and fill a localizePoint function.
#NOT TESTED!!!!
# class ShapeFileModel(MapModel):
#
#     _map_identity = 'ShapeFileModel'
#     __mapper_args__ = {
#         'polymorphic_identity': _map_identity
#     }
#
#     #will load the samples from the DB. Will check that samples agree with CardinalSectionsModel structure. Will estimate current CardinalSectionsModel parameters
#     def __init__(self, **kwargs):
#         super(ShapeFileModel, self).__init__()
#         self.polymorphicIdentity = self._map_identity
#         if not kwargs:
#             self._filePath = None
#             self._initialized = True
#             #i can t call checkintegrity here since it will use a list (geographic_samples) that will be loaded later in a query call..you  may want to do it manually afer the query (it is not needed, it is just a layer of security check)
#         else:
#             self.buildModel(**kwargs)
#
#     #will build a new model (with new samples calculated basing on kwargs parameters (defined by the model itself).
#     def buildModel(self, **kwargs):
#         filePath = kwargs.pop('filePath', None)
#         self._initialized = False
#         if (self.geographic_samples is None or len(self.geographic_samples) == 0) and self._modelParameters is None:
#             if isinstance(filePath, basestring):
#                 self.modelParameters = str(filePath)
#
#                 #sf = shapefile.Reader(filePath)
#                 #shapes = sf.shapes()
#                 #...
#                 #...
#
#                 #here you have to read the whole file and create many "geographicSample" with assigned a centroid for everyone..  Probably you
#                 #have to use the optional sampleParameters field to save the list of corners. This list will then be used in localizePoint method.
#                 # By assigning "geographicSample.mapModel = self" you will estabilish a list of samples related to the current model.
#                 #In the end you call checkIntegrity to check that the newly built model is not broken.
#                 #Check other classes to see examples.
#
#                 self.checkIntegrity()
#             else:
#                 raise AttributeError("Invalid attributes.")
#         else:
#             raise AttributeError("You must use an empty model.")
#
#     #check the consistency of the model (gemoteric checks). Used internally to be safe about the points and parameters loaded from the database.
#     def checkIntegrity(self):
#         self._initialized = True
#         #do more checks here? Since it is not a model built from geometrical params probably you do not have much checks to do.
#         if self.geographic_samples is None:
#             self._initialized = False
#
#     #localize function = x,y must be in a normalized reference system, x and y between [0,1] (x growing left to right, y growing bottom to top)
#     def localizePoint(self, x, y):
#         if self._initialized:
#             #Here you have to wrap a pyshp function able to return a "sample" (an area) given a query (x,y) point
#             #Then you have to guess the index of the sample inside the "self.geographic_samples" (you built it in buildModel method) and return the index.
#
#             (x, y) = self.convertPointToMapModelRefSystem(x, y)
#
#             return None
#         else:
#             return None
#
#     #private method. Convert a point the current mapModel reference system. x,y must be in a normalized reference system, x and y between [0,1] (x growing left to right, y growing bottom to top)
#     def convertPointToMapModelRefSystem(self, x, y):
#
#         #You have to convert here from standard normalized system to this mapModel system
#         return (x, y)
