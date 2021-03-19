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

import time
import os
import shutil
import logging
from datetime import date
from collections import namedtuple
import subprocess
import numpy
import simplekml
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
from matplotlib.colors import ListedColormap
import utm
import numpy as np
import geojson
from math import pow, sqrt, exp, floor
from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib.pyplot as pyplot
from matplotlib import colors
import scipy.interpolate
from osgeo import gdal
from osgeo import ogr, osr
import re

# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s.%(funcName)s [%("\
#              "process)d]: %(message)s"

LOG_LEVEL = logging.DEBUG

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(pathname)s:%(" \
             "lineno)d [%("\
             "process)d]: %(message)s"

LOG_FORMATTER = logging.Formatter(LOG_FORMAT)

__logger = None
plot_colors={'unrest': 'g',
             'magmatic': '#FF4500',
             'eruption': 'r'}

def create_run_dir(basedir, obs_time, prefix='bet'):
    logger = get_logger()
    seq = 0
    created = False
    if obs_time is not None:
        dir_name = ''
        while not created:
            dir_name = os.path.join(
                    basedir,
                    "{}_{}_{:02d}".format(prefix,
                                      obs_time.strftime("%Y%m%d_%H%M"),
                                          seq))
            if not os.path.exists(dir_name):
                try:
                    os.mkdir(dir_name)
                    created = True
                except OSError as e:
                    logger.exception(e.message)
                    return None
            else:
                seq += 1
        return dir_name
    else:
        raise ValueError("obs_time not specified!")

def find_offset_dirs(dir):
    off_re = re.compile('^[0-9]{2}$')
    return sorted([d for d in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, d)) and off_re.match(d)])


def inertiated_seis_events(events, sample_date, inertia_duration):
    tau = 7.
    val = 0.
    count = 0
    for ev in sorted(events, key=lambda x: x.date, reverse=True):
        val += exp(-1 * (float((sample_date - ev.date).days) / tau))
        count += 1
        if count >= inertia_duration.days:
            return val
    return val


# Return normalized val on inertia duration
def normalized_val(first_sample, last_sample, inertia_duration):
    logger = get_logger()

    try:
        raw_val = last_sample.value - first_sample.value
    except AttributeError:
        logger.error("Invalid first or last sample!")
        return 0
    try:
        norm_val = (raw_val / (last_sample.date - first_sample.date).days) * \
                  inertia_duration.days
    except ZeroDivisionError:
        logger.error("Division by zero normalizing value!")
        return 0
    return norm_val


def select_samples_interval(date, samples, min_diff=10, max_diff=18):
    return [s for s in samples
            if (min_diff <= (s.date - date).days <= max_diff)]


# Max of delta on 15-days base
def max_monthly_delta_15(samples):
    max_diff = 0
    first_sample = None
    second_sample = None

    for ref_sample in samples:
        samples_to_check = select_samples_interval(ref_sample.date,
                                                   samples)
        for diff_sample in samples_to_check:
            tmp_diff = ((ref_sample.value - diff_sample.value) /
                        (ref_sample.date - diff_sample.date).days) * 30
            if abs(tmp_diff) > max_diff:
                max_diff = tmp_diff
                first_sample = ref_sample
                second_sample = diff_sample
    return max_diff, first_sample, second_sample


def chunks(l, n):
    # IMPORTANT: chunks has to be contigous!! Some code rely on this!
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def get_season_number(rundate):
    Season = namedtuple("Season", ('begin', 'end', 'val'))
    seasons = [ Season(date(rundate.year-1, 12, 22), date(rundate.year, 3, 20), 4),
                Season(date(rundate.year, 3, 21), date(rundate.year, 6, 20), 1),
                Season(date(rundate.year, 6, 21), date(rundate.year, 9, 22), 2),
                Season(date(rundate.year, 9, 23), date(rundate.year, 12, 21), 3),
                Season(date(rundate.year, 12, 22), date(rundate.year+1, 3, 20), 4)]
    for s in seasons:
        if s.begin<=rundate.date() <= s.end:
            return s.val


def read_npy_chunk(filename, start_row, num_rows):
    """
    Reads a partial array (contiguous chunk along the first
    axis) from an NPY file.
    Parameters
    ----------
    filename : str
        Name/path of the file from which to read.
    start_row : int
        The first row of the chunk you wish to read. Must be
        less than the number of rows (elements along the first
        axis) in the file.
    num_rows : int
        The number of rows you wish to read. The total of
        `start_row + num_rows` must be less than the number of
        rows (elements along the first axis) in the file.
    Returns
    -------
    out : ndarray
        Array with `out.shape[0] == num_rows`, equivalent to
        `arr[start_row:start_row + num_rows]` if `arr` were
        the entire array (note that the entire array is never
        loaded into memory by this function).
    """
    __author__ = "David Warde-Farley"

    assert start_row >= 0 and num_rows > 0
    with open(filename, 'rb') as fhandle:
        major, minor = numpy.lib.format.read_magic(fhandle)
        shape, fortran, dtype = numpy.lib.format.read_array_header_1_0(fhandle)
        assert not fortran, "Fortran order arrays not supported"
        # Make sure the offsets aren't invalid.
        assert start_row < shape[0], (
            'start_row is beyond end of file'
        )
        assert start_row + num_rows <= shape[0], (
            'start_row + num_rows > shape[0]'
        )
        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        row_size = numpy.prod(shape[1:])
        start_byte = start_row * row_size * dtype.itemsize
        fhandle.seek(start_byte, 1)
        n_items = row_size * num_rows
        flat = numpy.fromfile(fhandle, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])


def read_npy_row(filename, row):
    return numpy.squeeze(read_npy_chunk(filename, row, 1))


def plot_vents(points_utm, vals, img_filename,
               title=None,
               scatter=False,
               basemap_res='h',
               llcrn=None,
               urcrn=None,
               **kwargs):
    logger = get_logger()
    logger.debug("Generating vent probabilities map")
    f = plt.figure(figsize=(8, 4.5), dpi=100)
    ax = f.add_subplot(111)
    # Load grid from static file
    points = [utm.to_latlon(v.easting,
                            v.northing,
                            v.zone_number,
                            v.zone_letter)
              for v in points_utm]

    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])

    logger.info("Map latitudes interval: len {} min {} max {}".format(
            len(lats), lats.min(), lats.max()))
    logger.info("Map longitudes interval: len {} min {} max {}".format(
            len(lons), lons.min(), lons.max()))

    if not llcrn:
        llcrn = (lats.min(), lons.min())
    if not urcrn:
        urcrn = (lats.max(), lons.max())

    logger.info("Maximum load {}".format(vals.max()))

    m = Basemap(
            resolution=basemap_res,
            projection='merc',
            area_thresh=0.1,
            llcrnrlat=llcrn[0], llcrnrlon=llcrn[1],
            urcrnrlat=urcrn[0], urcrnrlon=urcrn[1])

    numcols, numrows = 1000, 1000
    xi = np.linspace(lons.min(), lons.max(), numcols)
    yi = np.linspace(lats.min(), lats.max(), numrows)

    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(lons, lats, vals, xi, yi,  interp='linear')

    im = m.contourf(xi, yi, zi,
                    latlon=True,
                    cmap=plt.cm.rainbow,
                    alpha=0.8,
                    extend='max',
                    zorder=2)

    im.cmap.set_over('black')

    if scatter:
        m.scatter(lons, lats, c=vals, cmap=im.cmap, s=50,
                  # alpha=1,
                  zorder=2, vmin=vals.min(), vmax=vals.max(), latlon=True)


    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines(zorder=6)
    m.drawstates(zorder=6)
    m.drawmapboundary(fill_color='paleturquoise')
    m.fillcontinents(color='grey')
    m.drawcountries()
    m.drawmapscale(xi.min() + (xi.max()-xi.min())/5,
                   yi.min() + (yi.max()-yi.min())/10,
                   (xi.max()-xi.min())/2,
                   (yi.max()-yi.min())/2,
                   4,
                   barstyle='fancy',
                   zorder=10
                   )

    # Add Colorbar
    cbar = m.colorbar(im, location='right', pad="2%")
    cbar.set_label("Mean probability")

    # Add Title
    if title:
        ax.set_title(title)

    f.tight_layout()
    f.savefig(img_filename,dpi=100)
    logger.debug("Map saved {}".format(img_filename))


def plot_tephra(
        points_utm,
        vals,
        img_filename,
        title=None,
        scatter=False,
        basemap_res='h',
        **kwargs):

    logger = get_logger()
    logger.debug("Generating tephra static plot")
    f = plt.figure(figsize=(8, 4.5), dpi=100)
    ax = f.add_subplot(111)
    # Load grid from static file
    points = [utm.to_latlon(v.easting,
                            v.northing,
                            v.zone_number,
                            v.zone_letter)
              for v in points_utm]

    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])

    logger.info("Map latitudes interval: len {} min {} max {}".format(
            len(lats), lats.min(), lats.max()))
    logger.info("Map longitudes interval: len {} min {} max {}".format(
            len(lons), lons.min(), lons.max()))

    llp = (lats.min(), lons.min())
    urp = (lats.max(), lons.max())

    logger.info("Maximum load {}".format(vals.max()))

    m = Basemap(
            resolution=basemap_res,
            projection='merc',
            area_thresh=0.1,
            llcrnrlat=llp[0], llcrnrlon=llp[1],
            urcrnrlat=urp[0], urcrnrlon=urp[1])

    numcols, numrows = 1000, 1000
    xi = np.linspace(lons.min(), lons.max(), numcols)
    yi = np.linspace(lats.min(), lats.max(), numrows)

    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(lons, lats, vals, xi, yi,  interp='linear')

    # cmap = pyplot.get_cmap('afmhot_r')
    cmap = pyplot.get_cmap('gist_heat_r')
    # cmap = pyplot.get_cmap('hot_r')
    im = m.contourf(xi, yi, zi,
                    (1, 10, 100, 300, 500, 1000),
                    latlon=True,
                    cmap=cmap,
                    alpha=0.8,
                    zorder=2,
                    extend='max')
    im.cmap.set_over('black')

    if scatter:
        m.scatter(lons, lats, c=vals, s=100,
                  vmin=zi.min(), vmax=zi.max(), latlon=True)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines(zorder=6)
    m.drawstates(zorder=6)
    m.drawmapboundary(fill_color='paleturquoise')
    m.fillcontinents(color='grey')
    m.drawcountries()
    m.drawmapscale(xi.min() + (xi.max()-xi.min())/10,
                   yi.min() + (yi.max()-yi.min())/10,
                   (xi.max()-xi.min())/2,
                   (yi.max()-yi.min())/2,
                   30,
                   barstyle='fancy',
                   zorder=10
                   )

    # Add Colorbar
    cbar = m.colorbar(im, location='right', pad="2%",
                      cmap=cmap,
                      ticks=[1, 10, 100, 300, 500, 1000])
    cbar.set_ticklabels(['1', '10', '100', '300', '500', '1000'])
    cbar.set_label("$Load [kg/m^2]$")

    # Add Title
    if title:
        ax.set_title(title)
        # plt.title(title)

    f.tight_layout()
    f.savefig(img_filename,dpi=100)
    logger.debug("Plot saved {}".format(img_filename))


def plot_vh_prob(points_utm, vals, img_filename,
                 title=None, scatter=False, basemap_res='h', **kwargs):

    logger = get_logger()
    logger.debug("Generating vh probabilities plot: {} ".format(title))
    f = plt.figure(figsize=(8, 4.5), dpi=100)
    ax = f.add_subplot(111)
    # Load grid from static file
    points = [utm.to_latlon(v.easting,
                            v.northing,
                            v.zone_number,
                            v.zone_letter)
              for v in points_utm]

    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])

    logger.info("Map latitudes interval: len {} min {} max {}".format(
            len(lats), lats.min(), lats.max()))
    logger.info("Map longitudes interval: len {} min {} max {}".format(
            len(lons), lons.min(), lons.max()))

    llp = (lats.min(), lons.min())
    urp = (lats.max(), lons.max())

    logger.info("Maximum load {}".format(vals.max()))

    m = Basemap(
            resolution=basemap_res,
            projection='merc',
            area_thresh=0.1,
            llcrnrlat=llp[0], llcrnrlon=llp[1],
            urcrnrlat=urp[0], urcrnrlon=urp[1])

    numcols, numrows = 1000, 1000
    xi = np.linspace(lons.min(), lons.max(), numcols)
    yi = np.linspace(lats.min(), lats.max(), numrows)

    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(lons, lats, vals, xi, yi,  interp='linear')

    # cmap = pyplot.get_cmap('afmhot_r')
    cmap = pyplot.get_cmap('gist_heat_r')
    # cmap = pyplot.get_cmap('hot_r')
    im = m.contourf(xi, yi, zi,
                    (1, 10, 100, 300, 500, 1000),
                    latlon=True,
                    cmap=cmap,
                    alpha=0.8,
                    zorder=2,
                    extend='max')
    im.cmap.set_over('black')
    # im.cmap.set_under('blue')

    if scatter:
        m.scatter(lons, lats, c=vals, s=100,
                  vmin=zi.min(), vmax=zi.max(), latlon=True)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines(zorder=6)
    m.drawstates(zorder=6)
    m.drawmapboundary(fill_color='paleturquoise')
    m.fillcontinents(color='grey')
    m.drawcountries()
    m.drawmapscale(xi.min() + (xi.max()-xi.min())/10,
                   yi.min() + (yi.max()-yi.min())/10,
                   (xi.max()-xi.min())/2,
                   (yi.max()-yi.min())/2,
                   30,
                   barstyle='fancy',
                   zorder=10
                   )

    # Add Colorbar
    cbar = m.colorbar(im, location='right', pad="2%",
                      cmap=cmap,
                      ticks=[1, 10, 100, 300, 500, 1000])
    cbar.set_ticklabels(['1', '10', '100', '300', '500', '1000'])
    cbar.set_label("$Load [kg/m^2]$")

    if title:
        ax.set_title(title)

    f.tight_layout()
    f.savefig(img_filename,dpi=100)
    logger.debug("Map saved {}".format(img_filename))


def save_data(points_utm, vals, data_filename):
    points = [utm.to_latlon(v.easting,
                            v.northing,
                            v.zone_number,
                            v.zone_letter)
              for v in points_utm]
    with open(data_filename, 'w') as f:
        for i in range(len(points)):
            f.write("{0[0]:<14} {0[1]:<14} {1}\n".format(points[i],
                                                         vals[i]))


def to_geojson(points_utm, vals, data_filename):
    points = [utm.to_latlon(v.easting,
                            v.northing,
                            v.zone_number,
                            v.zone_letter)
              for v in points_utm]
    feats = [geojson.Feature(geometry=geojson.Point((points[i][1],
                                                     points[i][0])),
                             properties={"val":vals[i]})
             for i in range(len(points))]
    feats_collection = geojson.FeatureCollection(feats)
    dump = geojson.dumps(feats_collection, sort_keys=True)
    with open(data_filename, 'w') as f:
        f.write(dump)
    return feats


def get_load_kg(t_probs, loads_t, p_to_plot):
    if t_probs[0] > p_to_plot:
        for i_t in range(0, len(loads_t) - 1):
            if t_probs[i_t + 1] < p_to_plot:
                # ok, interpola e ritorna il risultato
                a = ((loads_t[i_t+1] - loads_t[i_t])/
                        (t_probs[i_t+1] - t_probs[i_t])) * \
                          (p_to_plot - t_probs[i_t]) \
                       + loads_t[i_t]

                return a / 0.00980665
        return 0
    else:
        return float('nan')


def plot_cumulative(node_key, mean, distr, img_filename):

    title_text={'unrest':'unrest', 'magmatic':'magmatic unrest', 'eruption': 'magmatic eruption'}

    color = plot_colors[node_key]
    f = plt.figure(figsize=(8, 4.5), dpi=100)

    gs = gridspec.GridSpec(2, 6, height_ratios = [1, 10], wspace=0.1, left=0.05, right=0.98, top=0.98, hspace=0.05)
    ax = plt.subplot(gs[0, :])
    bx = plt.subplot(gs[1, :-2])
    cx = plt.subplot(gs[1, -2:])
    ax.text(5, 2, "CDF of the conditional probability of {}".format(title_text[node_key]),
            size='x-large', fontweight='bold', ha='center', va='center')
    ax.set_axis_off()
    ax.axis([0, 10, 0, 4])

    p16 = np.percentile(distr, 16)
    p50 = np.percentile(distr, 50)
    p84 = np.percentile(distr, 84)

    bx_vals, base = np.histogram(distr, bins=len(distr))
    bx_vals = bx_vals / float(len(distr))

    bx_cdf = np.cumsum(bx_vals)
    bx.plot(base[:-1], bx_cdf, c=color)
    bx.axvline(x=mean, c=color, ls='-')
    bx.axvline(x=p16, c=color, ls=':')
    bx.axvline(x=p50, c=color, ls='--')
    bx.axvline(x=p84, c=color, ls=':')
    bx.axvspan(p16, p84, facecolor=color, alpha=0.3)
    bx.axis([0, 1, 0, 1])

    cx.get_xaxis().set_visible(False)
    cx.get_yaxis().set_visible(False)
    cx.text(5, 1, "Mean: {:0.3f}".format(mean),
             size='xx-large', ha='center', variant='small-caps')

    cx.text(4, 8, "p16: {:0.3f}".format(p16),
             size='large', variant='small-caps')
    cx.plot((1, 3.5), (8, 8), c=color, ls=':')

    cx.text(4, 6, "p50: {:0.3f}".format(p50),
             size='large', variant='small-caps')
    cx.plot((1, 3.5), (6, 6), c=color, ls='--')

    cx.text(4, 4, "p84: {:0.3f}".format(p84),
             size='large', variant='small-caps')
    cx.plot((1, 3.5), (4, 4), c=color, ls=':')

    cx.axis([0, 10, 0, 10])

    f.savefig(img_filename, dpi=100)


def plot_probabilities(times, data, img_filename):

    logger = get_logger()
    myFmt = mdates.DateFormatter('%d/%m')

    f = plt.figure(figsize=(8, 4.5), dpi=100)
    gs = gridspec.GridSpec(3, 9, height_ratios=[1, 1, 1], wspace=0.0,
                           left=0.07, right=0.93, hspace=0.25)
    ax0 = plt.subplot(gs[0, :])
    ax1 = plt.subplot(gs[1, :])
    ax2 = plt.subplot(gs[2, :])

    # UNREST
    unrest_color = plot_colors['unrest']
    try:
        unrest_mean, unrest_low, unrest_median, unrest_high = zip(*data['unrest'])
        ax0.plot(times, unrest_mean, '.-', color=unrest_color, label='Unrest')
        ax0.plot(times, unrest_low, ':', color=unrest_color)
        ax0.plot(times, unrest_median, '--', color=unrest_color)
        ax0.plot(times, unrest_high, ':', color=unrest_color)
        ax0.fill_between(times, unrest_low, unrest_high, facecolor=unrest_color,
                     alpha=0.3, interpolate=True)
        ax0.xaxis.set_major_formatter(myFmt)
    except TypeError:
        logger.warning("No unrest probabilities available")
    except ValueError:
        logger.warning("No unrest probabilities available")

    ax0.set_ylim(-0.1, 1.05)
    ax0.legend(loc="upper left", shadow=True, fancybox=True)

    # MAGMATIC
    magmatic_color = plot_colors['magmatic']
    try:
        magmatic_mean, magmatic_low, magmatic_median, magmatic_high = \
            zip(*data['magmatic'])
        ax1.plot(times, list(magmatic_mean), '.-', color=magmatic_color,
                 label='Magmatic')
        ax1.plot(times, list(magmatic_low), ':', color=magmatic_color)
        ax1.plot(times, list(magmatic_median), '--', color=magmatic_color)
        ax1.plot(times, list(magmatic_high), ':', color=magmatic_color)
        ax1.fill_between(times, magmatic_low, magmatic_high,
                         facecolor=magmatic_color, alpha=0.3, interpolate=True)
        ax1.xaxis.set_major_formatter(myFmt)
    except TypeError:
        logger.warning("No magmatic probabilities available")
    except ValueError:
        logger.warning("No magmatic probabilities available")

    ax1.set_ylim(-0.1, 1.05)
    ax1.legend(loc="upper left", shadow=True)

    # ERUPTION
    eruption_color = plot_colors['eruption']
    try:
        eruption_mean, eruption_low, eruption_median, eruption_high = \
            zip(*data['eruption'])
        ax2.plot(times, list(eruption_mean), '.-', color=eruption_color,
                 label='Eruption')
        ax2.plot(times, list(eruption_low), ':', color=eruption_color)
        ax2.plot(times, list(eruption_median), '--', color=eruption_color)
        ax2.plot(times, list(eruption_high), ':', color=eruption_color)
        ax2.fill_between(times, eruption_low, eruption_high,
                         facecolor=eruption_color, alpha=0.3, interpolate=True)
        ax2.xaxis.set_major_formatter(myFmt)
    except TypeError:
        logger.warning("No eruption probabilities available")
    except ValueError:
        logger.warning("No eruption probabilities available")

    ax2.set_ylim(-0.1, 1.05)
    ax2.legend(loc="upper left", shadow=True)

    f.savefig(img_filename, dpi=100)


def param_anomaly(param):
    logger = get_logger()
    logger.debug("{}: val {}, th1 {}, th2 {}, rel {}".format(
            param.name,
            param.value,
            param.threshold_1,
            param.threshold_2,
            param.relation))

    if param.relation == "=":
        if param.value == param.threshold_1:
            return False, True
        else:
            return False, False
    else:
        if param.relation == "<":
            if param.value < param.threshold_2:
                return False, True
            elif param.value < param.threshold_1:
                return True, False
        elif param.relation == ">":
            if param.value > param.threshold_2:
                return False, True
            elif param.value > param.threshold_1:
                return True, False
        return False, False


def export_contours(points_utm, values, contours, bet_conf,
                    rundir="/tmp",
                    basename="test", grid_step=500,
                    shapefile=True, kml=True, plot=False):

    logger = get_logger()
    logger.debug("Exporting contours")
    points_en = [(p.easting, p.northing) for p in points_utm]

    eastings = np.array([p[0] for p in points_en])
    northings = np.array([p[1] for p in points_en])

    x = np.arange(eastings.min(), eastings.max() + grid_step, grid_step)
    y = np.arange(northings.min(), northings.max() + grid_step,
                  grid_step)

    xi, yi = np.meshgrid(x, y)

    grid_values = griddata(eastings, northings, values,
                           xi, yi,  interp='linear')
    logger.debug("MAX values: {}, grid_values: {}".format(
            np.nanmax(values),
            np.nanmax(grid_values)))

    logger.debug("MIN values: {}, grid_values: {}".format(
            np.nanmin(values),
            np.nanmin(grid_values)))

    # Hazard grid star from downleft, while gdal expects to start from upperleft
    dataset = np.flipud(grid_values)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    srs.SetUTM(33, 1) # Set projected coordinate system to handle meters

    ncol = len(x)
    nrow = len(y)
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    geo_transform = [eastings.min(), grid_step, 0,
                     northings.max(), 0, -grid_step]
                    # cell height must be negative
                    # (-) to apply image space to map

    utm_tif_out = os.path.join(rundir, basename + '.tif')
    logger.debug("Using temp UTM .tif file {}".format(utm_tif_out))
    if os.path.exists(utm_tif_out):
        os.remove(utm_tif_out)

    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(utm_tif_out,
                                ncol, nrow,
                                nband,
                                gdal.GDT_Float32)
    dst_dataset.SetGeoTransform(geo_transform)

    dst_dataset.SetProjection(srs.ExportToWkt())
    # dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(dataset)
    dst_dataset = None

    longlat_tif_out = os.path.join(rundir, basename + '_longlat.tif')
    logger.debug("Using temp loglat .tif file {}".format(longlat_tif_out))
    if os.path.exists(longlat_tif_out):
        os.remove(longlat_tif_out)

    ext_args = [
        bet_conf['Scripts']['gdalwarp_wrapper'],
        utm_tif_out,
        longlat_tif_out]
    try:
        logger.info("Converting gdal reference system: {}".format(
                " ".join(ext_args)))
        exit_code = subprocess.call(ext_args)
    except OSError as e:
        logger.exception("OSError: %s" % e.strerror)
        exit_code = -1

    src_dataset = gdal.Open(longlat_tif_out)
    # src_data = src_dataset.ReadAsArray()

    # get parameters
    src_geotransform = src_dataset.GetGeoTransform()
    src_spatialreference = src_dataset.GetProjection()
    ncol = src_dataset.RasterXSize
    nrow = src_dataset.RasterYSize

    if shapefile:
        shapedir_out = os.path.join(rundir, basename + '_shp')
        logger.info("Saving shapefile")
        logger.debug("Shapefile dir {}".format(shapedir_out))
        if os.path.exists(shapedir_out):
            shutil.rmtree(shapedir_out)
        shapefile_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(
                shapedir_out)
        # shapefile_ds.SetGeoTransform(src_geotransform)
        # shapefile_ds.SetProjection(src_spatialreference)

        shapefile_layer = shapefile_ds.CreateLayer('contour')
        field_defn = ogr.FieldDefn('ID', ogr.OFTInteger)
        shapefile_layer.CreateField(field_defn)
        field_defn = ogr.FieldDefn(basename, ogr.OFTReal)
        shapefile_layer.CreateField(field_defn)

        gdal.ContourGenerate(src_dataset.GetRasterBand(1), 0, 0, contours, 0, 0,
                             shapefile_layer, 0, 1)

        shapefile_ds = None
        del shapefile_ds

    if kml:
        logger.info("Saving KML file")
        kml_out = os.path.join(rundir, basename + '.kml')
        kml_tmp = os.path.join(rundir, basename + '_tmp.kml')
        logger.debug("KML file: {}".format(kml_out))

        isobands(longlat_tif_out, 1, kml_tmp, 'KML', 'contour',
                 basename, 0, 0, contours, min_level=1*10**-10)

        convert_kml(kml_tmp, kml_out)

    if plot:
        logger.info("Plotting contours")
        f = plt.figure()
        # cmap = colors.ListedColormap([ 'cyan', 'limegreen', 'yellow',
        #                                'orange', 'red', 'black'], N=6)
        # cmap = pyplot.get_cmap('afmhot_r')
        cmap = pyplot.get_cmap('gist_heat_r')
        # cmap = pyplot.get_cmap('hot_r')
        ax0 = f.add_subplot(121)
        ax0.scatter(eastings, northings, c=values, s=6, cmap=cmap)
        ax0.set_aspect('equal')
        ax1 = f.add_subplot(122,  sharex=ax0)

        ax1.set_title("linear interpolation")
        ax1.imshow(grid_values,
                   vmin=values.min(), vmax=values.max(),
                   origin='lower',
                   extent=[xi.min(), xi.max(), yi.min(),  yi.max()])

        f.tight_layout()

        plot_out = os.path.join(rundir, basename + '_static.png')
        logger.debug('Saving contours plot in {}'.format(plot_out))
        if os.path.exists(plot_out):
            os.remove(plot_out)
        f.savefig(plot_out)


def isobands(in_file, band, out_file, out_format, layer_name, attr_name,
    offset, interval, contours, min_level=None):
    '''
    The method that calculates the isobands
    http://geoexamples.blogspot.it/2013/08/creating-vectorial-isobands-with-python.html
    '''

    logger = get_logger()
    logger.debug("Calculating isobands")

    #Loading the raster file
    ds_in = gdal.Open(in_file)
    band_in = ds_in.GetRasterBand(band)
    xsize_in = band_in.XSize
    ysize_in = band_in.YSize

    stats = band_in.GetStatistics(True, True)

    if min_level == None:
        min_value = stats[0]
        min_level = ( offset + interval *
            (floor((min_value - offset)/interval) - 1) )
    nodata_value = min_level - interval



    geotransform_in = ds_in.GetGeoTransform()

    srs = osr.SpatialReference()
    srs.ImportFromWkt( ds_in.GetProjectionRef() )

    data_in = band_in.ReadAsArray(0, 0, xsize_in, ysize_in)


    #The contour memory
    contour_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    contour_lyr = contour_ds.CreateLayer('contour',
        geom_type = ogr.wkbLineString25D, srs = srs )
    field_defn = ogr.FieldDefn('ID', ogr.OFTInteger)
    contour_lyr.CreateField(field_defn)
    field_defn = ogr.FieldDefn('elev', ogr.OFTReal)
    contour_lyr.CreateField(field_defn)

    #The in memory raster band, with new borders to close all the polygons
    driver = gdal.GetDriverByName( 'MEM' )
    xsize_out = xsize_in + 2
    ysize_out = ysize_in + 2

    column = numpy.ones((ysize_in, 1)) * nodata_value
    line = numpy.ones((1, xsize_out)) * nodata_value

    data_out = numpy.concatenate((column, data_in, column), axis=1)
    data_out = numpy.concatenate((line, data_out, line), axis=0)

    ds_mem = driver.Create( '', xsize_out, ysize_out, 1, band_in.DataType)
    ds_mem.GetRasterBand(1).WriteArray(data_out, 0, 0)
    ds_mem.SetProjection(ds_in.GetProjection())
    #We have added the buffer!
    ds_mem.SetGeoTransform((geotransform_in[0]-geotransform_in[1],
        geotransform_in[1], 0, geotransform_in[3]-geotransform_in[5],
        0, geotransform_in[5]))
    gdal.ContourGenerate(ds_mem.GetRasterBand(1), interval,
        offset, contours, 0, 0, contour_lyr, 0, 1)

    #Creating the output vectorial file
    drv = ogr.GetDriverByName(out_format)
    if os.path.exists(out_file):
        os.remove(out_file)
    dst_ds = drv.CreateDataSource( out_file )

    dst_layer = dst_ds.CreateLayer(layer_name,
        geom_type = ogr.wkbPolygon, srs = srs)

    fdef = ogr.FieldDefn( attr_name, ogr.OFTReal )
    dst_layer.CreateField( fdef )


    contour_lyr.ResetReading()

    geometry_list = {}
    for feat_in in contour_lyr:
        value = feat_in.GetFieldAsDouble(1)

        geom_in = feat_in.GetGeometryRef()
        points = geom_in.GetPoints()

        if ((points[0][0] == points[-1][0]) and
            (points[0][1] == points[-1][1])):
            pass
        else:
            points.append(points[0])

        if ((value >= min_level and points[0][0] == points[-1][0]) and
            (points[0][1] == points[-1][1])):
            if (value in geometry_list) is False:
                geometry_list[value] = []

            pol = ogr.Geometry(ogr.wkbPolygon)
            ring = ogr.Geometry(ogr.wkbLinearRing)

            for point in points:

                p_y = point[1]
                p_x = point[0]

                if p_x < (geotransform_in[0] + 0.5*geotransform_in[1]):
                    p_x = geotransform_in[0] + 0.5*geotransform_in[1]
                elif p_x > ( (geotransform_in[0] +
                    (xsize_in - 0.5)*geotransform_in[1]) ):
                    p_x = ( geotransform_in[0] +
                        (xsize_in - 0.5)*geotransform_in[1] )
                if p_y > (geotransform_in[3] + 0.5*geotransform_in[5]):
                    p_y = geotransform_in[3] + 0.5*geotransform_in[5]
                elif p_y < ( (geotransform_in[3] +
                    (ysize_in - 0.5)*geotransform_in[5]) ):
                    p_y = ( geotransform_in[3] +
                        (ysize_in - 0.5)*geotransform_in[5] )

                ring.AddPoint_2D(p_x, p_y)


            pol.AddGeometry(ring)
            geometry_list[value].append(pol)

    values = sorted(geometry_list.keys())

    geometry_list2 = {}

    for i in range(len(values)):
        geometry_list2[values[i]] = []
        interior_rings = []
        for j in range(len(geometry_list[values[i]])):
            if (j in interior_rings) == False:
                geom = geometry_list[values[i]][j]

                for k in range(len(geometry_list[values[i]])):

                    if ((k in interior_rings) == False and
                        (j in interior_rings) == False):
                        geom2 = geometry_list[values[i]][k]

                        if j != k and geom2 != None and geom != None:
                            if geom2.Within(geom) == True:

                                geom3 = geom.Difference(geom2)
                                interior_rings.append(k)
                                geometry_list[values[i]][j] = geom3

                            elif geom.Within(geom2) == True:

                                geom3 = geom2.Difference(geom)
                                interior_rings.append(j)
                                geometry_list[values[i]][k] = geom3

        for j in range(len(geometry_list[values[i]])):
            if ( (j in interior_rings) == False and
                geometry_list[values[i]][j] != None ):
                geometry_list2[values[i]].append(geometry_list[values[i]][j])


    for i in range(len(values)):
        value = values[i]
        if value >= min_level:
            for geom in geometry_list2[values[i]]:

                if i < len(values)-1:

                    for geom2 in geometry_list2[values[i+1]]:
                        if geom.Intersects(geom2) is True:
                            geom = geom.Difference(geom2)

                feat_out = ogr.Feature( dst_layer.GetLayerDefn())
                feat_out.SetField( attr_name, value )
                feat_out.SetGeometry(geom)
                if dst_layer.CreateFeature(feat_out) != 0:
                    print "Failed to create feature in shapefile.\n"
                    exit( 1 )
                feat_out.Destroy()

def convert_kml(in_file, out_file):
    logger = get_logger()
    logger.debug("Converting KML file")
    colors = {'10':   {'border': 'FD14F078', 'fill': 'A014F078'},
              '100':  {'border': 'FD78FFF0', 'fill': 'A078FFF0'},
              '300':  {'border': 'FD1485FF', 'fill': 'A01485FF'},
              '500':  {'border': 'FD1423FF', 'fill': 'A01423FF'},
              '1000': {'border': 'FD780078', 'fill': 'A0780078'}}

    kml_out = simplekml.Kml()

    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    xml_doc = ET.parse(in_file)
    kml_root = xml_doc.getroot()

    for doc in kml_root.findall('kml:Document', ns):
        for folder in doc.findall('kml:Folder', ns):
            for place in folder.findall('kml:Placemark', ns):
                place_name = place.find('kml:ExtendedData', ns).\
                    find('kml:SchemaData', ns).\
                    find('kml:SimpleData', ns).text

                for poly in place.findall('kml:Polygon', ns):
                    outer_coords = poly.find('kml:outerBoundaryIs', ns).\
                        find('kml:LinearRing', ns).find('kml:coordinates', ns).text

                    new_outer_coords = [
                        (float(c.split(",")[0]), float(c.split(",")[1]))
                                  for c in outer_coords.split(" ")]
                    pol = kml_out.newpolygon(name=place_name,
                                             outerboundaryis = new_outer_coords)

                    inner_line = poly.find('kml:innerBoundaryIs', ns)
                    if inner_line is not None:
                        inner_coords = inner_line.\
                            find('kml:LinearRing', ns).find('kml:coordinates', ns).text

                        new_inner_coords = [
                            (float(c.split(",")[0]), float(c.split(",")[1]))
                                      for c in inner_coords.split(" ")]

                        pol.innerboundaryis = new_inner_coords

                    pol.style.polystyle.color = colors[place_name]['fill']
                    pol.style.polystyle.outline = 1
                    pol.style.linestyle.color = colors[place_name]['border']
                    pol.style.linestyle.width = 2

    kml_out.save(out_file)
    logger.info("KML file saved: {}".format(kml_out))


def init_logger(level=logging.INFO):
    logging.basicConfig(level=level, format=LOG_FORMAT)


def log_to_file(rundir, filename, level=logging.INFO):
    global __logger
    file = os.path.join(rundir, 'logs', filename)
    if not os.path.isdir(os.path.join(rundir, 'logs')):
        os.mkdir(os.path.join(rundir, 'logs'))

    fh = logging.FileHandler(file, mode='a')
    fh.setLevel(level)
    fh.setFormatter(LOG_FORMATTER)
    __logger.addHandler(fh)
    __logger.debug("Log to file initialized")


def get_logger(level=logging.DEBUG, name=''):
    global __logger
    if __logger is None:
        init_logger(level=level)
        __logger = logging.getLogger(name)
    return __logger
