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

from datetime import datetime, time, timedelta
from collections import OrderedDict
from sys import argv
from bet.function.cli import opts_parser
from bet.function import plot_vents, plot_tephra, export_contours
from bet.conf import BetConf, MonitoringConf
import os.path
import simplejson as json
import re
import glob
import shutil
from collections import namedtuple
import netCDF4 as nc
import bet.messaging.celery.tasks
import pickle


class TephraModelOut(object):
    def __init__(self, model_name=None, base_path=None, model_res=None):
        self._model_name = model_name
        self._base_path = base_path
        if model_name is None:
            self._model_res = dict()
        else:
            self._model_res = model_res

    def __repr__(self):
        s = "TephraModelOut<"
        s += "basedir:%s, " % self._base_path
        s += "model_res:%s, " % self._model_res
        s += ">"
        return s

    @property
    def model_res(self):
        return self._model_res

    @property
    def base_path(self):
        return self._base_path

    def get_model_res(self, size):
        try:
            return self._model_res[size]
        except KeyError:
            return None

    def set_model_res(self, size, model_res):
        self._model_res[size] = model_res

    @classmethod
    def from_dict(cls, d):
        print "from dict"
        print d['model_res']
        tmo = cls()
        # tmo._sizes = d['sizes']
        tmo._model_name = d['model_name']
        tmo._base_path = d['base_path']
        # tmo._loads = dict((k, d['loads'][k]) for k in d['loads'])
        tmo._model_res = dict((k, d['model_res'][k]) for k in d['model_res'])
        # tmo._filenames = dict((k, d['filenames'][k]) for k in d['filenames'])
        return tmo


class TephraModelOutList(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def __repr__(self):
        s = "["
        for i in self:
            s += "%s, " % i
        s += "]"
        return s


class TephraOut(object):
    def __init__(self,
                 # run_date=datetime.now(),
                 # sizes=list(),
                 haz_models=list()):
        self._forecast_orig_t = None
        # self._run_date = run_date
        self._haz_loads = dict()
        # print "WARNING: GENERATING RANDOM TEPHRA SAMPLES!!!"
        # self._samples = dict((k, np.random.beta(1, 2, 1000))
        #                      for k in self._sizes)

    def __repr__(self):
        s = "TephraOut<"
        s += "forecast_orig(%s), " % self._forecast_orig_t
        for haz in self._haz_loads.keys():
            s += "hazard(%s): %s, " % (haz, self._haz_loads[haz])
        s += ">"
        return s

    @classmethod
    def from_dict(cls, d):
        haz_dict = dict((k, dict()) for k in d['haz_loads'])
        for haz in d['haz_loads'].keys():
            tep_out_obj_list = TephraModelOutList([])
            tep_out_list = d['haz_loads'][haz]
            for tep_out in tep_out_list:
                tep_obj = TephraModelOut(model_name=haz)
                for size in tep_out['model_res'].keys():
                    tep_obj.set_model_res(size, tep_out['model_res'][size])
                tep_out_obj_list.append(tep_obj)
            haz_dict[haz] = tep_out_obj_list
        # to = cls(
        #     run_date=datetime.strptime(d.get('date', ""), "%Y-%m-%d %H:%M:%S"))
        # to._forecast_orig_t = datetime.strptime(d.get('forecast_origin_t', ""),
        #                                         "%Y-%m-%d %H:%M:%S")
        to = cls()
        to._haz_loads = haz_dict
        # to._samples = dict((k, np.array(d['samples'][k])) for k in d['sizes'])
        return to

    def to_json(self, **kwargs):
        return json.dumps(self.__dict__(),
                          ensure_ascii=False,
                          sort_keys=True,
                          **kwargs)

    @classmethod
    def from_json(cls, c_ser, **kwargs):
        return cls.from_dict(json.loads(c_ser, **kwargs))

    @property
    def forecast_orig_t(self):
        return self._forecast_orig_t

    @property
    def haz_loads(self):
        return self._haz_loads

    @property
    def n_sim(self):
        n = 0
        for k in self._haz_loads.keys():
            n += len(self._haz_loads[k])
        return n

    @forecast_orig_t.setter
    def forecast_orig_t(self, val):
        self._forecast_orig_t = val

    def get_haz_loads(self, haz_name):
        return self._haz_loads[haz_name]

    def set_haz_loads(self, haz_name, loads):
        self._haz_loads[haz_name] = loads

    def get_haz_loads_n(self, haz_name, size):
        return len([x.get_model_res(size)
                    for x in self.get_haz_loads(haz_name)
                    if x.get_model_res(size)])


def get_tephra_data(exp_window, bet_conf):
    hazard_models = bet_conf.BET['Hazard']['HazardModels'].keys()
    output_data = None

    for haz_model in hazard_models:
        haz_list = find_tephra_models(
            exp_window,
            haz_model,
            bet_conf)
        if len(haz_list) > 0:
            output_data = TephraOut()
            output_data._haz_loads[haz_model] = haz_list

            print "WARNING, forecast_origin_t TO FIX!"
            forecast_orig_t = None
            for haz_model in output_data.haz_loads.keys():
                try:
                    print output_data.get_haz_loads(haz_model)[0].base_path
                    print output_data.get_haz_loads(haz_model)[0].base_path.split('/')[-1]
                    print output_data.get_haz_loads(haz_model)[0].base_path.split('/')[-1].split('-')[1]
                    output_data.forecast_orig_t = \
                        datetime.strptime(
                                output_data.get_haz_loads(haz_model)[0].base_path.
                                    split('/')[-1].split('-')[1], "%Y%m%d%H")
                    break
                except Exception as e:
                    print e.message
                    pass
    return output_data


def check_all_sizes(sizes, exclude=list()):
    pass


# Generate all possible interesting simulations path
def candidate_sims(exp_window, prefix, basedir):
    # TODO: da rifare con la nuova convenzione dei nomi
    # Cambiera' il suffisso della directory e dei files
    # Devo cercare il piu' aggiornato, con t_sim < run_date,
    # controllare lo status (in definizione) e collezionare le sim
    day_glob = "*-[{}|{}|{}|{}]-*".format(
            exp_window['end'].strftime("%Y%m%d"),
            exp_window['begin'].strftime("%Y%m%d"),
            (exp_window['begin'] - timedelta(hours=24)).strftime("%Y%m%d"),
            (exp_window['begin'] - timedelta(hours=48)).strftime("%Y%m%d"))
    print day_glob


    #
    # if run_date.time() < time(hour=12):
    #     target_datetime = datetime.combine(target_date, time(hour=00))
    # else:
    #     target_datetime = datetime.combine(target_date, time(hour=12))
    # base_glob = prefix + '?'
    # base_glob += '-' + target_datetime.strftime("%Y%m%d%H")
    # base_glob += '-00'
    # return list([os.path.join(basedir, base_glob)])


# Check if status file exists and contain a 0
def succ_sim(sim_path):
    status_filepath = os.path.join(
        sim_path,
        os.path.basename(sim_path) + '.status')
    if os.path.exists(status_filepath):
        with file(status_filepath) as f:
            res = f.read().strip(" \n\t")
        if res == '0':
            return True
    return False


def check_res(sim_path):
    file_path = os.path.join(sim_path,
                             os.path.basename(sim_path) + '.res.nc')
    if os.path.exists(file_path):
        return file_path


def check_inp(sim_path):
    file_path = os.path.join(sim_path,
                             os.path.basename(sim_path) + '.inp')
    if os.path.exists(file_path):
        return file_path


# Check if, for every run, there are all sizes and status correct
def avail_sims(haz_model, sim_list, sizes, exclude_sizes=list()):
    av_sims = list()
    for sim in sim_list:
        valid_sim = True
        sim_dict = dict()
        for size in [s for s in sizes if s not in exclude_sizes]:
            cur_sim = sim.replace('?', size)
            # print cur_sim
            if not os.path.isdir(cur_sim) or not succ_sim(cur_sim):
                valid_sim = False
            else:
                res_path = check_res(cur_sim)
                inp_path = check_inp(cur_sim)
                if not res_path or not inp_path:
                    valid_sim = False
                else:
                    sim_dict[size] = {
                        'res_file': res_path,
                        'inp_file': inp_path
                    }
        if valid_sim:
            mod_out = TephraModelOut(model_name=haz_model,
                                     base_path=sim,
                                     model_res=sim_dict)
            parse_model_inps(mod_out)
            av_sims.append(mod_out)

    return av_sims


def parse_model_inps(model_out):
    for key_size in model_out.model_res.keys():
        mod_size_dict = model_out.get_model_res(key_size)
        # print mod_size_dict
        mod_inp = parse_fall3d_inp(mod_size_dict['inp_file'])
        inp_times = mod_inp['children']['TIME_UTC']['children']
        meteo_base = inp_times['YEAR']['val'] + '-'
        meteo_base += "{:02d}-".format(int(inp_times['MONTH']['val']))
        meteo_base += "{:02d}".format(int(inp_times['DAY']['val']))
        meteo_base = datetime.strptime(meteo_base, "%Y-%m-%d")
        meteo_offset = timedelta(hours=int(inp_times['BEGIN_METEO_DATA_('
                                                     'HOURS_AFTER_00)']['val']))
        mod_size_dict['weather_date'] = meteo_base + meteo_offset
        mod_size_dict['eruption_offset'] = inp_times['ERUPTION_START_(HOURS_AFTER_00)']['val']
        inp_grid=mod_inp['children']['GRID']['children']['UTM']['children']
        grid = dict(
            xmin=inp_grid['XMIN']['val'],
            xmax=inp_grid['XMAX']['val'],
            ymin=inp_grid['YMIN']['val'],
            ymax=inp_grid['YMAX']['val'],
            nx=mod_inp['children']['GRID']['children']['NX']['val'],
            ny=mod_inp['children']['GRID']['children']['NY']['val'],
        )
        vent = dict(
            x=inp_grid['X_VENT']['val'],
            y=inp_grid['Y_VENT']['val']

        )
        mod_size_dict['grid'] = grid
        mod_size_dict['vent'] = vent
        # print mod_size_dict


def find_tephra_models(exp_window, haz_model, bet_conf):
    sizes = bet_conf.BET['Styles']['sizes']
    file_prefix = bet_conf['Apollo'][haz_model]['file_prefix']
    result_dir = bet_conf['Apollo'][haz_model]['results_dir']

    av_sims = valid_sims(exp_window, haz_model, result_dir, file_prefix,
                         sizes, exclude_size=['E'])

    return TephraModelOutList(av_sims)


def parse_fall3d_inp(path):
    # TODO: bugged, to fix !
    ConfEntry = namedtuple('ConfEntry', ['indent', 'conf'])
    comments_re = re.compile(".*!.*")
    conf_stack = [ConfEntry(-1, dict(val=None, children=dict()))]
    new_top = conf_stack[len(conf_stack) - 1].conf
    ln = 0
    with open(path) as inp:
        for l in inp:
            if comments_re.match(l):
                continue
            l_ind = len(l) - len(l.lstrip(' \t\n'))
            line_split = l.strip(' \t\n').split('=')
            key = line_split[0].strip(' \t\n')
            if len(line_split) > 1:
                val = line_split[1].strip(' \t\n')
            else:
                val = None
            if l_ind <= conf_stack[len(conf_stack) - 1].indent:
                while l_ind < conf_stack[len(conf_stack) - 1].indent:
                    b = conf_stack.pop()
                new_top = {'val': val, 'children': dict()}
                conf_stack[len(conf_stack) - 1].conf['children'][key] = new_top
            elif l_ind > conf_stack[len(conf_stack) - 1].indent:
                new_top['children'][key] = {'val': val, 'children': dict()}
                conf_stack.append(ConfEntry(l_ind, new_top))
                new_top = new_top['children'][key]
    return conf_stack[0].conf


def load_tephra_matrix(model_name, load_file):
    nc_format = "NETCDF3_64BIT"
    if model_name == 'fall3d':
        rootgrp = nc.Dataset(load_file, "r", format=nc_format)
        #   Caso fall3d, ci sono solo 2 frame e leggo il secondo
        mat = rootgrp['LOAD'][len(rootgrp.dimensions['time']) - 1]
    elif model_name == 'hazmap':
        rootgrp = nc.Dataset(load_file, "r", format=nc_format)
        mat = rootgrp['LOAD'][len(rootgrp.dimensions['time']) - 1]
    else:
        rootgrp = nc.Dataset(load_file, "r", format=nc_format)
        mat = rootgrp['LOAD'][len(rootgrp.dimensions['time']) - 1]

    return mat


if __name__ == "__main__":

    opts = vars(opts_parser().parse_args(argv[1:]))
    print opts
    load_dir = opts['load']

    if opts['obs_time']:
        obs_time = datetime.strptime(opts['obs_time'], "%Y%m%d_%H%M%S")
    else:
        obs_time = datetime.now()

    if load_dir:
        print("Loading data from {0}".format(load_dir))
        with open(os.path.join(load_dir, "bet_conf.pick")) as f:
            bet_conf = pickle.load(f)
        with open(os.path.join(load_dir, "mon_conf.pick")) as f:
            mon_conf = pickle.load(f)
        with open(os.path.join(load_dir, "bet_ef_out.pick")) as f:
            bet_ef_out = pickle.load(f)
        with open(os.path.join(load_dir, "tephra_out.pick")) as f:
            tephra_out = pickle.load(f)
        with open(os.path.join(load_dir, "bet_vh_out.pick")) as f:
            bet_vh_out = pickle.load(f)
        print("All data loaded.")
        run_dir = load_dir
        bet_conf.load_tephra_grid()
        for haz_model in tephra_out.haz_loads.keys():
            for haz_data in tephra_out.get_haz_loads(haz_model):
                for size in bet_conf.BET['Styles']['sizes']:
                    try:
                        res_file = haz_data.get_model_res(size)['res_file']
                        res_file = os.path.join(
                                   "/home/jigen/data/dev/bet_package/examples/tephra",
                                res_file.split('/')[-2],
                        res_file.split('/')[-1])
                        print res_file

                    except TypeError:
                        res_file = None

                    if res_file:

                        print "Plotting data from file: %s" % res_file
                        load = load_tephra_matrix(haz_model,
                                                  res_file).T.ravel()
                        title = "tephra_{0}_{1}".format(haz_model, size)
                        img = os.path.join(run_dir,
                                           title + ".png")
                        plot_tephra(bet_conf.tephra_grid, load, img, title=title)


    else:
        # obs_date = datetime.strptime("20160117_130000", "%Y%m%d_%H%M%S")
        bet_conf = BetConf(opts['conf'], obs_time=obs_time)
        # bet_conf.merge_local_conf()
        bet_conf.obs_time = obs_time
        # models_out = bet.messaging.celery.tasks.tephra_models(bet_conf)
        tephra_out = get_tephra_data(obs_time, bet_conf)
        # print models_out


def valid_sims(exp_window, haz_model, basedir, prefix, sizes,
               exclude_size=list()):

    valid_sizes = [size for size in sizes if size not in exclude_size]
    valid_weather = [exp_window['end'].strftime("%Y%m%d") + '[0-9][0-9]',
                     exp_window['begin'].strftime("%Y%m%d") + '[0-9][0-9]',
                     (exp_window['begin'] - timedelta(hours=24)).
                         strftime("%Y%m%d") + '[0-9][0-9]',
                     (exp_window['begin'] - timedelta(hours=48)).
                         strftime("%Y%m%d") + '[0-9][0-9]']

    sims_re = "{}({})-({})-([0-9][0-9])".format(
            prefix,
            "|".join(valid_sizes),
            "|".join(valid_weather))

    try:
        res = [r for r in [re.search(sims_re, d) for d in os.listdir(basedir)] if\
                r is not None]
    except OSError as e:
        if e.errno == 2:
            print "Warning, directory {} does not exists".format(basedir)
            return []
        else:
            raise e

    t_list = [(datetime.strptime(r.group(2), "%Y%m%d%H"),
               timedelta(hours=int(r.group(3)))) for r in res]

    uni_t = sorted(list(set([t for t in t_list
            if (exp_window['begin'] <= (t[0] + t[1]) <= exp_window['end'])])),
            key=lambda d: d[0], reverse=True)

    t_groups = OrderedDict()
    for t in uni_t:
        if t[0] in t_groups.keys():
            t_groups[t[0]].append((t[0], t[1]))
        else:
            t_groups[t[0]] = list()
            t_groups[t[0]].append((t[0], t[1]))

    valid_sim_times = []
    for inst in t_groups.keys():
        # print "weather time {}".format(inst)
        t_groups[inst] = sorted(t_groups[inst],
                                key=lambda d: d[0],
                                reverse=True)
        sim_i = 0
        while sim_i < len(t_groups[inst]):
            sim_t = t_groups[inst][sim_i]
            sim_is_valid = True
            sim_dict = dict()
            h = int(sim_t[1].days * 24 + sim_t[1].seconds/3600)
            # print "{} {}".format(sim_t[0].strftime("%Y%m%d%H"), h)
            for size in valid_sizes:
                sim_dir = os.path.join(basedir,
                                       "{}{}-{}-{:02d}".format(
                                               prefix,
                                               size,
                                               sim_t[0].strftime("%Y%m%d%H"),
                                               h))
                val = os.path.isdir(sim_dir) and succ_sim(sim_dir)
                if val:
                    res_path = check_res(sim_dir)
                    inp_path = check_inp(sim_dir)
                    if not res_path or not inp_path:
                        val = False
                    else:
                        sim_dict[size] = {
                             'res_file': res_path,
                             'inp_file': inp_path
                         }


                # if sim_dir == './examples/fall3d_res/CFL-2016030800-00':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFH-2016030800-00':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFM-2016030800-00':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFL-2016030700-24':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFH-2016030700-24':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFM-2016030700-24':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFL-2016030700-18':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFH-2016030700-18':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFM-2016030700-18':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFL-2016030712-00':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFH-2016030712-00':
                #     val = True
                # if sim_dir == './examples/fall3d_res/CFM-2016030712-00':
                #     val = True
                # print "{} is valid {}".format(sim_dir, val)
                sim_is_valid &= val

            if sim_is_valid:
                sim_base = os.path.join(basedir,
                                        "{}?-{}-{:02d}".format(
                                                prefix,
                                                sim_t[0].strftime("%Y%m%d%H"),
                                                h))
                mod_out = TephraModelOut(model_name=haz_model,
                                         base_path=sim_base,
                                         model_res=sim_dict)
                parse_model_inps(mod_out)
                valid_sim_times.append(mod_out)

            sim_i += 1

        if len(valid_sim_times) > 0:
            break

    return valid_sim_times


def tephra_problem_name(scenario, year, month, day, hour):
    return "{}-{}{}{}{}-{}".format(scenario[0],
                                   year, month, day, hour, scenario[1])


def copy_gmt_imgs(tephra_out, run_dir):
    # Copy static images generated from model specific script
    if tephra_out is not None:
        for haz_model in tephra_out.haz_loads.keys():
            haz_dir = os.path.join(run_dir, haz_model)
            if not os.path.isdir(haz_dir):
                os.mkdir(haz_dir)
            for haz_data in tephra_out.get_haz_loads(haz_model):
                # print haz_data.base_path
                img_glob = os.path.join(haz_data.base_path, 'GMT', "*.LOAD.003.jpg")
                for img in glob.glob(img_glob):
                    shutil.copy(img, haz_dir)


def all_images(bet_conf, tephra_out, run_dir):
    for haz_model in tephra_out.haz_loads.keys():
        haz_dir = os.path.join(run_dir, haz_model)
        if not os.path.isdir(haz_dir):
            os.mkdir(haz_dir)
        for haz_data in tephra_out.get_haz_loads(haz_model):
            for size in bet_conf.BET['Styles']['sizes']:
                try:
                    res_file = haz_data.get_model_res(size)['res_file']
                except TypeError:
                    res_file = None

                if res_file:
                    print "Plotting data from file: %s" % res_file
                    load = load_tephra_matrix(haz_model,
                                              res_file).T.ravel()
                    if size == 'M' and haz_model == 'fall3d':
                        title = "Fall3D simulations (vent in the centre of " \
                                "the CF caldera)"

                    else:
                        title = "tephra_{0}_{1}".format(haz_model, size)
                    fn = "tephra_{0}_{1}.png".format(haz_model, size)
                    img = os.path.join(run_dir, fn)
                    if bet_conf.tephra_grid is None:
                        bet_conf.load_tephra_grid()
                    plot_tephra(bet_conf.tephra_grid, load, img, title=title)

                    export_contours(bet_conf.tephra_grid,
                        load,
                        [10, 100, 300, 500, 1000],
                        bet_conf,
                        basename="tephra_" + haz_model + "_" + size,
                        rundir=run_dir,
                        plot=True)



def static_images(bet_conf, tephra_out, run_dir):
    for haz_model in tephra_out.haz_loads.keys():
        haz_dir = os.path.join(run_dir, haz_model)
        if not os.path.isdir(haz_dir):
            os.mkdir(haz_dir)
        for haz_data in tephra_out.get_haz_loads(haz_model):
            for size in bet_conf.BET['Styles']['sizes']:
                try:
                    res_file = haz_data.get_model_res(size)['res_file']
                except TypeError:
                    res_file = None

                if res_file:
                    print "Plotting data from file: %s" % res_file
                    load = load_tephra_matrix(haz_model,
                                              res_file).T.ravel()
                    if size == 'M' and haz_model == 'fall3d':
                        title = "Fall3D simulations (vent in the centre of " \
                                "the CF caldera)"

                    else:
                        title = "tephra_{0}_{1}".format(haz_model, size)
                    fn = "tephra_{0}_{1}.png".format(haz_model, size)
                    img = os.path.join(run_dir, fn)
                    if bet_conf.tephra_grid is None:
                        bet_conf.load_tephra_grid()
                    plot_tephra(bet_conf.tephra_grid, load, img, title=title)
