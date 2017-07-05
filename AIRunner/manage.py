import time
import datetime
import glob
import json
import os
from functools import wraps
import subprocess
import platform

import submodules as sbmod
from enum import Enum
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class BackGroundProcess:

    proc=None

    def __init__(self):
        self.proc=None

    def read_std(self):
        if (self.proc is None):
            raise Exception("proc is None")
            return ""

        b_stdout_data, b_stderr_data = self.proc.communicate()

        stdout_data =  b_stdout_data.decode("utf-8")
        stderr_data =  b_stderr_data.decode("utf-8")
        return stdout_data, stderr_data

    def close(self):
        if (self.proc is None):
            raise Exception("proc is None")

        self.proc.terminate()
        self.proc = None

    def popen(self, args, cws):
        self.proc = subprocess.Popen(args,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           cwd=cws,
                                           close_fds=True,
                                           )

    def isNotNone(self):
        return self.proc  !=  None

    def isRunning(self):

        # is Stopping
        if (self.proc == None): 
            is_runngin = False
            return is_runngin

        if self.proc.poll() is None :
            is_runngin = True
        else:
            is_runngin = False

        return is_runngin


class Manager:

    class State(Enum):
        UNEXCEPT=0
        SUCESS=1
        ERORR=2

    """
    project_rootdir : child(models, params, results)
    """
    model_pyfile_paths = []
    param_confile_paths = []

    param_ext = ".conf"
    model_ext = ".py"

    def __init__(self, project_rootdir, models_dirname, params_dirname, results_dirname):
        self.project_rootdir = project_rootdir
        self.models_dirname = models_dirname
        self.params_dirname = params_dirname
        self.results_dirname = results_dirname

    def search_models(self):
        models_searchpath = os.path.join(self.project_rootdir, self.models_dirname, "*.py")
        self.model_pyfile_paths = glob.glob(models_searchpath)

    def get_model_names(self):
        return self._get_basenames(self.model_pyfile_paths)

    def search_params(self):
        params_searchpath = os.path.join(self.project_rootdir, self.params_dirname, "*.conf")
        self.param_confile_paths = glob.glob(params_searchpath)

    def get_param_names(self):
        return self._get_basenames(self.param_confile_paths)

    def _get_basenames(self, fullpathlist):
        _base_names = [os.path.basename(p) for p in fullpathlist]
        base_names = [os.path.splitext(p)[0] for p in _base_names]
        return base_names

    def get_workspace_path(self, model_name, param_name):
        return sbmod.generate_dir([self.project_rootdir, self.results_dirname, model_name, param_name])

    def get_state(self, workspace_path):
        if not os.path.exists(os.path.join(workspace_path, "start.txt")):
            return self.State.UNEXCEPT

        if not os.path.exists(os.path.join(workspace_path, "success.txt")):
            return self.State.SUCESS    

        if not os.path.exists(os.path.join(workspace_path, "error.txt")):
            return self.State.ERORR

    def next(self):
        self.search_params()
        self.search_models()

        for param in self.get_param_names():
            for model in self.get_model_names():

                ws_path = self.get_workspace_path(model, param)
                
                state = self.get_state(ws_path)

                if(state == self.State.UNEXCEPT) : 
                    self.primary = (model, param)
                    return True

        return False
    

    def get_primary_model(self):
        return self.primary[0]

    def get_primary_param(self):
        return self.primary[1]


    def get_param_path(self, param_name):
        return os.path.join(self.project_rootdir, self.params_dirname, param_name+self.param_ext)

    def get_model_path(self, model_name):
        return os.path.join(self.project_rootdir, self.models_dirname, model_name+self.model_ext)


def test():
    manager = Manager(ROOT_DIR, "models", "params", "results")
    manager.search_params()
    manager.search_models()

    for param in manager.get_param_names():
        for model in manager.get_model_names():
            model, param = manager.get_next()
            print(model, param)

    pass

def main():
    backgraoud_proc = BackGroundProcess()

    manager = Manager(ROOT_DIR, "models", "params", "results")
    

    while(1):
        time.sleep(1)

        ## if isrunning
        if backgraoud_proc.isRunning():
            print(".")
        else:
            print(" ---- Proc is Start --- ")

            n = manager.next()

            if(n == False):
                return
            
            param_name = manager.get_primary_param()
            model_name = manager.get_primary_model()

            param_filepath = manager.get_param_path(param_name)
            model_filepath = manager.get_model_path(model_name)

            runner_filepath = os.path.join(ROOT_DIR, "runner.py")

            args=["python3", runner_filepath,
                    "--model", model_filepath, 
                    "--conf", param_filepath ,
                    ]

            working_dir_path = manager.get_workspace_path(model_name, param_name)


            backgraoud_proc.popen(args, cws=working_dir_path)

            for line in backgraoud_proc.proc.stdout:
                print(line)

            
        # stdout, stderr = backgraoud_proc.read_std()
        # print(stdout)
        # print(stderr)




if __name__ == '__main__':
    main()
