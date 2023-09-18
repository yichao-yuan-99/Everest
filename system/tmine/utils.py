import subprocess
import re
import os
import re
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from . import locations

import os
os.environ['PATH'] = '/usr/local/cuda-11.4/bin:' + os.environ['PATH']

def libNameToLibFile(name):
  return 'lib' + name + '.so'

def libName(filename):
  return re.match(r".*lib(\w+)\.so", filename).group(1) 

def libFileToLibName(filename):
  return re.match(r".*lib(\w+)\.so", filename).group(1) 

def compileModule(file, out):
  command = f'nvcc -O3 --std=c++17 -arch=sm_86 -I../include -Xcompiler -fPIC -shared -o {out} {file}'
  logging.debug(f'build Lib: {command}')
  p = subprocess.Popen(command, shell=True)
  p.wait()
  if p.returncode == 0:
    logging.debug(f'%Successfully build {out}')

def compilePlugin():
  createDirIfNotExist(locations.dynLibRoot())
  compileModule(locations.dynLibSource(), locations.dynLibPath())

def compileDyn():
  compilePlugin('DYN', locations.dynLibSource())

def allLibs(dir):
  """
  Returns a list of all files in the given directory that end with .so.
  """
  files = os.listdir(dir)
  so_files = [dir.rstrip('/') + '/' + file for file in files if file.endswith('.so')]
  lib_name = [libName for filename in so_files]
  return lib_name, so_files

def allTxts(dir):
  files = os.listdir(dir)
  txt_files = [dir.rstrip('/') + '/' + file for file in files if file.endswith('.txt')]
  return txt_files

def createDirIfNotExist(dirpath):
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)