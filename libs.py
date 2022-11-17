from os import listdir
from os.path import isfile, join

def all_lines(path):
  content = ''
  # ignore emoji
  with open(path, encoding='utf-8', errors='ignore') as f:
    content = f.readlines()
  return content

def get_files(path):
  return [ join(path, f) for f in listdir(path) if isfile(join(path, f))]

  