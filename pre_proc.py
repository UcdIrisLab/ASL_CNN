"""
Create by lzw @ 20170318

Modified by Donghoon Kim at UC Davis 03212023
"""
import argparse

from utils import gen_PCASL_base_datasets_DK, gen_conv3d_PCASL_datasets_DK

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="The path of raw data")
parser.add_argument("--subjects", help="subjects", nargs='*')
parser.add_argument("--base", help="gen base data", action="store_true")
parser.add_argument("--conv3d", help="gen conv3d data", action="store_true")

#file = open('move.txt','r')

args = parser.parse_args()

path = args.path
subjects = args.subjects
base = args.base
conv3d = args.conv3d

if base:
    for subject in subjects:
        gen_PCASL_base_datasets_DK(path, subject, fdata=True, flabel=True)

if conv3d:
    gen_conv3d_PCASL_datasets_DK(path, subjects, 3, 1, base=1, test=False)
