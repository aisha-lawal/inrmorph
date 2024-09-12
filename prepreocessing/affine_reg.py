import argparse
import os
import glob
import shutil


# fixed = path +"t0.nii"
# moving = path +"t1.nii, t2, t3..." 

"""
Just change the path default

"""
parser = argparse.ArgumentParser()
parser.add_argument("--fixed", type=str, help="path to template image folder",
                     default="control/MPR-R__GradWarp__B1_Correction/t0.nii")
parser.add_argument("--moving", type=str, help="path to moving image folder", default="control/MPR-R__GradWarp__B1_Correction/")
parser.add_argument("--output", type=str, help="path to output folder", default="control/affine_registered/")
args = parser.parse_args()
fixed = args.fixed
moving = sorted(glob.glob(args.moving+"*.nii")) 
moving = [f for f in moving if not f.endswith('t0.nii')]
output = args.output

def affine_reg():
   
    try:
        shutil.copy(fixed, output)
    except IOError as io_err:
        os.makedirs(os.path.dirname(output))
        shutil.copy(fixed, output)
        
    for mi in moving:
        # affine_matrix = output + mi.split("/")[-1][:-4]+".txt"
        transformed_image = output + mi.split("/")[-1]
        os.system("reg_aladin -ref {} -flo {} -aff affine_matrix.txt".format(fixed, mi)) #generates affine transformation matrix
        #warp
        os.system("reg_resample -ref {} -flo {} -trans affine_matrix.txt -res {}".format(fixed, mi, transformed_image))

    #run synthseg script here

if __name__ == "__main__":
    affine_reg()