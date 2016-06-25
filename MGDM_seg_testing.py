# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:41:28 2016

@author: chris
"""

import numpy as np
import nibabel as nb
import os
import sys
sys.path.append('/home/chris/Documents/code/pyProjects/CBSTools/cbstools')
import cbstools as cbs
#import TractREC as tr
#from sklearn.preprocessing import scale

def normalise(img_d):
    return (img_d - np.min(img_d))/np.max(img_d)



    

atlas='/home/chris/mipav/plugins/atlases/brain-segmentation-prior3.0/brain-atlas-3.0.3.txt'
TopologyLUT_dir='/home/chris/Documents/code/python/cbstools-python/ToloplogyLUT/' #NEED THE LAST SLASH for input to JAVA, make a note to add it when taking in directories

data_dir='/home/chris/Documents/code/python/cbstools-python/test-python'
out_dir='/home/chris/Documents/code/python/cbstools-python/test-python/out'

t1_fname='t1map_stripped.nii.gz'
uni_fname='uni_stripped.nii.gz'
pre_fname='filters.nii.gz'

#d,a=tr.imgLoad(os.path.join(data_dir,t1_fname))

t1_i=nb.load(os.path.join(data_dir,t1_fname))
t1_a=t1_i.affine
t1_d=t1_i.get_data()
t1_res=t1_i.header.get_zooms()
t1_res=[a1.item() for a1 in t1_res] #cast to regular python float type

uni_d=nb.load(os.path.join(data_dir,uni_fname)).get_data()
pre_d=nb.load(os.path.join(data_dir,pre_fname)).get_data()

root_fname=os.path.join(data_dir,t1_fname)[0:os.path.join(data_dir,t1_fname).find('.')]

cbs.cj.initVM() #can set the initial and max mem in this initialheap=6000M, maxheap=6000M
mgdm=cbs.cj.BrainMgdmMultiSegmentation2()
mgdm.setTopologyLUTdirectory(TopologyLUT_dir)
mgdm.setDimensions(t1_d.shape[0],t1_d.shape[1],t1_d.shape[2])
mgdm.setResolutions(t1_res[0],t1_res[1],t1_res[2])

#mgdm.setContrastImage1(cbstools.JArray('float')((t1_d.flatten('F')).astype(float)))
#mgdm.setContrastType1('T1MAP7T')

mgdm.setContrastImage1(cbs.jc.JArray('float')((uni_d.flatten('F')).astype(float)))
mgdm.setContrastType1('MP2RAGE7T')

#mgdm.setContrastImage3(cbstools.JArray('float')((pre_d.flatten('F')).astype(float)))
#mgdm.setContrastType3('Filters')

mgdm.setAtlasFile(atlas)

mgdm.setOutputImages('segmentation');
mgdm.setOrientations(mgdm.AXIAL,mgdm.R2L,mgdm.A2P,mgdm.I2S);
mgdm.setAdjustIntensityPriors(False) #default is True
mgdm.setComputePosterior(False)
mgdm.setDiffuseProbabilities(False)
mgdm.setSteps(5)
mgdm.setTopology('wcs')  # {'wcs','no'} no=off for testing, wcs=default


#mgdm.setOrientations(-1,-1,-1,-1);
#mgdm.setSteps(0);
#mgdm.setComputePosterior(False);
#mgdm.setDiffuseProbabilities(False);

#run it
mgdm.execute()

#outputs
seg_im=np.reshape(np.array(mgdm.getSegmentedBrainImage(),dtype=np.uint32),t1_d.shape,'F') #reshape fortran stype to convert back to the format the nibabel likes
lbl_im=np.reshape(np.array(mgdm.getPosteriorMaximumLabels4D(),dtype=np.uint32),t1_d.shape,'F')
ids_im=np.reshape(np.array(mgdm.getSegmentedIdsImage(),dtype=np.uint32),t1_d.shape,'F')


# save
out_im=nb.Nifti1Image(seg_im,t1_a)
nb.save(out_im,os.path.join(out_dir,root_fname+'_seg_cjs_v1.nii.gz'))
out_im=nb.Nifti1Image(lbl_im,t1_a)
nb.save(out_im,os.path.join(out_dir,root_fname+'_lbl_cjs_v1.nii.gz'))
out_im=nb.Nifti1Image(ids_im,t1_a)
nb.save(out_im,os.path.join(out_dir,root_fname+'_ids_cjs_v1.nii.gz'))

"""
Your notes:
If the seg looks a lot like the priors, look at the labels to see what MGDM thinks they are. Remove all of the binary options and run again with 0 steps.
For some data, adjusting the intensity priors messes up the images (as it is not super stable), so removing this option will likely make it work (set to False, default is true)
"""
