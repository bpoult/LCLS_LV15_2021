import numpy as np
import psana as ps
import sys
import os
os.environ['PS_SRV_NODES'] = '1' #added for SLURM

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from scipy.ndimage.filters import gaussian_filter

def sum_ANDOR(l):
    return np.sum(l-np.average(l[0:500]))

#exp = 'rixx43518' # changeme
exp = 'rixlw1019' # changeme
run_number = int(sys.argv[1])

preprocessed_folder = '/reg/data/ana16/rix/%s/results/preproc/v2/' % exp #changeme
filename = preprocessed_folder+'run%d_v2.h5' % run_number #changeme
# if (os.path.isfile(filename) or os.path.isfile(filename.split('.')[0]+'_v2.h5')):
#     raise ValueError('h5 files for run %d already exist! check folder: %s'%(run_number, preprocessed_folder))
#     # not sure which one to check for so check for both

ds = ps.DataSource(exp=exp, run=run_number)
smd = ds.smalldata(filename=filename)

update = 50 # Print update (per core)
default_val = -9999.0

##########################################
Nfound = 0
Nbad = 0
times = None
##########################################
atm_roi_x = [200,800]
atm_roi_y = [0,1024]

# #####################################################
for run in ds.runs():
    
    # detectors - epics defined below
    timing = run.Detector("timing")
    if ('rix_fim2', 'raw') in run.detinfo:
        rix_fim2 = run.Detector('rix_fim2') 
    else: print('No fim2')
    if ('rix_fim1', 'raw') in run.detinfo:
        rix_fim1 = run.Detector('rix_fim1')
    else: print('No fim1')
    if ('rix_fim0', 'raw') in run.detinfo:
        rix_fim0 = run.Detector('rix_fim0')
    else: print('No fim0')
    if ('xgmd','raw') in run.detinfo:
        xgmd = run.Detector('xgmd')
    else: print('No xgmd')
    if ('gmd','raw') in run.detinfo:
        gmd = run.Detector('gmd')
    else: print('No gmd')
    if ('andor_dir','raw') in run.detinfo:
        andor = run.Detector('andor_dir')
    else: print('No andor')
    if ('ebeam', 'raw') in run.detinfo:
        ebeam = run.Detector('ebeam')
    else: print('No ebeam')
    if ('mono_encoder', 'raw') in run.detinfo:
        mono_encoder = run.Detector('mono_encoder')
    else: print('No mono encoder')
    if ('atmopal', 'raw') in run.detinfo:
        atm = run.Detector('atmopal')
    else: print('No atm')
    
    if hasattr(run, 'epicsinfo'):
        epics_strs = [item[0] for item in run.epicsinfo.keys()][1:] # first one is weird
        epics_detectors = [run.Detector(item) for item in epics_strs]

    for nevent, event in enumerate(run.events()):
        
        if nevent%update==0: print("Event number: %d, Valid shots: %d" % (nevent, Nfound))
            
        data = {'epics_'+epic_str: epic_det(event) for epic_str, epic_det in zip(epics_strs, epics_detectors)}
        
        if any(type(val) not in [int, float] for val in data.values()):
            print("Bad EPICS: %d" % nevent)
            Nbad += 1
            continue

        if ('andor_dir','raw') in run.detinfo:
            andor_data = andor.raw.value(event)
            if andor_data is None:
                print("andor: %d" % nevent)
                Nbad += 1
                continue   
        if ('mono_encoder', 'raw') in run.detinfo:
            mono_data = mono_encoder.raw.value(event)
            if mono_data is None:
                print("Mono encoder: %d" % nevent)
                Nbad += 1
                continue 

        evrs = timing.raw.eventcodes(event)
        if evrs is None:
            print("Bad EVRs: %d" % nevent)
            Nbad += 1
            continue
        evrs = np.array(evrs)
        if evrs.dtype == int:
            data['evrs'] = evrs.copy()
        else:
            print("Bad EVRs: %d" % nevent)
        
        bad = False
        for (detname, method), attribs in run.detinfo.items():
            
            if (detname not in ['timing', 'hsd', 'andor','andor_dir','andor_vls', 'epicsinfo', 'atmopal', 'xtcav', 'manta']) and not \
            (detname=='rix_fim0' and method=='raw') and not (detname=='rix_fim1' and method=='raw') and not \
            (detname=='rix_fim2' and method=='raw'):
                for attrib in attribs:
                    #print(detname, method, attrib)
                    val = getattr(getattr(locals()[detname], method), attrib)(event)
                    if val is None:
                        if detname in ['ebeam', 'gmd', 'xgmd'] and evrs[161]: # BYKIK gives None for these, but we still want to process the shots
                            val = default_val
                        else:
                            bad = True
                            print("Bad %s: %d" % (detname, nevent))
                            Nbad += 1
                            break
                    data[detname+'_'+attrib] = val
        if bad:
            continue

        #ANDOR:
        if ('andor_dir','raw') in run.detinfo:
            andor_data = andor_data.mean(0)
            data['andor'] = andor_data.copy()
            data['andor_sum'] = sum_ANDOR(andor_data)

        # Monochromator encoder
        if ('mono_encoder', 'raw') in run.detinfo:
            print(mono_data)
            data['mono_encoder'] = mono_data.copy()
        
        if ('atmopal', 'raw') in run.detinfo:
            atmim = atm.raw.image(event)
            atm_lo_x = np.average(atmim[atm_roi_x[0]:atm_roi_x[1],atm_roi_y[0]:atm_roi_y[1]],axis = 0)
            atm_lo_y = np.average(atmim[atm_roi_x[0]:atm_roi_x[1],atm_roi_y[0]:atm_roi_y[1]],axis = 1)
            data['atm_lo_x'] = atm_lo_x.copy()
            data['atm_lo_y'] = atm_lo_y.copy()
            #print(atmim.shape)
        
        # # time tool fex
        if ('atmopal','ttfex') in run.detinfo:
            for ii, attribute in enumerate(run.detinfo[('atmopal','ttfex')]):
                #print(attribute)
                # if attribute not in ['calib','image','raw']:
                if attribute in ['fltpos','fltposfwhm','fltpos_ps']:
                    val = getattr(atm.ttfex,attribute)(event)
                    if val is None:
                        print("Bad ttfex {:}: {:}".format(attribute,nevent))
                        # data['ttfex_'+attribute] = -9999.0
                        Nbad += 1
                    else:
                        data['ttfex_'+attribute] = val
                        # print(ii, attribute, np.array(val).shape)
                        
  
        # if ('atmopal','ttfex') in run.detinfo:
        #     for ii, attribute in enumerate(run.detinfo[('atmopal','ttfex')]):
        #         #print(attribute)
        #         if attribute not in ['calib','image','raw']:
        #             val = getattr(atm.ttfex,attribute)(event)
        #             if val is None:
        #                 print("Bad ttfex {:}: {:}".format(attribute,nevent))
        #                 data['ttfex_'+attribute] = -9999.0
        #             else:
        #                 data['ttfex_'+attribute] = val

        # I0 Monitors
        if ('rix_fim0', 'raw') in run.detinfo:
            fim0_raw = np.zeros((8,256))
            for ii, attribute in enumerate(run.detinfo[('rix_fim0', 'raw')]):
                fim_tmp = getattr(rix_fim0.raw,attribute)(event)
                if fim_tmp is None:
                    print("Bad fim0 {:}: {:}".format(attribute,nevent))
                    Nbad += 1
                    continue  
                fim0_raw[ii,:] = fim_tmp
            data['fim0_raw'] = fim0_raw.copy()
        
        if ('rix_fim1', 'raw') in run.detinfo:
            fim1_raw = np.zeros((8,256))
            for ii, attribute in enumerate(run.detinfo[('rix_fim1', 'raw')]):
                fim_tmp = getattr(rix_fim1.raw,attribute)(event)
                if fim_tmp is None:
                    print("Bad fim0 {:}: {:}".format(attribute,nevent))
                    Nbad += 1
                    continue  
                fim1_raw[ii,:] = fim_tmp
            data['fim1_raw'] = fim1_raw.copy()
        
        if ('rix_fim2', 'raw') in run.detinfo:
            fim2_raw = np.zeros((8,256))
            for ii, attribute in enumerate(run.detinfo[('rix_fim2', 'raw')]):
                fim_tmp = getattr(rix_fim2.raw,attribute)(event)
                if fim_tmp is None:
                    print("Bad fim0 {:}: {:}".format(attribute,nevent))
                    Nbad += 1
                    continue  
                fim2_raw[ii,:] = fim_tmp
            data['fim2_raw'] = fim2_raw.copy()
                
        valid_data = True
        for key, val in data.items():
            if (type(val) not in [int, float]) and (not hasattr(val, 'dtype')):
                print("Bad data:", key)
                valid_data = False
                break
        
        if valid_data:
            smd.event(event, **data)
            Nfound += 1
        else:
            Nbad += 1
            continue
        
if smd.summary:
    Nbad = smd.sum(Nbad)
    Nfound = smd.sum(Nfound)
    smd.save_summary(Nfound=Nfound, Nbad=Nbad)
    
smd.done()
    
#if rank == (size - 1):
#    perms = '444' # fo-fo-fo
#    for f in [filename.split('.')[0]+'_part0.h5', filename]:
#        os.chmod(f, int(perms, base=8))
