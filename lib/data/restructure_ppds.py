import os
import shutil

#src = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/pp_output_external'
src = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/pp_output_external_v.1/Bust/RENDER'
src_obj = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/decimated_obj-dataset_vt'
dst = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/pp_output_external_v.1/Bust/GEO/OBJ'

### TEMPRORARY TRICK TO BYPASS UNRESOLVED FILE LOADING DUE TO NAME ENCODING ###
unresolved = [x[:-1].split('/')[-2] for x in open('/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/pp_output_external_v.1/unresolved_obj.txt', 'r').readlines()]
###

folders = folders = sorted([x for x in os.listdir(src) if not x in unresolved and not x.startswith('.') and not x.endswith('.txt')])

for folder in folders:
    if not os.path.exists(os.path.join(dst, folder)):
        os.mkdir(os.path.join(dst, folder))
    if not os.path.exists(os.path.join(dst, folder, folder + '.obj')):
        folder_obj = sorted([x for x in os.listdir(src_obj) if not x.startswith('.') and not x.endswith('.txt')], key=int)[:7]
        for folder_ in folder_obj:
            reps = [x for x in os.listdir(os.path.join(src_obj, folder_)) if not x.endswith('.txt') and not x.startswith('.')]
            for rep in reps:
                sculpts = [x for x in os.listdir(os.path.join(src_obj, folder_, rep)) if
                           not x.startswith('.') and not x.endswith('.txt')]
                for sculpt in sculpts:
                    if sculpt == folder + '.obj':
                        shutil.copy(os.path.join(src_obj, folder_, rep, sculpt), os.path.join(dst, folder, folder + '.obj'))
                    else:
                        continue





"""
folders = sorted([x for x in os.listdir(src) if not x.startswith('.') and not x.endswith('.txt')], key=int)[:7]

unresolved_obj = []

for folder in folders:
    reps = [x for x in os.listdir(os.path.join(src, folder)) if not x.endswith('.txt') and not x.startswith('.')]
    for rep in reps:
        if not os.path.exists(os.path.join(dst, rep)):
            os.mkdir(os.path.join(dst, rep))
        data_folders = [x for x in os.listdir(os.path.join(src, folder, rep)) if not x.startswith('.') and not x.endswith('.txt')]

        for df in data_folders:
            if not os.path.exists(os.path.join(dst, rep, df)):
                os.mkdir(os.path.join(dst, rep, df))
            if df == 'GEO':
                if not os.path.exists(os.path.join(dst, rep, df, 'OBJ')):
                    os.mkdir(os.path.join(dst, rep, df, 'OBJ'))
                sculpts = [x for x in os.listdir(os.path.join(src, folder, rep, df, 'OBJ')) if not x.startswith('.') and not x.endswith('.txt')]

                for sculpt in sculpts:
                    if not os.path.exists(os.path.join(dst, rep, df, 'OBJ', sculpt)):
                        os.mkdir(os.path.join(os.path.join(dst, rep, df, 'OBJ', sculpt)))
                    try:
                        shutil.copy(os.path.join(src_obj, folder, rep, sculpt + '.obj'), os.path.join(dst, rep, df, 'OBJ', sculpt, sculpt + '.obj'))
                    except FileNotFoundError:
                        unresolved_obj.append(os.path.join(dst, rep, df, 'OBJ', sculpt, sculpt + '.obj'))
                        with open(os.path.join(src, 'unresolved_obj.txt'), 'w') as f:
                            for element in unresolved_obj:
                                f.write(f'{element}\n')
            else:
                sculpts = [x for x in os.listdir(os.path.join(src, folder, rep, df)) if not x.startswith('.') and not x.endswith('.txt')]

                for sculpt in sculpts:
                    if not os.path.exists(os.path.join(dst, rep, df, sculpt)):
                        os.mkdir((os.path.join(dst, rep, df, sculpt)))
                    files = [x for x in os.listdir(os.path.join(src, folder, rep, df, sculpt)) if not x.startswith('.') and not x.endswith('.txt')]

                    for file_ in files:
                        shutil.copy(os.path.join(src, folder, rep, df, sculpt, file_), os.path.join(dst, rep, df, sculpt, file_))
"""
