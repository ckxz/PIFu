import os
import shutil

src = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/pp_output_external'
src_obj = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/decimated_obj-dataset_vt'
dst = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/pp_output_external_v.1'

folders = folders = sorted([x for x in os.listdir(src) if not x.startswith('.') and not x.endswith('.txt')], key=int)[:7]

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
                    shutil.copy(os.path.join(src_obj, folder, rep, sculpt + '.obj'), os.path.join(dst, rep, df, 'OBJ', sculpt, sculpt + '.obj'))
            else:
                sculpts = [x for x in os.listdir(os.path.join(src, folder, rep, df)) if not x.startswith('.') and not x.endswith('.txt')]

                for sculpt in sculpts:
                    if not os.path.exists(os.path.join(dst, rep, df, sculpt)):
                        os.mkdir((os.path.join(dst, rep, df, sculpt)))
                    files = [x for x in os.listdir(os.path.join(src, folder, rep, df, sculpt)) if not x.startswith('.') and not x.endswith('.txt')]

                    for file_ in files:
                        shutil.copy(os.path.join(src, folder, rep, df, sculpt, file_), os.path.join(dst, rep, df, sculpt, file_))
