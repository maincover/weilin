
# coding: utf-8

# In[ ]:


import os


def get_immediate_subdirectories(a_dir,title=''):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and title in name]

def getMatch(string1, string2):
    from difflib import SequenceMatcher
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    return string1[match.a: match.a + match.size] 

def getCommonName(lf):
    answers = []
    for i, fo in enumerate(lf):
        for j, foj in enumerate(lf):        
            if getMatch(fo,foj)=='':
                continue
            answers.append(getMatch(fo,foj))

    l = [[x,answers.count(x)] for x in set(answers)]
    v = -1
    big = []
    for m in l:
        if m[1]>v:
            v= m[1]
            big = m
    common_name = big[0]
    return common_name


def getPlateNum(fn, s0='Plate'):
    p0 = fn.find(s0)    
    return fn[p0:len(fn)]

def generateWellName():
    list1 = map(chr, range(65, 73))
    list2 = range(1,13,1)
    list2 = map(str, list2)  
    well_name = []
    for l in list1:
        for n in list2:
            well_name.append(l+n)
    return well_name

def getValuesfromPlate(fn, values_in_plate):
    import numpy as np
    import re  
    import os
    f = open(fn, 'r')    
    well_names = generateWellName() 
    for line in f:        
        line_content = re.split(r'\t+',line)
        name = line_content[0]
        if name in well_names:            
            od = float(line_content[1])
            try: 
                fluo1 = float(line_content[2])
            except:
                fluo1 = -1         
            try:
                fluo2 = float(line_content[3])
            except:
                fluo2 = -1
            if name not in values_in_plate.keys():
                values_in_plate[name]= []
            values_in_plate[name].append([od,fluo1,fluo2])
    f.close()


def alphanum_key(s):
    import re
    convert = lambda text: int(text) if text.isdigit() else text 
    return [convert(c) for c in re.split('([0-9]+)', s)]


def plateToFile(values_in_plate, plate_name,output_path):
    import csv
    import numpy as np
    output_folder = os.path.join(output_path,plate_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)    
    keys = values_in_plate.keys()
    keys.sort(key=alphanum_key)        
    value_matrix = dict()
    out_names = ['OD','GFP_100','GFP_50','GFP_extra']
    head_line = ['']

    for well in keys:
        head_line.append(well)
        values = values_in_plate[well]
        npvalues = np.asarray(values)
        [r,c] = npvalues.shape
        for col in range(c):        
            v_m = npvalues[:,col]
            v_m = v_m.reshape(-1,1)
            if col not in value_matrix.keys():
                value_matrix[col] = v_m
            else:
                value_matrix[col] = np.concatenate((value_matrix[col], v_m), axis = 1)

    for col in value_matrix.keys():
        out_name = out_names[int(col)]
        fn = os.path.join(output_folder,plate_name+'_'+str(out_name)+'.csv')        
        with open(fn, mode='w') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)        
            data_writer.writerow(head_line)
            mat = value_matrix[col]
            [r, c] = mat.shape
            for i in range(r):
                a = [i]
                a.extend(mat[i,:].tolist())
                data_writer.writerow(a)
    return output_folder
                
def runPreprocessing(folderPath, extension='.asc'):
    import csv
    import os
    import numpy as np
    
    content = 'Plate'
    fns =[]
    num_plate = []
       
    for root, dirs, files in os.walk(folderPath):
        for fn in files:            
            if fn.endswith(extension) and content in fn:
                pth = os.path.join(root, fn)
                fns.append(os.path.abspath(pth))                
                if getPlateNum(fn, content) not in num_plate:
                    num_plate.append(getPlateNum(fn, content))

    files_order_by_plate = dict()
    for val in num_plate:        
        my_list = filter(lambda x: val in x, fns)
        files_order_by_plate[val] = my_list

    plates_reader = dict()
    
    outl = []    
    for plate in sorted(files_order_by_plate.keys()):
        list_fns_plate = files_order_by_plate[plate]
        values_in_plate = dict()        
        list_fns_plate.sort(key=lambda x: os.path.getmtime(x))        
        for fn in list_fns_plate:       
            #print fn
            getValuesfromPlate(fn, values_in_plate)
        plates_reader[plate] = values_in_plate 
        # to be changed according to the table
        plate_name = plate[0:plate.find('.')] 
        #od_file = os.pe
        outputPath = os.path.split(fn)[0]                
        out = plateToFile(values_in_plate, plate_name,outputPath)        
        outl.append(out)
    return outl


def fluo_found(od_path,gfp_path,output_path):
    import numpy
    import argparse
    import csv
    di=csv.excel()
    inputdelimiter=','
    outputdelimiter=','

    di.delimiter=inputdelimiter
    od_file = csv.reader(open(od_path,'rU'),dialect=di) 
    fluo_file = csv.reader(open(gfp_path,'rU'),dialect=di) 

    di.delimiter=outputdelimiter
    out_file = csv.writer(open(output_path, 'wb'),delimiter='\t')


    # We first parse the CSV files to get two numpy arrays, od_table and fluo_table
    #print "od_file %s " % (args.od)

    od_liste=[row for row in od_file]
    fluo_liste=[row for row in fluo_file]

    nb_points=len(od_liste)-1
    assert(len(fluo_liste)-1==nb_points)
    nb_samples=len(od_liste[0])-1
    assert(len(fluo_liste[0])-1==nb_samples)

    od_table=numpy.zeros((nb_points,nb_samples))
    fluo_table=numpy.zeros((nb_points,nb_samples))

    for i in range(0,nb_points):
        for j in range(0,nb_samples):
            od_table[i,j]=od_liste[i+1][j+1]
            fluo_table[i,j]=fluo_liste[i+1][j+1]

    # We choose the OD points wo want
    od_points=numpy.arange(0,1.01,0.02)
    n=len(od_points)
    fluo_od_table=numpy.zeros((n,nb_samples))

    # And for each of these points we interpolate the two surrouding fluo values.
    # If the OD(t) curve is strictly increasing (should be the case in most cases), we have only one candidate pair.
    # Otherwise the OD(t) curve can cross the y=wantedOD line several times, and we just pick the first candidate pairs
    for nbod,od in enumerate(od_points):
        for sample in range(0,nb_samples):
            # We look the two 'surrounding' OD points
            candidates = [i for i in range(0,nb_points-1) if od_table[i,sample]<od and od_table[i+1,sample]>=od]
            if len(candidates)==0:
                fluo_od_table[nbod,sample]=numpy.nan
                continue
            if len(candidates)>1: # and args.verbose
                print "several candidates for interpolation on plate. Choosing the first one."
            point1=candidates[0]
            point2=candidates[0]+1
            # And we perform a linear interpolation
            dist1=abs(od-od_table[point1,sample])
            dist2=abs(od_table[point2,sample]-od)
            t=dist1/(dist1+dist2)
            fluo=fluo_table[point1,sample] * (1-t) + fluo_table[point2,sample] * t
            fluo_od_table[nbod,sample]=fluo

    with open(output_path,'wb') as myfile:
        wrtr = csv.writer(myfile, delimiter='\t', quotechar='"')
        ligne=['od']
        ligne.extend(od_liste[0][1:len(od_liste[0])])
        wrtr.writerow(ligne)
        for nbod,od in enumerate(od_points):
            ligne=[od]
            ligne.extend(fluo_od_table[nbod,:])
            wrtr.writerow(ligne)
            myfile.flush() # whenever you want
            
def makePair(l):
    newL = []
    l = sorted(l)
    for r in l:
        if 'OD' in r and 'GFP' in r:
            continue
        if 'OD' in r:
            odFile = r
            continue
        newL.append(r)    
    out=[]
    for r in newL:
        out.append([odFile,r])
    return out


def run_fluo_found(outl):
    li = dict()
    for plate in outl:    
        print plate
        if plate not in li.keys():
            li[plate]=[]
        for root,d,files in os.walk(plate):
            p = makePair(files)
            
            for content in p:
                folder_pa = root
                od_path = os.path.join(folder_pa,content[0])
                gfp_path = os.path.join(folder_pa,content[1])
                newName = os.path.splitext(content[0])[0] +'_'+ (os.path.splitext(content[1])[0])+'.csv'
                output_path = os.path.join(folder_pa,newName)            
                print output_path
                fluo_found(od_path,gfp_path,output_path)
                li[plate].append(output_path)            
    return li


def parsePlateName(pName):
    import re
    ds = re.findall(r'\d+', pName)
    return int(ds[0])


def getFnwithNum(l, num):
    for m in l:
        name = os.path.split(m)[1]        
        if num == parsePlateName(name):            
            return m

        
def getTreament_Control_list(outl):
    treatments_dict = dict()
    controls_dict = dict()
    for l in outl:
        for f in l.keys():                
            plateName = os.path.split(f)[1]
            plateNum = parsePlateName(plateName)    
            if plateNum > 0 and plateNum < 6:               
                if plateNum not in treatments_dict.keys():
                    treatments_dict[plateNum] = []
                treatments_dict[plateNum].append(l[f])        
                plateNum_control = plateNum + 5       
                if plateNum_control not in controls_dict.keys():
                    controls_dict[plateNum_control] = []            
                k = l[getFnwithNum(l.keys(),plateNum_control)]
                controls_dict[plateNum_control].append(k)
    l_treatement = dict()
    l_control = dict()
    for k in treatments_dict.keys():
        v = treatments_dict[k]
        k_control = k + 5
        v_control = controls_dict[k_control]
        group = dict()
        group_control = dict()    
        for m in range(len(v)):
            for p in range(len(v[m])): 
                #print v[m][p]
                if p not in group.keys():
                    group[p]=[]
                    group_control[p]=[]
                group[p].append(v[m][p])
                group_control[p].append(v_control[m][p])
        l_treatement[k]=group
        l_control[k_control]=group_control
    return (l_treatement, l_control)

def mkdir_p(path):
    import errno    
    import os
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
            
def runStats(treatments, controls,folder_input, numplate,blanksample=11):
    import numpy
    import argparse
    import csv
    import scipy.stats
    import os.path
    import warnings 
    from pylab import *

    if len(treatments) > 0:
        fn0 = os.path.split(treatments[0])[1]
        fn1 = os.path.split(treatments[1])[1]
        cname = getMatch(fn0,fn1)
    else:
        cname = os.path.split(treatments[0])[1]
    cname = os.path.splitext(cname)[0]
    output = os.path.join(folder_input,'RES',cname)
    mkdir_p(output)
    output_figure = os.path.join(output,'figures')
    mkdir_p(output_figure)
    output_figure_raw = os.path.join(output, 'figures_raw')
    mkdir_p(output_figure_raw)
    name='promoters'
    images=True
    outliers = True
    promoterlesspath=None    
    lbonly=False
    blankmedian=False

    #!/usr/bin/env python
    #-*-coding:utf-8 -*

    di=csv.excel()
    di.delimiter='\t'

    treatment_inputs=list(treatments)
    control_inputs=list(controls)
    assert(len(treatment_inputs)==len(control_inputs))
    Nrep=len(treatment_inputs)

    allinputs=list(treatments)
    allinputs.extend(controls)

    # Create the many output files
    ratio_file = csv.writer(open(output+"/summary_ratio.csv", 'wb'),dialect=di)
    pvalues_file = csv.writer(open(output+"/summary_pvalue.csv", 'wb'),dialect=di)
    significant01_file = csv.writer(open(output+"/summary_significant01.csv", 'wb'),dialect=di)
    significant05_file = csv.writer(open(output+"/summary_significant05.csv", 'wb'),dialect=di)
    candidate_file = open(output+"/list_candidates.txt",'wb')

    if outliers:
        outliers_file = open(output+"/outliers.txt",'wb')

    corr_file=dict()
    for inp in allinputs:
        corr_file[inp] = csv.writer(open(output+"/corr_"+os.path.basename(inp), 'wb'),dialect=di)

    med_Amp_file = csv.writer(open(output+"/med_Amp.csv", 'wb'),dialect=di)
    med_LB_file = csv.writer(open(output+"/med_LB.csv", 'wb'),dialect=di)

    # Load correspondence between promoter name and numbers
    if name!=None:
        #numplate=int(os.path.dirname(allinputs[0])[len(os.path.dirname(allinputs[0]))-2:len(os.path.dirname(allinputs[0]))]) # extract number of the currently treated plate using input path. It assumes the two last chars of the directory are digits giving the number of the plate.
        promnames = open(name,'r')
        promname=dict()
        for line in promnames:
            parts=line.split(';')
            if int(parts[0][2:])==numplate:
                promname[parts[1]]=parts[1]+'_'+parts[2][0:len(parts[2])-1]
                #promname[parts[1]]=parts[2][0:len(parts[2])-1] # remove last char = end of line

    # Load all the input files into arrays
    tables=dict()

    first_row=None

    for inp in allinputs:
        file = csv.reader(open(inp, 'rU'),dialect=di)
        file_liste=[row for row in file]
        nb_points=len(file_liste)-1
        nb_samples=len(file_liste[0])-1
        tables[inp]=numpy.zeros((nb_points,nb_samples))
        for i in range(0,nb_points):
            for j in range(0,nb_samples):
                tables[inp][i,j]=file_liste[i+1][j+1]
        if first_row==None:
            first_row=file_liste[0]
        first_column=[file_liste[i+1][0] for i in range(0,nb_points)]

    # Also load the files with median of promoterless
    if promoterlesspath != None:
        file_pl_TMP_C10 = csv.reader(open(promoterlesspath+"meanTMP_C10",'rU'),dialect=di)
        pl_TMP_C10 = [row[0] for row in file_pl_TMP_C10]
        pl_TMP_C10.pop(0)

        file_pl_LB_C10 = csv.reader(open(promoterlesspath+"meanLB_C10",'rU'),dialect=di)
        pl_LB_C10 = [row[0] for row in file_pl_LB_C10]
        pl_LB_C10.pop(0)

        file_pl_TMP_F3 = csv.reader(open(promoterlesspath+"meanTMP_F3",'rU'),dialect=di)
        pl_TMP_F3 = [row[0] for row in file_pl_TMP_F3]
        pl_TMP_F3.pop(0)

        file_pl_LB_F3 = csv.reader(open(promoterlesspath+"meanLB_F3",'rU'),dialect=di)
        pl_LB_F3 = [row[0] for row in file_pl_LB_F3]
        pl_LB_F3.pop(0)

    # Load the files with the fluorescence of LB
    if lbonly:
        lb_value=dict()
        for inp in allinputs:
            af = open(os.path.splitext(inp)[0]+"_fluoLB", 'r')
            lb_value[inp]=float(af.read())


    # Create the output arrays
    ratio=numpy.zeros((nb_points,nb_samples))
    pvalues=numpy.zeros((nb_points,nb_samples))
    treatment_table=numpy.zeros((nb_points,nb_samples))
    control_table=numpy.zeros((nb_points,nb_samples))

    corr_table=dict() # Correct with promoterless or with LB autofluorescence depending on parameters

    for inp in allinputs:
        corr_table[inp]=numpy.zeros((nb_points,nb_samples))
    is_outlier=dict()
    if outliers:
        for inp in allinputs:
            is_outlier[inp]=numpy.zeros((nb_points,nb_samples))

    # Define a function to calculate median
    for i in range(0,nb_points):
        for j in range(0,nb_samples):
            # Substract promoterless to data
            if promoterlesspath != None:
                for inp in treatment_inputs:
                    corr_table[inp][i,j]=tables[inp][i,j]-float(pl_TMP_F3[i])
                for inp in control_inputs:
                    corr_table[inp][i,j]=tables[inp][i,j]-float(pl_LB_F3[i])
            elif blanksample != None:
                if blankmedian:
                    for inp in treatment_inputs:
                        corr_table[inp][i,j]=tables[inp][i,j]-numpy.median([tables[t][i,blanksample] for t in treatments])
                    for inp in control_inputs:
                        corr_table[inp][i,j]=tables[inp][i,j]-numpy.median([tables[t][i,blanksample] for t in controls])
                else:
                    for inp in treatment_inputs:
                        corr_table[inp][i,j]=tables[inp][i,j]-tables[inp][i,blanksample]
                    for inp in control_inputs:
                        corr_table[inp][i,j]=tables[inp][i,j]-tables[inp][i,blanksample]
            elif lbonly: # Other calculation : we only substract LB well (A12), being carreful with possible contamination
                for inp in treatment_inputs:
                    corr_table[inp][i,j]=tables[inp][i,j]-lb_value[inp]
                for inp in control_inputs:
                    corr_table[inp][i,j]=tables[inp][i,j]-lb_value[inp]
            else:
                assert (False)
            # Calculate medians
            treatment_value=numpy.median([corr_table[t][i,j] for t in treatments])
            treatment_table[i,j]=treatment_value
            control_value=numpy.median([corr_table[t][i,j] for t in controls])
            control_table[i,j]=control_value

            # Deal with nan, zeros, negative values, ...
            if treatment_value<0:
                treatment_value=0.001
            if control_value<=0:
                control_value=0
                ratio[i,j]=numpy.nan
            else:
                ratio[i,j]=numpy.log2(treatment_value/control_value)
            warnings.simplefilter("ignore")
            pvalues[i,j]=scipy.stats.ttest_ind([corr_table[t][i,j] for t in treatments],[corr_table[t][i,j] for t in controls],equal_var=True)[1]
            # Try to detect outliers
            if outliers:
                MAD_t=numpy.median([abs(treatment_value - corr_table[t][i,j]) for t in treatments])
                for t in treatments:
                    if abs((corr_table[t][i,j]-treatment_value)/MAD_t)>6:
                        outliers_file.write("%s : %d,%d is an outlier\n" % (t,i,j))
                        is_outlier[t][i,j]=1
                MAD_c=numpy.median([abs(control_value - corr_table[t][i,j]) for t in controls])
                for t in controls:
                    if abs((corr_table[t][i,j]-control_value)/MAD_c)>6:
                        outliers_file.write("%s : %d,%d is an outlier\n" % (t,i,j))
                        is_outlier[t][i,j]=1

    if outliers:
        outliers_file.close()

    # Produce files with ratio and p-value
    ratio_file.writerow(first_row)
    for i in range(0,nb_points):
        ligne=[first_column[i]]
        ligne.extend(ratio[i,:])
        ratio_file.writerow(ligne)

    pvalues_file.writerow(first_row)
    for i in range(0,nb_points):
        ligne=[first_column[i]]
        ligne.extend(pvalues[i,:])
        pvalues_file.writerow(ligne)

    significant01_file.writerow(first_row)
    for i in range(0,nb_points):
        ligne=[first_column[i]]
        ligne.extend([str(ratio[i,j]) if pvalues[i,j]<0.01 else '' for j in range(0,nb_samples)])
        significant01_file.writerow(ligne)

    significant05_file.writerow(first_row)
    for i in range(0,nb_points):
        ligne=[first_column[i]]
        ligne.extend([str(ratio[i,j]) if pvalues[i,j]<0.05 else '' for j in range(0,nb_samples)])
        significant05_file.writerow(ligne)

    # Also produce "intermediate" output (corrected fluo as a function of OD)
    med_Amp_file.writerow(first_row)
    for i in range(0,nb_points):
        ligne=[first_column[i]]
        ligne.extend(treatment_table[i,:])
        med_Amp_file.writerow(ligne)

    med_LB_file.writerow(first_row)
    for i in range(0,nb_points):
        ligne=[first_column[i]]
        ligne.extend(control_table[i,:])
        med_LB_file.writerow(ligne)

    for inp in allinputs:
        corr_file[inp].writerow(first_row)
        for i in range(0,nb_points):
            ligne=[first_column[i]]
            ligne.extend(corr_table[inp][i,:])
            corr_file[inp].writerow(ligne)


    do_points=numpy.arange(0,1.01,0.02)

    if images:
      # Produce corrected curves
        for j in range(0,nb_samples):
            f=figure()
            for inp in controls:
                plot(do_points,corr_table[inp][:,j],'b')
            if outliers:
                outx=[do_points[i] for i in range(0,nb_points) if is_outlier[inp][i,j]==1]
                outy=[corr_table[inp][i,j] for i in range(0,nb_points) if is_outlier[inp][i,j]==1]
                scatter(outx,outy,s=80,facecolors='none',edgecolors='b')
            for inp in treatments:
                plot(do_points,corr_table[inp][:,j],'r')
                if outliers:
                    outx=[do_points[i] for i in range(0,nb_points) if is_outlier[inp][i,j]==1]
                    outy=[corr_table[inp][i,j] for i in range(0,nb_points) if is_outlier[inp][i,j]==1]
                    scatter(outx,outy,s=80,facecolors='none',edgecolors='r')
            if name != None:
                out = os.path.join(output_figure,promname[first_row[j+1]])
                savefig(out)
            else:
                out = os.path.join(output_figure,'sample'+str(j))
                savefig(out)
            close()

      # Produce raw curves
    for j in range(0,nb_samples):
        f=figure()
        for inp in controls:
              plot(do_points,tables[inp][:,j],'b')
        for inp in treatments:
              plot(do_points,tables[inp][:,j],'r')
        if name != None:
            out = os.path.join(output_figure_raw,promname[first_row[j+1]])
            savefig(out)
        else:
            out = os.path.join(output_figure_raw,'sample'+str(j))
            savefig(out)
        close()

    # Produce list of candidates for which several consecutive points are significant
    for j in range(0,nb_samples):
        list_pv=[i for i in range(0,nb_points-1) if ( pvalues[i,j]<0.01 and pvalues[i+1,j]<0.01 )]
        if len(list_pv)>0:
            if name != None:
                candidate_file.write("%s : at OD %f\n" % (promname[first_row[j+1]],do_points[list_pv[0]]) )
            else:
                candidate_file.write("%s : at OD %f\n" % (first_row[j+1],do_points[list_pv[0]]))    
                
 

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process asc files.')
    parser.add_argument('--path', type=str, default='.', nargs='?',
                    help='folder path containing sub folders of replicats')
    args = parser.parse_args()
    print vars(args)
    folder_input = args.path     
    lf = get_immediate_subdirectories(folder_input)
    commonName = getCommonName(lf)
    lf_with_commonName = get_immediate_subdirectories(folder_input,commonName)
    folder_list = []
    outl=[]
    for subFolder in lf_with_commonName:
        input_folder = os.path.join(folder_input,subFolder)
        folder_list.append(runPreprocessing(input_folder))

    for l in folder_list:
        outl.append(run_fluo_found(l))    
    l_treatement, l_control = getTreament_Control_list(outl)

    for k in l_treatement.keys():
        print k
        k_control = k + 5
        n = len(l_treatement[k])
        for i in range(n):
            treatments = sorted(l_treatement[k][i])
            controls = sorted(l_control[k_control][i])
            runStats(treatments,controls, folder_input, numplate=k)
               
if __name__=="__main__":
	main()
    

    
