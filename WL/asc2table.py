
# coding: utf-8

# In[9]:


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
    print fn
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


def plateToFile(values_in_plate,root, plate_name):
    import os
    import csv
    import numpy as np
    output_folder = os.path.join(root,plate_name)
    os.mkdir(output_folder)
    
    keys = values_in_plate.keys()
    keys.sort(key=alphanum_key)        
    value_matrix = dict()
    out_names = ['OD','GFP_50','GFP_100','GFP_extra']

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
        print "create ", fn    
        with open(fn, mode='w') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)        
            data_writer.writerow(head_line)
            mat = value_matrix[col]
            [r, c] = mat.shape
            for i in range(r):
                a = [i]
                a.extend(mat[i,:].tolist())
                data_writer.writerow(a)
                
def main():
    import csv
    import os
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser(description='Process asc files.')
    parser.add_argument('--path', type=str, default='.', nargs='?',
                    help='folder path containing .asc files')
    args = parser.parse_args()
    print vars(args)

    folder = args.path

    extension = '.asc'
    content = 'Plate'
    fns =[]
    num_plate = []

    for root, dirs, files in os.walk(folder):
        for fn in files:
            if fn.endswith(extension) and content in fn:
                pth = os.path.join(root, fn)
                fns.append(os.path.abspath(pth))
                if getPlateNum(fn, content) not in num_plate:
                    num_plate.append(getPlateNum(fn, content))

    fns = sorted(fns)
    if len(fns) == 0:
	print "No asc files found, please change path"
	return

    files_order_by_plate = dict()
    for val in num_plate:
        my_list = filter(lambda x: val in x, fns)
        files_order_by_plate[val] = my_list

    plates_reader = dict()
    for plate in sorted(files_order_by_plate.keys()):
        list_fns_plate = files_order_by_plate[plate]
        values_in_plate = dict()
        print "Process ",plate
        for fn in list_fns_plate:
            #print plate
            getValuesfromPlate(fn, values_in_plate)
        plates_reader[plate] = values_in_plate 

        # to be changed according to the table
        plate_name = plate[0:plate.find('.')] 
        #od_file = os.pe
        plateToFile(values_in_plate,root, plate_name)
    
                
if __name__ == '__main__':
    main()

