"""
The ReadNetworkData library contains a function that reads in network data
stored in a external *.txt file. Used for assignemnt work in the course
46705 - Power Grid Analysis
"""
import re
import csv

def read_network_data_from_file(filename):
    bus_data = []
    load_data = []
    gen_data = []
    line_data = []
    tran_data = []
    mva_base = 0.0
    data_input_type = '' #label keeping track of the type of model data being read
    #Open the network file and process the data
    with open(filename,newline='')as f:
        csv_reader = csv.reader(f)  # read in all of the data
        for line in csv_reader:
            #check if the line is a header line:
            if "//BEGIN " in line[0]: #New section of data starts with //BEGIN
                if 'BEGIN BUS DATA' in line[0]:
                    data_input_type = 'BUS'
                elif 'BEGIN LOAD DATA' in line[0]:
                    data_input_type = 'LOAD'
                elif 'BEGIN GENERATOR DATA' in line[0]:
                    data_input_type = 'GEN'
                elif 'BEGIN LINE DATA' in line[0]:
                    data_input_type = 'LINE'
                elif 'BEGIN TRANSFORMER DATA' in line[0]:
                    data_input_type = 'TRAN'
                elif 'BEGIN MVA SYSTEM' in line[0]:
                    data_input_type = 'MVA'                   
                else:
                    data_input_type = 'UNSPECIFIED'
                
                continue #read next line which contains data

            
            #Check if there is a comment at the end of the line // is in the data (that is comment) and remove that part
            dummy = []
            for k in line:
                if '//' in k: 
                    dummy.append(k.split('//')[0])
                    break
                else:
                    dummy.append(k)
            line = dummy

            if line[0] == '': # If the line was commented out, skip it..             
                continue
            if line[0].isspace(): # If the line has only whitespace, skip it..             
                continue
            
            
            if data_input_type == 'MVA':
                mva_base = parse_mva_data(line)
            elif data_input_type == 'BUS':
                a = parse_bus_data(line)
                bus_data.append(a)
            elif data_input_type == 'LOAD':
                b = parse_load_data(line)
                load_data.append(b)                
            elif data_input_type == 'GEN':
                d = parse_gen_data(line)
                gen_data.append(d)                
                continue
            elif data_input_type == 'LINE':
               c =  parse_transmission_line_data(line)
               line_data.append(c)
            elif data_input_type == 'TRAN':
               f =  parse_transformer_data(line)
               tran_data.append(f)
            else:
                print('DATA TYPE is unspecified!!!!')
                print('Input not treaded:', line)
            
    

    #create mappping objects from bus_nr to matrix indices and vice verse
    bus_to_ind = {} # empty dictionary
    ind_to_bus = {}
    for ind, line in zip(range(len(bus_data)),bus_data): 
        bus_nr = line[0]
        bus_to_ind[bus_nr] = ind
        ind_to_bus[ind] = bus_nr
            
    return bus_data,load_data,gen_data,line_data,tran_data,mva_base, bus_to_ind, ind_to_bus



#test data for branch:

'''parse_line_data(row_)
parses a string that contains the transmission line data
The input contains following FROM_BUS, TO_BUS, ID, R, X, B_HALF
'''
def parse_transmission_line_data(row_):
    # unpack the values:    
    fr,to,br_id,R,X,B,MVA_rate,X2,X0 = row_[0:9]
    #fr,to,br_id,R,X,B_half = row_[0:6]
    fr = int(fr)    #convert the string to int
    to = int(to)    #convert the string to int
    br_id = re.findall("'([^']*)'",br_id)[0]
#    br_id = re.findall(r"'.*'",br_id)[0] #regular expression to find the id str
#    br_id = br_id[1:-1]     #get the id out of the string
    R = float(R) #convert the R str to float
    X = float(X) #convert the X str to float
    B = float(B) #convert the B_half str to float
    X2 = float(X2)
    X0 = float(X0)
    MVA_rate = float(MVA_rate)
    return [fr,to,br_id,R,X,B,MVA_rate,X2,X0]

    #return [fr,to,br_id,R,X,B_half] 


def parse_mva_data(row_):
    mva_size = row_[0]
    mva_size = float(mva_size)        #convert string to float
    return mva_size 


def parse_gen_data(row_):
    # unpack the values:
    bus_nr, mva_size, p_gen, p_max, q_max, q_min, X1, X2, X0, Xn, grnd = row_[0:11]
    bus_nr = int(bus_nr)     #convert the srting to int
    mva_size = float(mva_size)        #convert string to float
    p_gen = float(p_gen)        #convert string to float
    p_max = float(p_max)        #convert string to float
    q_max = float(q_max)        #convert string to float
    q_min = float(q_min)        #convert string to float
    X1 = float(X1)
    X2 = float(X2)
    X0 = float(X0)
    Xn = float(Xn)
    grnd = bool(grnd)
    return [bus_nr, mva_size, p_gen, p_max, q_max, q_min, X1, X2, X0, Xn, grnd] 



#//BEGIN BUS DATA,(BUS_NR, LABEL, KV_BASE, BUSCODE)
def parse_bus_data(row_):
    # unpack the values:    
    bus_nr, label, vm_init, theta_init, buscode, kv_base, v_min, v_max = row_[0:8]
    bus_nr = int(bus_nr)    #convert the srting to int
    label = re.findall(r'\b.*\b',label)[0] #regular expression to get the label
    vm_init = float(vm_init)        #convert string to float
    theta_init = float(theta_init)  #convert string to float
    buscode = int(buscode)          #convert the buscode str to int
    kv_base = float(kv_base)        #convert string to float
    v_min = float(v_min)            #convert string to float
    v_max = float(v_max)            #convert string to float
    #v0 = vm_init * np.exp(1j*theta_init*np.pi/180)
    return [bus_nr, label, vm_init, theta_init, buscode, kv_base, v_min, v_max] 



#//BEGIN LOAD DATA (BUS_NR, P_load MW, Q_load MVAR)  
def parse_load_data(row_):
    # unpack the values:    
    bus_nr, p_ld, q_ld = row_[0:3]
    bus_nr = int(bus_nr)     #convert the srting to int
    p_ld = float(p_ld)        #convert string to float
    q_ld = float(q_ld)        #convert string to float
    return [bus_nr, p_ld, q_ld] 


def parse_transformer_data(row_):
    # unpack the values:
    fr,to,br_id,R,X,n,ang1,MVA_rate,fr_co,to_co,X2,X0 = row_[0:12]
    #fr,to,br_id,R,X,n,ang1 = row_[0:7]
    fr = int(fr)    #convert the srting to int
    to = int(to)    #convert the string to int
    br_id = re.findall("'([^']*)'",br_id)[0]
#    br_id = re.findall(r"'.*'",br_id)[0] #regular expression to find the id str
#    br_id = br_id[1:-1]     #get the id out of the string
    R = float(R) #convert the R str to float
    X = float(X) #convert the X str to float
    n = float(n) #convert the n str to float
    ang1 = float(ang1) #convert the ang1 str to float
    X2 = float(X2)
    X0 = float(X2)
    MVA_rate=float(MVA_rate)
    return [fr,to,br_id,R,X,n,ang1,MVA_rate,fr_co,to_co,X2,X0] 

    

    