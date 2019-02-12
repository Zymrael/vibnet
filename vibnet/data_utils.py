# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:31:46 2018

@author: Zymieth
"""

from .imports import *
from .utils import readFromFolder

def CWRU_dataLoaders(PATH='L:/vibnet-hankooktire/CWRUdata/'):
    '''
    Returns a train DataLoader (2080 time series) and validation DataLoader (892)
    of CWRU mechanical bearing fault detection dataset. 
    The classes are balanced (baseline, ball fault, outer race, inner race)
    '''
        
    ## reading from folders
    P = f'{PATH}baseline/'
    baselined = readFromFolder(P)
    P = f'{PATH}ball/'
    balld = readFromFolder(P)
    P = f'{PATH}inner_race/'
    iraced = readFromFolder(P)
    P = f'{PATH}outer_race/'
    oraced = readFromFolder(P)
        
    baseDE = np.array(baselined[0]['X100_DE_time'])
    baseDE = np.append(baseDE, baselined[1]['X097_DE_time'])
    baseDE = np.append(baseDE, baselined[2]['X098_DE_time'])
    baseDE = np.append(baseDE, baselined[3]['X099_DE_time'])
    
    ballDE = np.array(balld[0]['X118_DE_time'])
    ballDE = np.append(ballDE, balld[1]['X119_DE_time'])
    ballDE = np.append(ballDE, balld[2]['X120_DE_time'])
    ballDE = np.append(ballDE, balld[3]['X121_DE_time'])
    ballDE = np.append(ballDE, balld[4]['X185_DE_time'])
    ballDE = np.append(ballDE, balld[5]['X186_DE_time'])
    ballDE = np.append(ballDE, balld[6]['X187_DE_time'])
    ballDE = np.append(ballDE, balld[7]['X188_DE_time'])
    ballDE = np.append(ballDE, balld[8]['X222_DE_time'])
    ballDE = np.append(ballDE, balld[9]['X223_DE_time'])
    ballDE = np.append(ballDE, balld[10]['X224_DE_time'])
    ballDE = np.append(ballDE, balld[11]['X225_DE_time'])
    
    oraceDE = np.array(oraced[0]['X130_DE_time'])
    oraceDE = np.append(oraceDE, oraced[1]['X131_DE_time'])
    oraceDE = np.append(oraceDE, oraced[2]['X132_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[3]['X133_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[4]['X144_DE_time'])
    oraceDE = np.append(oraceDE, oraced[5]['X145_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[6]['X146_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[7]['X147_DE_time'])
    oraceDE = np.append(oraceDE, oraced[8]['X156_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[9]['X158_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[10]['X159_DE_time'])
    oraceDE = np.append(oraceDE, oraced[11]['X160_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[12]['X197_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[13]['X198_DE_time'])
    oraceDE = np.append(oraceDE, oraced[14]['X199_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[15]['X200_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[16]['X234_DE_time'])
    oraceDE = np.append(oraceDE, oraced[17]['X235_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[18]['X236_DE_time'])
    oraceDE = np.append(oraceDE, oraced[19]['X237_DE_time'])
    oraceDE = np.append(oraceDE, oraced[20]['X246_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[21]['X247_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[22]['X248_DE_time'])
    oraceDE = np.append(oraceDE, oraced[23]['X249_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[24]['X258_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[25]['X259_DE_time'])
    oraceDE = np.append(oraceDE, oraced[26]['X260_DE_time'])
    #oraceDE = np.append(oraceDE, oraced[27]['X261_DE_time']
    
    iraceDE = np.array(iraced[0]['X105_DE_time'])
    iraceDE = np.append(iraceDE, iraced[1]['X106_DE_time'])
    iraceDE = np.append(iraceDE, iraced[2]['X107_DE_time'])
    iraceDE = np.append(iraceDE, iraced[3]['X108_DE_time'])
    iraceDE = np.append(iraceDE, iraced[4]['X169_DE_time'])
    iraceDE = np.append(iraceDE, iraced[5]['X170_DE_time'])
    iraceDE = np.append(iraceDE, iraced[6]['X171_DE_time'])
    iraceDE = np.append(iraceDE, iraced[7]['X172_DE_time'])
    iraceDE = np.append(iraceDE, iraced[8]['X209_DE_time'])
    iraceDE = np.append(iraceDE, iraced[9]['X210_DE_time'])
    iraceDE = np.append(iraceDE, iraced[10]['X211_DE_time'])
    iraceDE = np.append(iraceDE, iraced[11]['X212_DE_time'])
    
    
    # Preparing pd.DataFrame for data labelled 0 (no fault)
    
    n_samples = int(np.floor((len(baseDE))/(0.25*8192)))
    n_samples
    
    base_train = np.empty((829,2048), dtype = float)
    
    for i in range(n_samples-1):
        base_train[i] = baseDE[i*2048:(i+1)*2048]
    
    y = np.zeros((1,829),dtype=int)
    
    basedf = pd.DataFrame(data=base_train)
    
    basedf[2048] = y.squeeze(0)
    
    # Preparing pd.DataFrame for data labelled 1 (ball fault)
    
    n_samples = int(np.floor((len(ballDE))/(0.25*8192)))
    n_samples
    
    ball_train = np.empty((714,2048), dtype = float)
    
    for i in range(n_samples-1):
        ball_train[i] = ballDE[i*2048:(i+1)*2048]
        
    y = np.ones((1,714),dtype=int)
    
    balldf = pd.DataFrame(data=ball_train)
    balldf[2048] = y.squeeze(0)
    
    # Preparing pd.DataFrame for data labelled 2 (inner race fault)
    
    n_samples = int(np.floor((len(iraceDE))/(0.25*8192)))
    #n_samples = 714
    
    irace_train = np.empty((714,2048), dtype = float)
    
    for i in range(n_samples-1):
        irace_train[i] = iraceDE[i*2048:(i+1)*2048]
        
    y = np.full((1,714), 2, dtype=int)
    
    iracedf = pd.DataFrame(data=irace_train)
    iracedf[2048] = y.squeeze(0)
    
    # Preparing pd.DataFrame for data labelled 3 (outer race fault)
    
    n_samples = int(np.floor((len(oraceDE))/(0.25*8192)))
    #n_samples = 714
    
    orace_train = np.empty((714,2048), dtype = float)
    
    for i in range(n_samples-1):
        orace_train[i] = oraceDE[i*2048:(i+1)*2048]
        
    y = np.full((1,714), 3, dtype=int)
    
    oracedf = pd.DataFrame(data=orace_train)
    oracedf[2048] = y.squeeze(0)
    
    # Random uniform sampling, 70% train 30% valid/test
    
    baseline_input = basedf.sample(frac=0.7, random_state=1)
    ball_input = balldf.sample(frac=0.7, random_state=1)
    irace_input = iracedf.sample(frac=0.7, random_state=1)
    orace_input = oracedf.sample(frac=0.7, random_state=1)
    
    baseline_valid = basedf.drop(baseline_input.index)
    ball_valid = balldf.drop(ball_input.index)
    irace_valid = iracedf.drop(irace_input.index)
    orace_valid = oracedf.drop(orace_input.index)
    
    concatdf = [baseline_input, ball_input, irace_input, orace_input]
    concatdf_valid = [baseline_valid, ball_valid, irace_valid, orace_valid]
    x = pd.concat(concatdf)
    x_val = pd.concat(concatdf_valid)
    
    y = x[2048]
    y_val = x_val[2048]
    
    x.drop(x.columns[len(x.columns)-1], axis=1, inplace=True)
    x_val.drop(x_val.columns[len(x_val.columns)-1], axis=1, inplace=True)
    
    # Data massaging for PyTorch DataLoaders
    
    torch.cuda.is_available()
    
    #using float so we don't run into float vs double inconsistencies during training
    x = torch.from_numpy(np.array(x)).float().cuda()
    x_val = torch.from_numpy(np.array(x_val)).float().cuda()
    y = torch.from_numpy(np.array(y)).long().cuda()
    y_val = torch.from_numpy(np.array(y_val)).long().cuda()
    
    train = TensorDataset(x, y)
    val = TensorDataset(x_val, y_val)
    
    trainloader = DataLoader(train, batch_size=1,
                            shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=1,
                            shuffle=True, num_workers=0)

    return(trainloader, valloader)