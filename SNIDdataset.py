import numpy as np
import SNIDsn as snid





def loadDataset(pathdir, snlist):
    dataset = dict()
    with open(snlist) as f:
        lines = f.readlines()
        f.close()
    for sn in lines:
        filename = sn.strip()
        snname = sn.strip().split('.')[0]
        snidObj = snid.SNIDsn()
        snidObj.loadSNIDlnw(pathdir+filename)
        dataset[snname] = snidObj
    return dataset

def subset(dataset, keys):
    subset = {key:dataset[key] for key in keys if key in dataset}
    return subset

def datasetTypeDict(dataset):
    typeinfo = dict()
    for sn in dataset.keys():
        sntype = dataset[sn].type
        if sntype in typeinfo:
            typeinfo[sntype].append(sn)
        else:
            typeinfo[sntype] = [sn]
    for key in typeinfo.keys():
        typeinfo[key] = np.array(typeinfo[key])
    return typeinfo
