import numpy as np
import pickle





def getType(tp, subtp):
    """
Convert tuple type designation from SNID to string.
    """
    if tp == 1:
        sntype = 'Ia'
        if subtp == 2: snsubtype = 'norm'
        if subtp == 3: snsubtype = '91T'    
        if subtp == 4: snsubtype = '91bg'
        if subtp == 5: snsubtype = 'csm'
        if subtp == 6: snsubtype = 'pec'
        if subtp == 7: snsubtype = '99aa'
        if subtp == 8: snsubtype = '02cx'
        else: snsubtype = ''
    if tp == 2:
        sntype = 'Ib'
        if subtp == 2: snsubtype = 'norm'
        if subtp == 3: snsubtype = 'pec'
        if subtp == 4:
            sntype = 'IIb'
            snsubtype = ''
        if subtp == 5: snsubtype = 'norm'
        else: snsubtype = ''
    if tp == 3:
        sntype = 'Ic'
        if subtp == 2: snsubtype = 'norm'
        if subtp == 3: snsubtype = 'pec'
        if subtp == 4:
            sntype = 'IcBL'
            snsubtype = ''
        else: snsubtype = ''
    if tp == 4:
        sntype = 'II'
        if subtp == 2: snsubtype = 'P'
        if subtp == 3: snsubtype = 'pec'
        if subtp == 4: snsubtype = 'n'
        if subtp == 5: snsubtype = 'L'
        else: snsubtype = ''
    if tp == 5:
        snsubtype = ''
        if subtp == 1: sntype = 'NotSN'
        if subtp == 2: sntype = 'AGN'
        if subtp == 3: sntype = 'Gal'
        if subtp == 4: sntype = 'LBV'
        if subtp == 5: sntype = 'M-star'
        if subtp == 6: sntype = 'QSO'
        if subtp == 7: sntype = 'C-star'
        else: sntype = ''
    return sntype, snsubtype




class SNIDsn:
    def __init__(self):
        self.header = None
        self.continuum = None
        self.phases = None
        self.phaseType = None
        self.wavelengths = None
        self.data = None
        self.type = None
        self.subtype = None

        return

    def loadSNIDlnw(self, lnwfile):
        """
Loads a SNID .lnw file into a SNIDsn object. Header information from the .lnw file \
is stored in a header dictionary. Other data is stored in the following fields: continuum, phases,\
phaseType, wavelengths, data, type, subtype.
        """
        with open(lnwfile) as lnw:
            lines = lnw.readlines()
            lnw.close()
        header_line = lines[0].strip()
        header_items = header_line.split()
        header = dict()
        header['Nspec'] = int(header_items[0])
        header['Nbins'] = int(header_items[1])
        header['WvlStart'] = float(header_items[2])
        header['WvlEnd'] = float(header_items[3])
        header['SplineKnots'] = int(header_items[4])
        header['SN'] = header_items[5]
        header['dm15'] = float(header_items[6])
        header['TypeStr'] = header_items[7]
        header['TypeInt'] = int(header_items[8])
        header['SubTypeInt'] = int(header_items[9])
        self.header = header

        tp, subtp = getType(header['TypeInt'], header['SubTypeInt'])
        self.type = tp
        self.subtype = subtp

        phase_line_ind = len(lines) - self.header['Nbins'] - 1
        phase_items = lines[phase_line_ind].strip().split()
        self.phaseType = int(phase_items[0])
        phases = np.array([float(ph) for ph in phase_items[1:]])
        self.phases = phases

        wvl = np.loadtxt(lnwfile, skiprows=phase_line_ind + 1, usecols=0)
        self.wavelengths = wvl
        lnwdtype = []
        colnames = []
        for ph in self.phases:
            colname = 'Ph'+str(ph)
            if colname in colnames:
                colname = colname + 'v1'
            while(colname in colnames):
                count = 2
                colname = colname[0:-2] + 'v'+str(count)
                count = count + 1
            colnames.append(colname)
            dt = (colname, 'f4')
            lnwdtype.append(dt)
        #lnwdtype = [('Ph'+str(ph), 'f4') for ph in self.phases]
        data = np.loadtxt(lnwfile, dtype=lnwdtype, skiprows=phase_line_ind + 1, usecols=range(1,len(self.phases) + 1))
        self.data = data

        continuumcols = len(lines[1].strip().split())
        continuum = np.ndarray((phase_line_ind - 1,continuumcols))
        for ind in np.arange(1,phase_line_ind - 1):
            cont_line = lines[ind].strip().split()
            continuum[ind - 1] = np.array([float(x) for x in cont_line])
        self.continuum = continuum
        return

    def wavelengthFilter(self, wvlmin, wvlmax):
        """
Filters the wavelengths to a user specified range, and adjusts the spectra data\
array to reflect that wavelength range.
        """
        wvlfilter = np.logical_and(self.wavelengths < wvlmax, self.wavelengths > wvlmin)
        wvl = self.wavelengths[wvlfilter]
        self.data = self.data[wvlfilter]
        self.wavelengths = self.wavelengths[wvlfilter]
        return
       
    def removeSpecCol(self, colname):
        """
Removes the column named colname from the structured spectra matrix. Removes the \
phase from the list of phases.
        """
        newdtype = [(dt[0], dt[1]) for dt in self.data.dtype.descr if dt[0] != colname]
        newshape = (len(self.data),len(newdtype))
        ndarr = np.ndarray(newshape)
        for i in range(len(newdtype)):
            ndtype = newdtype[i]
            nm = ndtype[0]
            ndarr[:,i] = self.data[nm]
        newstructarr = np.array([tuple(row.tolist()) for row in ndarr], dtype=newdtype)
        newphases = self.phases.tolist()
        newphases.remove(float(colname[2:]))
        self.phases = np.array(newphases)
        self.data = newstructarr
        return


    def snidNAN(self):
        """
SNID uses 0.0 as a placeholder value when observed data is not available for \
a certain wavelength. This method replaces all 0.0 values with NaN.
        """
        colnames = self.getSNCols()
        for col in colnames:
            self.data[col][self.data[col] == 0] = np.nan
        return 

    def getSNCols(self):
        """
Returns spectra structured array column names for the user.
        """
        return self.data.dtype.names

    def save(self, path='./'):
        """
Saves the SNIDsn object using pickle. Filename is automatically generated using \
the sn name stored in the SNIDsn.header['SN'] field.
        """
        filename = self.header['SN']
        f = open(path+filename+'.pickle', 'w')
        pickle.dump(self, f)
        f.close()
        return
