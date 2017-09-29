import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os
import re
from scipy.interpolate import interp1d
from lmfit.models import VoigtModel, LorentzianModel

alpha1 = 1.54056
alpha2 = 1.54439

####################### variables #################################

report_folder = r'./'

exp_ang = {"100": 31.77, "002": 34.422, "101": 36.253, "110": 56.603}
tol_o = {"100": 0.3, "002": 0.3, "101": 0.4, "110": 0.6}


sample_list = ["709"]

##################### XRD FUNCTIONS #########################
"""dataset is an element in the list of dictionaries returned by collectXRDfolder"""


class xrdSample:
    no_peaks = False
    c = 0
    a = 0
    def readXRDfile(self,filename):
        """ Read the Angles and intensities off the file. Assume that the intensity of the AL2O3 peak is constant for all samples because
        the films are thin. Then we can normalise the intensities."""

        with open(filename, "r") as fhandle:
            cor = [x for x in re.findall('\s+([0-9]\S*\.?\S*)\,?\s+([0-9]\S*)\,?', fhandle.read())]
        data = pd.DataFrame()
        data["Angle"] = [float(a[0].strip(",")) for a in cor]
        data["PSD"] = [float(a[1].strip(",")) for a in cor]
        data["PSD"] = data["PSD"]*100/max(data["PSD"])
        return data
    def runFromName(self):
        try:
            run_no,sub = re.findall("(\d\d\d)([rRCcaAmM])", self.filename)
        except:
            run_no,sub = re.findall("(\d\d\d)[-_]Z\S*([rRCcaAmM])-",self.filename)[0]
        self.sub = sub.upper()
        self.run_no = int(run_no)
        return("run: %i sub: %s" %(self.run_no,self.sub))
    def nan_helper(self,y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            #>>> # linear interpolation of NaNs
            #>>> nans, x= nan_helper(y)
            #>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]
    def removeBackground(self):
        """ Performs a simple interpolation of data between peaks, and substracts it. Uses linear interpolation."""
        sub = self.sub
        run_no = self.run_no
        data = self.data.copy()

        x = data.loc[:, "Angle"]
        y = data.loc[:, "PSD"]

        #y = y.rolling(window=10).min()
        #y = y.rolling(window=10).max()
        data.loc[:,"y_mean"] = y.rolling(window=100).mean()
        data.loc[:,'y_back'] = data.loc[:, "y_mean"]

        if sub == "C":
            x_sections = [[20, 29], [39.5, 40], [45, 54], [60, 70], [75, 80]]
        if sub == "R":
            x_sections = [[20, 29], [29, 30], [39, 49], [61, 70], [75, 80]]

        if sub == "A":
            x_sections = []

        if sub == "M":
            x_sections = []

        ##### Find the background intensity to get the magnitude of the peaks ###################
        try:
            no_peaks = ((x > x_sections[0][0]) & (x < x_sections[0][1])) | \
                       ((x > x_sections[1][0]) & (x < x_sections[1][1])) | \
                       ((x > x_sections[2][0]) & (x < x_sections[2][1])) | \
                       ((x > x_sections[3][0]) & (x < x_sections[3][1])) | \
                       ((x > x_sections[4][0]) & (x < x_sections[4][1]))


            # deselect ranges without peaks to get the background
            data["y_back"][~no_peaks] = np.nan

            back_ = data.loc[:, "y_back"].copy()
            yy_mean = back_.rolling(window=5).mean()

            nans, xx = self.nan_helper(yy_mean)
            yy_mean[nans] = np.interp(xx(nans), xx(~nans), yy_mean[~nans])
            ###this line assigns interpolated values to the nans in the background column in the dataframe,
            ###not only to yy


            background = interp1d(x, yy_mean)
            self.data["background"] = background.y
            self.data["y-b.g."] = self.data.loc[:,"PSD"]-background.y
        except:
            self.no_peaks = True
            print(self.filename+": not able to remove background. See removeBackground.")
            self.data["background"] = np.nan
            self.data["y-b.g."] = np.nan
    def extract_peak(self,**kwargs):
        """ Extract the peaks according to locations set in exp_ang. The tolerance, tol_o, need to be adjusted to account for
        poor background treatment (there might be some gradients, so if the range is too wide for the small peaks, the maximum will
        reflect the highest point on a bigger slope.)"""

        exp_ang = {"100": 31.77, "002": 34.422, "101": 36.253, "110": 56.603}
        #exp_ang = {"002": 34.422,"110": 56.603}
        tol_o = {"100": 0.3, "002": 0.3, "101": 0.2, "110": 0.8}

        if len(kwargs.keys())>0:

            for key in kwargs.keys():

                hkl = re.findall("\d\d\d",key)[0]
                value = kwargs[key]
                print(hkl,value)

                tol_o.update({hkl:value})


        for hkl in exp_ang.keys():

            tol = tol_o[hkl]
            theoretical_peak_position = exp_ang[hkl]


            x = self.data.loc[:,"Angle"]
            located_within_tol = abs( x - theoretical_peak_position) < tol
            ###test 1 checks to see if the peak is near the theoretical peak
            temp_peak = self.data[located_within_tol].copy()

            peak_index = temp_peak.idxmax(axis=0)["y-b.g."]
            measured_peak_location = temp_peak.loc[peak_index, "Angle"]

            peak_within_tol = (abs(x - measured_peak_location) < tol)
            self.data["peak"+hkl] = np.nan
            self.data["peak"+hkl] = self.data.loc[:,"y-b.g."][peak_within_tol]
    def fit_peak(self,hkl,keep_out = False):
        """ Model the peak of a given peak using a two voigt function. The functions have the same gamma and sigma, and the amplitude of
        v2_ is half that of v1_. There is a shift between the two, but this is limited to some max value to avoid it from being too big.
        The shift max is chosen through trial and error. """

        max_shift = {"100": 0.0798, "002": 0.091, "101": 0.1158, "110": 0.175}
        from lmfit.models import VoigtModel
        mod1 = VoigtModel(prefix = "v1_")
        mod2 = VoigtModel(prefix = "v2_")
        mod =mod1+mod2
        peak_column = "peak"+hkl

        xy = self.data.loc[:,["Angle",peak_column]].dropna()
        x = xy["Angle"]
        x_labels = x.index
        y = list(xy[peak_column])

        pars = mod.make_params(verbose = False)
        pars['v1_sigma'].set(min=0)
        pars['v2_sigma'].set(expr="v1_sigma")
        pars['v1_amplitude'].set(value=max(y)/2, min=0, max=max(y), vary=True, expr="")
        pars['v2_amplitude'].set(value = max(y)/2,min=0, expr="v1_amplitude/2")

        pars.add('shift', value=max_shift[hkl], min=max_shift[hkl]-0.01, max=max_shift[hkl]+0.01)
        pars['v2_center'].set(expr="v1_center+shift")
        pars['v2_gamma'].set(expr='v1_gamma')


        out = mod.fit(y, pars, x=x)
        if keep_out:
            self.out.update({hkl:out})
        comps = out.eval_components(x=x)

        self.data.loc[x_labels,"v1_comp"+hkl] = comps["v1_"]
        self.data.loc[x_labels,"v2_comp"+hkl] = comps["v2_"]
        self.data.loc[x_labels,"fit"+hkl] = out.best_fit

        fwhm,fwhm_error = re.findall("v1_fwhm:\s+(\S+)\s+\+/-\s+\S+\s+\((\S+)\%\)",out.fit_report())[0]
        fheight,fheight_error = re.findall("v1_height:\s+(\S+)\s+\+/-\s+\S+\s+\((\S+)\%\)",out.fit_report())[0]
        height = max(comps["v1_"])



        if re.search("v1_center:\s+(\S+)\s+\+/-\s+\S+\s+\((\S+)\%\)",out.fit_report()):
            peak_location,peak_location_error = re.findall("v1_center:\s+(\S+)\s+\+/-\s+\S+\s+\((\S+)\%\)",out.fit_report())[0]

        redchisqr = out.redchi
        self.peak.loc[hkl,"fwhm"] = float(fwhm)
        self.peak.loc[hkl,"fwhm_error"] = float(fwhm_error)
        self.peak.loc[hkl,"fheight"] = float(fheight)
        self.peak.loc[hkl,"fheight_error"] = float(fheight_error)
        self.peak.loc[hkl,"height"] = float(height)


        self.peak.loc[hkl,"peak"] = float(peak_location)
        self.peak.loc[hkl,"peak_error"] = float(peak_location_error)
        self.peak.loc[hkl,"redchisqr"] = float(redchisqr)
    ########## Calculated parameters  ########################################
    def peak_params(self):
        """ calculates peak parameters: a, c, d, grain sizes, and errors based on fwhm. inserts values into
        peak_df. Must be used after peaks are modelled."""
        peak_df = self.peak

        def get_a(d, hkl, c = 5.207):
            if len(hkl)==3:
                h = float(hkl[0])
                k = float(hkl[1])
                l = float(hkl[2])
            else:
                print("ERROR: hkl too long: "+hkl)
            a2_inv = (1 / d ** 2 - l ** 2 / c ** 2) / (4.0 / 3.0 * (h ** 2 + k ** 2 + h * k))
            if a2_inv < 0:
                print(h, k, l, "smaller than zero")
            return 1 / math.sqrt(a2_inv)
        def get_c(d, hkl, a = 3.252):
            if len(hkl)==3:
                h = float(hkl[0])
                k = float(hkl[1])
                l = float(hkl[2])
            else:
                print("ERROR: hkl too long: "+hkl)

            c2_inv = 1 / (l ** 2) * (1 / d ** 2 - ((h ** 2 + k ** 2 + h * k) / a ** 2) * 4.0 / 3.0)

            return 1 / math.sqrt(c2_inv)
        def calc_d(two_theta):
            return alpha1 / (2 * math.sin(two_theta * math.pi / 360))
        def lattice_error(two_theta,fwhm):
                """Returns the relative error based on fwhm"""
                return fwhm/2.3548 * math.cos(two_theta/360 * math.pi)/two_theta*2  #fwhm/(2*sqrt(2ln2))
        def scherrer(fwhm,peak_position):

                k = 0.9
                wl = alpha1
                B = fwhm/180*math.pi #must be units radian
                p = peak_position/360 * math.pi #half of 2-theta
                D = k * wl /(B* math.cos(p))
                #### Holzwarth2011
                return D

            ################# Calculate d, a, c, and grain size for each peak ###########################
        for hkl in peak_df.index:
            fwhm = peak_df.loc[hkl,"fwhm"]
            fwhm_error = peak_df.loc[hkl,"fwhm_error"]
            peak_loc = peak_df.loc[hkl, "peak"]

            d = calc_d(peak_loc)
            peak_df.loc[hkl,"d"]=d

            peak_error = lattice_error(peak_loc,fwhm)

            peak_df.loc[hkl,"peak_error"] = peak_error*peak_loc

                        ###### Get the a and c parameters #####
            if hkl == "110":
                peak_df.loc[hkl,"a"] = get_a(d,hkl)
                self.a = get_a(d,hkl)
                peak_df.loc[hkl,"a_error"] = peak_df.loc[hkl,"a"]*lattice_error(peak_loc,fwhm)
            if hkl == "002":
                peak_df.loc[hkl,"c"] = get_c(d, hkl)
                self.c = get_c(d,hkl)
                peak_df.loc[hkl,"c_error"] = peak_df.loc[hkl,"c"]*lattice_error(peak_loc,fwhm)

            if hkl == "100":
                peak_df.loc[hkl,"a"] =  get_a(d, hkl)
                peak_df.loc[hkl,"a_error"] = peak_df.loc[hkl,"a"]*lattice_error(peak_loc,fwhm)
            if hkl == "101":
                if "002" in peak_df.index:
                    dummy_c = get_c(d,"002")
                    peak_df.loc[hkl,"a"] = get_a(d,hkl,c=dummy_c)
                    peak_df.loc[hkl,"a_error"] = peak_df.loc[hkl,"a"]*lattice_error(peak_loc,fwhm)
                    dummy_a = get_a(d,"110")
                    peak_df.loc[hkl,"c"] = get_c(d,hkl,a=dummy_a)
                    peak_df.loc[hkl,"c_error"] = peak_df.loc[hkl,"c"]*lattice_error(peak_loc,fwhm)


            peak_df.loc[hkl,"grain"] = scherrer(fwhm,peak_loc)
            peak_df.loc[hkl,"grain_error"] =fwhm_error/100*scherrer(fwhm,peak_loc)
            ###### calculate u and b for sample #######
    def calc_u(self):
        a = self.a
        c = self.c
        self.u =  1/3*(a**2/c**2)+1/4
    def calc_b(self):
        u = self.u
        c = self.c
        self.b = c*u

    def fit_all_peaks(self,report_folder=""):
        run_no = self.run_no
        sub  = self.sub
        f, ax = plt.subplots(2,2)
        f.suptitle(str(self.run_no) + self.sub)
        ax_l = {"100": ax[0][0], "002": ax[0][1], "101": ax[1][0], "110": ax[1][1]}

        for hkl in ["002","101","100","110"]:
            try:
                self.fit_peak(hkl)
            except:
                print(run_no,sub,hkl)
            #################### plotting and recording results from fit ####################

            ax = ax_l[hkl]
            xy = self.data.loc[:,["Angle","peak"+hkl,"v1_comp"+hkl,"v2_comp"+hkl,"fit"+hkl]].dropna()

            x = xy.loc[:,"Angle"]
            raw = xy.loc[:,"peak"+hkl]
            v1 = xy.loc[:,"v1_comp"+hkl]
            v2 = xy.loc[:,"v2_comp"+hkl]
            fit = xy.loc[:,"fit"+hkl]

            ax.plot(x, raw)
            #ax.plot(x, init, 'k--')

            ax.plot(x, fit, 'r-')
            ax.plot(x, v1, 'b--')
            ax.plot(x, v2, 'g--')

            ax.annotate(hkl, xy=(0.2, 0.8), xycoords='axes fraction', fontsize=16,
                        horizontalalignment='right', verticalalignment='bottom')

            max_y = max(raw)

            ax.set_ylim(0, max_y * 1.1)

        if report_folder == "":
            f.show()
        else:
            f.savefig(report_folder+"/XRD_fit_%s%s.png" % (self.run_no, self.sub), dpi=300)
        plt.close(f)
        self.peak_params()
        self.calc_u()
        self.calc_b()


    def __init__(self,filename):
        self.filename = filename
        self.data = self.readXRDfile(filename)
        self.runFromName()
        self.removeBackground()
        if not self.no_peaks:
            self.extract_peak()
            self.peak = pd.DataFrame(index = ["100","101","002","110"])
            self.out = {}




def collectXRDfolder(folder):
    """ Read all XRD files in folder and return a list of dictionaries.
        XRD_data = get_XRD(folder_with_XRD_files)
    """

    files_in_folder = os.listdir(folder)
    files_to_read = [x for x in files_in_folder if (re.search(".txt", x) and re.search("\d\d\d",x))]

    XRD_data = []
    for inputfile in files_to_read:
            if len(inputfile)<1:
                continue
            try:
                xsample = xrdSample(folder+"/"+inputfile)
                new_sample = {'filename': inputfile,
                              'run_no': xsample.run_no,
                              "sub": xsample.sub,
                              "data": xsample}

                XRD_data.append(new_sample)
            except:
                print(inputfile)
    return XRD_data

def get_rockfolder(folder):
    files_in_folder = os.listdir(folder)
    files_to_list = []
    folder_index = "folderIndex.txt"
    for filename in files_in_folder:
        if re.search(".txt", filename) and filename != folder_index:
            files_to_list.append(filename)

    list_keys = sorted(files_to_list)
    rock_data = []
    for line in list_keys:

        filename = line.strip()
        if len(filename) < 1:
            continue
        try:
            sample, sub = re.findall("(\d\d\d)([rRCcaAmM])", filename)[0]

        except:
            sample, sub = re.findall("(\d\d\d)[-_]\S*([rRCcaAmM])-", filename)[0]

        d_range = re.findall("(\d\d)-(\d\d).txt", filename)[0]

        sub = sub.upper()
        data = pd.DataFrame()

        with open(folder + "/" + filename, "r") as fhandle:

            cor = [x for x in re.findall('\s+([0-9]\S*\.?\S*),?\s+([0-9]\S*),?', fhandle.read())]

            data["Angle"] = [float(a[0].strip(",")) for a in cor]
            data["PSD"] = [float(a[1].strip(",")) for a in cor]

        new_sample = {'filename': filename,
                      'run_no': sample,
                      "sub": sub,
                      "data": data,
                      "range": d_range}

        rock_data.append(new_sample)
    return rock_data

def fit_rocking(dataset, shift, plot_figure=False):
    """
    fit using a single model, return out,comps,init, and list of parameters
    """
    from lmfit.models import PseudoVoigtModel
    import numpy as np
    model = PseudoVoigtModel

    max_shift = {"100": 0.0798, "002": 0.091, "101": 0.1158, "110": 0.175}
    sub = dataset["sub"]
    max_shift = shift

    ############# GET COORDINATES #################
    x = get_x(dataset=dataset)
    y = get_y(dataset=dataset)

    mod1 = model(prefix='v1_')
    mod2 = model(prefix='v2_')

    pars = mod1.guess(y, x=x)
    pars.update(mod2.guess(y, x=x))
    # pars['v1_center'].set( min=34.4, max=35)
    pars['v1_sigma'].set(min=0)
    pars['v1_amplitude'].set(value=max(y), min=0, vary=True, expr="")
    pars['v1_fraction'].set(min=0, max=1)

    pars.add('shift', value=max_shift, min=max_shift - 0.01, max=max_shift + 0.01, vary=True)

    pars['v2_center'].set(expr="v1_center+shift")
    pars['v2_sigma'].set(expr="v1_sigma")
    pars['v2_amplitude'].set(min=0, expr="v1_amplitude/2")
    pars['v2_fraction'].set(min=0, expr='v1_fraction')

    mod = mod1 + mod2

    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    comps = out.eval_components(x=x)
    peak_height = max(comps['v1_'])
    par_list = out.best_values
    par_list.update({"v1_height": peak_height, "v2_height": max(comps['v2_'])})

    if plot_figure:
        plt.plot(x, y)
        plt.plot(x, init, 'k--')

        out = mod.fit(y, pars, x=x)

        comps = out.eval_components(x=x)

        print(out.fit_report())
        plt.yscale("linear")
        plt.plot(x, out.best_fit, 'r-')
        plt.plot(x, comps['v1_'], 'b--')
        plt.plot(x, comps['v2_'], 'g--')
        plt.show()
        print(max(comps['v1_']))

    return out, comps, init, par_list

def fit_rocking_folder(rocking_folder, rock_data, ):
    """

    :param rocking_folder: folder with rocking data
    :param rock_data: list of dictionaries with rocking curve data
    :return: returns nothing, but saves the data in a subfolder xrd_folder/XRD_fit/
    """
    import numpy as np
    report_folder = rocking_folder + "/rocking_fit/"
    max_shift = {"100": 0.0798, "002": 0.091, "101": 0.1158, "110": 0.175}
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    for dataset in rock_data:
        max_shift = {"100": 0.0798, "002": 0.091, "101": 0.1158, "110": 0.175}
        run_no = dataset["run_no"]
        sub = dataset["sub"]
        d_range = dataset["range"]
        print(run_no + sub)
        f, ax = plt.subplots(1, 1)
        f.suptitle(run_no + sub)
        if sub == "R":
            max_shift = max_shift["110"]
        if sub == "C":
            max_shift = max_shift["002"]
        else:
            max_shift = 0.15

        with open(report_folder + "rock_report_%s%s_%s-%s.txt" % (run_no, sub,d_range[0],d_range[1]), "w") as rep_fh:

            out, comps, init, par_list = fit_rocking(dataset, shift=max_shift)

            x = get_x(dataset)
            y = get_y(dataset)

            ax.plot(x, y)
            ax.plot(x, init, 'k--')
            ax.set_yscale("linear")
            ax.plot(x, out.best_fit, 'r-')
            ax.plot(x, comps['v1_'], 'b--')
            ax.plot(x, comps['v2_'], 'g--')
            ax.set_yscale("linear")
            ax.set_ylim(min(y), 1.1 * max(y))

            rep_fh.write(out.fit_report())
            rep_fh.write("\n")

        f.savefig(report_folder + "rock_%s%s_%s-%s.png" %(run_no, sub,d_range[0],d_range[1]), dpi=300)
        plt.close("all")

def rockFile(filename):

    with open(filename) as fin:

        print(re.findall("\d\d\d[aAmMrRcC]",filename))
        content = fin.read()
        angles_in_file = re.search("v1_center:\s+(\d+\.\d+)", content)
        if angles_in_file:
            angles = re.findall("v1_center:\s+(\d+\.\d+)", content)

        fractions_in_file = re.search("v1_fraction:\s+(\d+\.\d+)", content)
        if fractions_in_file:
            fractions = re.findall("v1_fraction:\s+(\d+\.\d+)", content)

        amps_in_file = re.search("v1_amplitude:\s+(\-?\d+\.\d+)", content)
        if amps_in_file:
            amplitudes = re.findall("v1_amplitude:\s+(\-?\d+\.\d+)", content)
        sigmas_in_file = re.search("v1_sigma:\s+(\-?\d+\.\d+)", content)
        if sigmas_in_file:
            sigmas = re.findall("v1_sigma:\s+(\-?\d+\.\d+)", content)
        gammas_in_file = re.search("v1_gamma:\s+(\-?\d+\.\d+)", content)
        if gammas_in_file:
            gammas = re.findall("v1_gamma:\s+(\-?\d+\.\d+)", content)
        shift_in_file = re.search("shift:\s+(\d+\.\d+)", content)
        if shift_in_file:
            shifts = re.findall("shift:\s+(\d+\.\d+)", content)
        heights_in_file = re.search("v1_height:\s+(\d+\.\d+)", content)
        if heights_in_file:
            heights = re.findall("v1_height:\s+(\d+\.\d+)", content)
            ### height is taken from max(comp["v1_"])

        i = 0
        rocks_dic = {}
        angle,gamma, sigma, amplitude = None, None, None, None
        shift = None
        fwhm_g,fwhm_l,fwhm_v = None,None,None

        if angles_in_file:
            angle = float(angles[i])
            rocks_dic.update({"peak": angle})

        if fractions_in_file:
            fraction = float(fractions[i])
            rocks_dic.update({"fraction": fraction})

        if sigmas_in_file:
            sigma = float(sigmas[i])
            fwhm_g = 2 * sigma
            rocks_dic.update({"sigma": sigma,
                              "fwhm_g": fwhm_g})
        if gammas_in_file:
            gamma = float(gammas[i])
            fwhm_l = 2 * gamma
            rocks_dic.update({"gamma": gamma,
                              "fwhm_l": fwhm_l})

        if shift_in_file:
            shift = float(shifts[i])
            rocks_dic.update({"shift": shift})

        if heights_in_file:
            height = heights[i]
            rocks_dic.update({"height": height})

        if sigmas_in_file and gammas_in_file:
            fwhm_v = 0.5346*fwhm_l+math.sqrt(0.2166*fwhm_l**2+fwhm_g**2)
            rocks_dic.update({"fwhm": fwhm_v})
        if sigmas_in_file and not gammas_in_file:
            rocks_dic.update({"fwhm": fwhm_g})

    return rocks_dic

def rockFolder(folder, par_or_report="report"):
    """ Collect all fitted data in folder into one dictionary {"run_no":, "sub":, "data":} """

    files_in_folder = os.listdir(folder)
    files_to_list = []
    for filename in files_in_folder:
        if re.search("%s\S+.txt" % par_or_report, filename):
            files_to_list.append(filename)

    sample_dicts = []
    for filename in files_to_list:
        print(filename)
        sample, sub = re.findall("(\d\d\d)([MmAaCcRr])", filename)[0]
        d_range = re.findall("(\d\d)-(\d\d).txt", filename)[0]
        sam = {"run_no": sample, "sub": sub, "range": d_range}
        rock = rockFile(folder+filename)
        for key in rock:
            sam.update({key: rock[key]})
        sample_dicts.append(sam)

    return sample_dicts

def rock_to_list(samples_data,rock_list):
    """

    :param samples_data:
    :param rock_list: = rockFolder()
    :return: returns nothing, but adds fwhm of rocking curve to samples_data
    """
    for r_dat in rock_list:
        for sam_dat in samples_data:
            if r_dat["run_no"] == sam_dat["run_no"] and r_dat["sub"] == sam_dat["sub"]:
                    d_range = r_dat["range"]
                    if d_range[0] in ["12","13"]:
                        rkey = "rocking13"
                        other_key = "rocking23"
                    if d_range[0] in ["23"]:
                        rkey = "rocking23"
                        other_key = "rocking13"

                    sam_dat.update({rkey:r_dat["fwhm"],
                                    other_key: np.nan})
                    continue

def fit_peaks_lor(XRD_data, XRD_folder=report_folder):
    """ Make fits for the expected peaks in the XRD spectra. Stores the results in image files and
        text files. The text files can be added to the XRD_data list by using some other function.

        PeakFile will read a single file and generate a single pandas dataframe
        peakFolder will read all the files in a folder and generate a list of dataframes
        append_peaks_data will append the peak data to samples_data list
    """

    report_folder = XRD_folder + "/XRD_fit_lor"

    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    for dataset in XRD_data:
        # if dataset["sub"]=="R" and dataset["run_no"] in sample_list:

        data = dataset["data"]
        run_no = dataset["run_no"]
        sub = dataset["sub"]
        ###Other data in list entries

        x = data.loc[:, "Angle"]
        y = data.loc[:, "PSD"]

        data["y_mean"] = y.rolling(window=10).mean()
        data['y_back'] = data.loc[:, "y_mean"]

        if dataset["sub"] == "C":
            x_sections = [[20, 29], [39.5, 40], [45, 54], [60, 70], [75, 80]]
        if dataset["sub"] == "R":
            x_sections = [[20, 29], [29, 30], [39, 49], [61, 70], [75, 80]]

        if dataset["sub"] == "A":
            continue

        if dataset["sub"] == "M":
            continue

        ##### Find the background intensity to get the magnitude of the peaks ###################

        no_peaks = ((x > x_sections[0][0]) & (x < x_sections[0][1])) | \
                   ((x > x_sections[1][0]) & (x < x_sections[1][1])) | \
                   ((x > x_sections[2][0]) & (x < x_sections[2][1])) | \
                   ((x > x_sections[3][0]) & (x < x_sections[3][1])) | \
                   ((x > x_sections[4][0]) & (x < x_sections[4][1]))

        # deselect ranges without peaks to get the background
        data["y_back"][~no_peaks] = np.nan

        back_ = data.loc[:, "y_back"].copy()
        yy_mean = back_.rolling(window=100).mean()

        nans, xx = nan_helper(yy_mean)
        yy_mean[nans] = np.interp(xx(nans), xx(~nans), yy_mean[~nans])
        ###this line assigns interpolated values to the nans in the background column in the dataframe,
        ###not only to yy


        background = interp1d(x, yy_mean)

        ##### Select the peaks based on the expected angles (list at top of cell)
        temp_peaks = pd.DataFrame(x, columns=["Angle"])

        ########## SETUP PLOT AND MAKE REPORT FILES #######################################
        f, ax = plt.subplots(2, 2)
        f.suptitle(run_no + sub)
        ax_l = {"100": ax[0][0], "002": ax[0][1], "101": ax[1][0], "110": ax[1][1]}

        rep_fh = open(report_folder + "/XRD_fit_report_lor_%s%s.txt" % (run_no, sub), "w")

        for i, hkl in enumerate(exp_ang):
            # if hkl == "002":
            tol = tol_o[hkl]

            test = (abs(x - exp_ang[hkl]) < tol)
            temp_peaks.loc[:, hkl] = y[test]
            temp_peaks.loc[:, hkl] = temp_peaks.loc[:, hkl] - background(x)

            peak_index = temp_peaks[hkl].idxmax(axis=0)
            peak_location = temp_peaks.loc[peak_index, "Angle"]

            test2 = (abs(x - peak_location) < tol)
            temp_peaks[hkl] = y[test2]
            temp_peaks[hkl] = temp_peaks[hkl] - background(x)

            ################ Fit peak to voigt model and obtain fwhm from fit ########################
            peaks_without_nan = temp_peaks.loc[:, ["Angle", hkl]].dropna()
            x2 = np.array(peaks_without_nan["Angle"])
            y2 = np.array(peaks_without_nan[hkl])

            #################### Modelling of peak ##########################################
            min_ang = exp_ang[hkl]
            max_y = max(y2)

            ####Setup model and variables
            lor1 = LorentzianModel(prefix='v1_')
            lor2 = LorentzianModel(prefix='v2_')

            pars = lor1.guess(y2, x=x2)
            pars.update(lor2.guess(y2, x=x2))

            pars['v1_center'].set(min=min_ang)
            pars['v1_sigma'].set(value=0.1, min=0, vary=True)
            pars['v1_amplitude'].set(value=max_y / 3, min=0, vary=True, expr="")

            pars.add('shift', value=0.1, min=0.05, max=0.2)

            pars['v2_center'].set(min=min_ang + 0.1, expr="v1_center+shift")
            pars['v2_sigma'].set(value = 0.1, vary = True, expr = "v1_sigma")
            pars['v2_amplitude'].set(value=max_y / 6, expr="v1_amplitude/2")

            mod = lor1 + lor2
            ax = ax_l[hkl]

            ##### generate output #####
            try:
                out = mod.fit(y2, pars, x=x2)

                ##### generate data for plots #####
                comps = out.eval_components(x=x2)
                init = mod.eval(pars, x=x2)

                #################### plotting and recording results from fit ####################


                ax.plot(x2, y2)
                ax.plot(x2, init, 'k--')
                ax.plot(x2, out.best_fit, 'r-')
                ax.plot(x2, comps['v1_'], 'b--')
                ax.plot(x2, comps['v2_'], 'g--')

                ax.annotate(hkl, xy=(0.2, 0.8), xycoords='axes fraction', fontsize=16,
                            horizontalalignment='right', verticalalignment='bottom')

                max_y = max(y2)

                ax.set_ylim(0, max_y * 1.1)
                rep_fh.write("Peak: " + hkl + "\n")
                rep_fh.write("Max peak intensity: %f \n" % max(comps['v1_']))
                rep_fh.write(out.fit_report(min_correl=0.25))
            except:
                rep_fh.write("Peak-error: " + hkl + "\n")
                print(run_no + sub, hkl, ": fit error")
                ax.plot(x2, y2)

        rep_fh.close()
        f.savefig(report_folder + "/XRD_fit_lor_%s%s.png" % (run_no, sub), dpi=300)
        plt.close(f)

def fit_xrd_folder(xrd_folder, XRD_data, noise = 1e-2):
    """

    :param xrd_folder: folder with XRD data
    :param XRD_data: list of dictionaries with XRD data
    :param noise: the limit for whether the peaks can be included or not.
    :return: returns nothing, but saves the data in a subfolder xrd_folder/XRD_fit/
    """
    report_folder = xrd_folder + "/XRD_fit/"

    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    for dataset in XRD_data:
        run_no = dataset["run_no"]
        sub = dataset["sub"]
        print(run_no + sub)
        f, ax = plt.subplots(2, 2)
        f.suptitle(run_no + sub)
        ax_l = {"100": ax[0][0], "002": ax[0][1], "101": ax[1][0], "110": ax[1][1]}
        shifts = {"100": ax[0][0], "002": 0.091, "101": ax[1][0], "110": ax[1][1]}

        with open(report_folder + "XRD_report_%s%s.txt" % (run_no, sub), "w") as rep_fh:
            with open(report_folder + "XRD_pars_%s%s.txt" % (run_no, sub), "w") as par_fh:

                for hkl in ["100", "101", "002", "110"]:
                    print(hkl)
                    try:
                        out, comps, init, par_list = single_fit_pvoigt(dataset, hkl)
                    except:
                        continue
                    ax = ax_l[hkl]
                    x, y = extract_peak(dataset, hkl)
                    ax.plot(x, y)
                    ax.plot(x, init, 'k--')
                    ax.set_yscale("log")
                    ax.plot(x, out.best_fit, 'r-')
                    ax.plot(x, comps['v1_'], 'b--')
                    ax.plot(x, comps['v2_'], 'g--')
                    ax.annotate(hkl, xy=(0.2, 0.8), xycoords='axes fraction', fontsize=16,
                                horizontalalignment='right', verticalalignment='bottom')
                    ax.set_yscale("log")
                    ax.set_ylim(min(y), 1.1 * max(y))

                    if par_list["v1_height"] < noise:
                        continue
                    par_fh.write("Peak: %s \n" % hkl)
                    for key in par_list:
                        par_fh.write("%s: %s \n" % (key, par_list[key]))
                    par_fh.write("\n")

                    rep_fh.write("Peak: %s \n" % hkl)
                    rep_fh.write(out.fit_report())
                    rep_fh.write("\n")
        f.savefig(report_folder + "XRD_%s%s.png" % (run_no, sub), dpi=300)
        plt.close()
