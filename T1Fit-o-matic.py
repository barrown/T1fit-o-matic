# coding: utf-8

"""
Takes NMR data path as argument.
Opens processed satt1 spectrum, integrates each slice with limits from title, 
fits to stretched exponential and sets D1 of next EXPNO to (5**(1/beta))*T1

Requires a line in title file starting with the word "limits".
Takes the last two items in that line (delimited by spaces, comma optional) as
upper and lower bounds for integration (high low or low high is okay).
Forces D1 to be at least 0.1s.

Target time is set by an optional line in title file starting with the word "target"
Takes the second item in that line (delimited by spaces, comma optional) as the target time in hours
Number of scans (NS) is set to match target time as close as possible (rounded down)
An optional third item on that line to set the length of the phase cycle,
NS will be a multiple of this, default value is 4.
Forces NS to be at least 1.

Creates a png plot of the fit, in the processed data directory
Option to set D1 for maximum s/n rather than fully quantitative ("max s/n") in title file.
Option to specify minimum D1 (x) in title file ("min D1 x")
Catch excessively long T1: set D1 to 10 s and acquire for 20 minutes or so (NS=128)

"""

#Import statements, function definitions and options
import os
import nmrglue as ng
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
import warnings
import logging

#Stops the warning when using guess_udic, as the problem is corrected on the next line.
warnings.simplefilter('ignore', UserWarning)

#defines stretched exponential function
def stretchT1(tau, A, T1, beta, C):
    y = A * (1.0 - (np.exp(-1.0 * (tau/T1)**beta))) + C
    return(y)

#Cleans a line from the title file
def clean_title_line(line):
    line = line.replace(',',' ')
    line = line.replace('=',' ')
    return(line)

#Set default minimum and maximum D1
min_D1 = 0.1
#61200 seconds = 17 hours
max_D1 = 61200

# Logging settings
username = os.environ.get('USERNAME').lower()
logfile = os.path.normpath("C:/Users/{}/Desktop/T1.log".format(username))
logging.basicConfig(filename=logfile ,level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d/%m/%Y %H:%M")

#Read in data and options from title file

#Take path of input data from argument to python call
raw_data_dir = os.path.normpath(sys.argv[1])
# Read data and convert to 64 bit numpy array with scaling from NC_proc
proc_data_dir = os.path.normpath(raw_data_dir + '/pdata/1')

# Experiment ID for logging
path_elements = raw_data_dir.split(os.path.normpath("/"))
exp_id = "{}/{}/{}".format(path_elements[-3], path_elements[-2], path_elements[-1])

dic, data = ng.bruker.read_pdata(proc_data_dir)
data = np.array(data, dtype='int64')
ncproc = dic['procs']['NC_proc']
data = data * 2**ncproc

title = proc_data_dir + '\\title'
with open(title, 'r') as t:
    title_lines = t.readlines()
    
short_title = title_lines[1].strip()

#Default is to use existing NS if no target time is specified    
target = False
#Default to 5*T1 for quantitative spectra
max_sn = False

for line in title_lines:
    line = line.lower()
    #if the word limit appears in a line, take the integration limits from that line, format: "limit <lower> <upper>"
    if 'limit' in line:
        line = clean_title_line(line)
        limits = float(line.split()[-2]), float(line.split()[-1])
    #Optionally specify target time and (optionally) length of phase cycle, format: "target <time in hours> <length of phase cycle>"
    elif 'target' in line:
        line = clean_title_line(line)
        target_line = line.split()
        target = float(target_line[1])
        #If phase cycle length is present, it will be 3rd item on line
        #setpc stores whether a phase cycle has been specified (True) or not (False)
        try:
            pc = int(target_line[2])
        #If it's missing, default to 4 (for one pulse, etc)
        except IndexError:
            pc = 4
            setpc = False
        else:
            setpc = True
    #Title file can request maximum s/n instead of quantitative spectra
    elif 'max s/n' in line:
        max_sn = True
    #Optionally specify minimum D1
    elif 'min d1' in line:
        line = clean_title_line(line)
        min_D1 = float(line.split()[-1])

# Generate ppm scale via unit converter
udic = ng.bruker.guess_udic(dic, data)
udic[1]['sw'] = dic['procs']['SW_p']
udic[1]['obs'] = dic['procs']['SF']
uc = ng.fileiobase.uc_from_udic(udic)
ppm_scale = uc.ppm_scale()

#The SW of the ppm scale is now correct, but with the wrong offset, so calculate how much we need to shift it by.
#"OFFSET" in dic is the position of the first (left-most) data point in the spectrum
offset = dic['procs']['OFFSET'] - ppm_scale[0]

# Get vdlist and trim to number of time slices
data = data[1:]
data = data[np.any(data, axis=1)]

with open(raw_data_dir + '\\vdlist', 'r') as v:
    vdlist = v.readlines()
vdlist = vdlist[1:len(data)+1]
vdlist = np.array([float(t) for t in vdlist])

# Integrate each time slice from the data
integral_list = np.array([], dtype='float64')
start, end = limits

#"Unreference" the start and end values of the integral
start_unref = start - offset
end_unref = end - offset

#Use the unit converter to get the data indices from our unreferenced chemical shifts
min_point = uc(start_unref, 'ppm')
max_point = uc(end_unref, 'ppm')

#Straighten out the min and max if needed
if min_point > max_point:
    min_point, max_point = max_point, min_point
for slice in data:
    #Peak is the range of data we need, store the sum of that range (the integral)
    peak = slice[min_point:max_point + 1]
    integral_list = np.append(integral_list, peak.sum())
integral_list = integral_list/max(integral_list)

#Get probe ID and nucleus:
path_list = raw_data_dir.split(os.path.normpath('/'))[:-1]
with open(os.path.normpath(raw_data_dir + '/shimvalues'), 'r') as s:
    probe_lines = s.readlines()
for line in probe_lines:
    if '#$$PROBEINFO' in line:
        probe_ID = line.split()[-1]
nucleus = path_list[-1].split('_')[0]

# Do the fit
# Crude estimate of T1 from time delay closest to 0.63 of the way to the top of the curve
T1_guess_intensity = 0.63 * (max(integral_list) - min(integral_list)) + min(integral_list)
T1_guess = vdlist[np.abs(integral_list - T1_guess_intensity).argmin()]
init_params = np.array([max(integral_list), T1_guess, 0.75, min(integral_list)])

try:
    popt, pcov = curve_fit(stretchT1, vdlist, integral_list, p0=init_params)
except RuntimeError:
    # Curve fit failed
    D1 = 10
    target = True
    ns = 128
    with open(title, 'a') as t:
        t.write('\nCurve fit failed: acquiring 128 scans with D1 = 10 s')
    logging.info("{} : Curve fit failed".format(exp_id))

else:
    # Curve fit worked

    #covariance matrix pcov is converted to one standard deviation errors on the popt values
    perr = np.sqrt(np.diag(pcov))
    A = popt[0]
    T1 = popt[1]
    beta = popt[2]
    C = popt[3]
    T1_error = perr[1]
    beta_error = perr[2]

    #Set D1 to either 5^(1/beta)*T1 (quant) or 1.25^(1/beta)*T1 (max s/n) and writes T1, beta and D1 to the title file
    if max_sn:
        D1 = T1 * (1.25**(1/beta))
    else:
        D1 = T1 * (5**(1/beta))

    #If a target time was specified, calcualte NS to be nearest multiple of phase cycle below target time, or at least one phase cycle.
    if target:
        #If we set a phase cycle, do at least one full phase cycle, otherwise do at least one scan.
        if setpc:
            min_scans = pc
        else:
            min_scans = 1
        target_seconds = target * 3600
        ns = max(int(((target_seconds/max(D1, min_D1))//pc)*pc), min_scans)   

    with open(title, 'a') as t:
        if max_sn:
            t.write('\nT1 = %.3f s, beta = %.2f. Set D1 to %.2f s (for maximum s/n)' % (T1, beta, D1))
        else:
            t.write('\nT1 = %.3f s, beta = %.2f. Set D1 to %.2f s' % (T1, beta, D1))

    # If T1 is very large, there was probably a problem with the fit, or at least the experiment will not be worth running.
    # Instead run a short experiment with D1 of 10 s.
    if D1 > max_D1:
        D1 = 10
        target = True
        ns = 128
        with open(title, 'a') as t:
            t.write('\nFound D1 is too long: acquiring 128 scans with D1 = 10 s')
    
    # Makes and stores plot of data in processed data directory
    # File name for plot
    fname_out = proc_data_dir + '\\t1plot.png'
    # Creates a new tau list with 200 points (equally spaced when plotted on log axis) used for smooth line on plot output.
    tau_smooth = np.logspace(np.log10(vdlist.min()), np.log10(vdlist.max()), num = 200)
    fig = plt.figure(figsize=(5.12,3.54)) # make figure 5.12 in x 3.54 in = 13 cm x 9 cm
    ax = fig.add_axes([0.12, 0.15, 0.85, 0.79])  # [left, bottom, width, height] values in 0-1 relative figure coordinates          
    ax.plot(vdlist, integral_list, 'ko')
    ax.plot(tau_smooth, stretchT1(tau_smooth, A, T1, beta, C), 'r')
    ax.set_xscale('log')
    ax.set_title(short_title + ': Integral between %.1f ppm and %.1f ppm' %(float(limits[0]), float(limits[1])), fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    #textstr = "T1 = (%.3f \u00B1 %.3f) s\nbeta = (%.2f \u00B1 %.2f)" %(T1, T1_error, beta, beta_error)
    textstr = '$T_1 = ({:.3f} \pm {:.3f})$ s\n$\\beta = ({:.2f} \pm {:.2f})$'.format(T1, T1_error, beta, beta_error)
    ax.text(0.03, 0.94, textstr, fontsize=12,verticalalignment='top', horizontalalignment='left', transform = ax.transAxes)
    ax.set_xlabel('$T_1$ relaxation time in seconds')
    ax.set_ylabel('Normalised Integrated Intensity')
    #saves the figure in png format
    fig.savefig(fname_out, format='png')
    #clears the figure
    fig.clf()

    logging.info("{} : T1 = {:.3f}, beta = {:.2f}".format(exp_id, T1, beta))
        
# Opens the acqu file of the next EXPNO and sets D1
#Get the location of the acqu file for EXPNO + 1
next_exp_no = int(raw_data_dir.split(os.path.normpath('/'))[-1]) + 1
path_list.extend([str(next_exp_no), '/acqu'])
next_acqu = os.path.normpath('/'.join(path_list))

#Open the next acqu file and edit the D1 value to be 5*T1, or at least 0.1 s.
try:
    with open(next_acqu, 'r+') as a:
        acqu_line_list = a.readlines()
        i = acqu_line_list.index('##$D= (0..63)\n')
        dlist = acqu_line_list[i+1].split()
        dlist[1] = str(max(D1, min_D1))
        acqu_line_list[i+1] = " ".join(dlist)
        
        #If a target time was specified, set NS to the appropriate number
        if target:
            for j, line in enumerate(acqu_line_list):
                if '##$NS=' in line:
                    acqu_line_list[j] = '##$NS= ' + str(ns) + '\n'
                    break
        
        #Go to start of file and overwrite with the modified lines to update D1 (and NS)
        a.seek(0,0)
        a.writelines(acqu_line_list)
except FileNotFoundError:
    # Next experiment not created yet
    pass