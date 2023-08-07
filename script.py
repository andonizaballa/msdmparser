import sys, random, os, subprocess, time, shutil, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from subprocess import Popen, PIPE

from fig_config import (
    add_grid,
    figure_features,
)  # <--- import customized functions

# Define column names for the DataFrames

column_names = [
        "DELTA T",
        "POLAR ANGLE 0-5",
        "POLAR ANGLE 0-30",
        "POLAR ANGLE 0-90",
        "POLAR ANGLE 0-180",
    ]

def main():
    random_seeds = set()
    parent_directory = 'DATA/2. - 1. - 1./'

    while len(random_seeds) < int(sys.argv[1]):
        random_seeds.add(random.randint(10000000, 99999999))
    print(random_seeds)
    i=1
    for seed in random_seeds:
        print('Number of iterarions: ', i)
        directory = str(seed)
        dirpath = make_directories(parent_directory,directory)
        print(dirpath)
        fileWithSeed(seed)
        runFortranNightmare()
        moveShit(dirpath)
        print(dirpath)
        runPrintagt(dirpath)
        deleteShit()
        i+=1

    neutrons_spectra_df_total, protons_spectra_df_total, gamma_spectra_df_total, piminus_spectra_df_total, piplus_spectra_df_total, pi0_spectra_df_total, antineutrons_spectra_df_total, antiprotons_spectra_df_total, kminus_spectra_df_total, kplus_spectra_df_total, k0_spectra_df_total, antik0_spectra_df_total, printagt_spectra_df_total = append_dataframes()


    plot_spectra(neutrons_spectra_df_total, "neutrons_spectra_df_total")
    plot_spectra(protons_spectra_df_total, "protons_spectra_df_total")
    plot_spectra(gamma_spectra_df_total, "gamma_spectra_df_total")
    plot_spectra(piminus_spectra_df_total, "piminus_spectra_df_total")
    plot_spectra(piplus_spectra_df_total, 'piplus_spectra_df_total')
    plot_spectra(pi0_spectra_df_total, 'pi0_spectra_df_total')
    plot_printagt(printagt_spectra_df_total)


def deleteShit():
    os.remove('for034')
    os.remove('for025')
    os.remove('for037')

def moveShit(dirpath):
    shutil.copy('./for034', dirpath)
    shutil.copy('./for025', dirpath)
    shutil.copy('./for037', dirpath)
    shutil.copy('./for033.dat', dirpath)
    shutil.copy('for037', './printagt/')

def runFortranNightmare():

    # Run the first command
    # os.system('./msdm_exe')

    # ps = subprocess.Popen('./msdm_exe',shell=True,stdin=subprocess.PIPE)
    # ps.communicate(os.linesep.join(["y", "y"], text=True))

    p = Popen('./msdm_exe') #NOTE: no shell=True here
    time.sleep(0.1)
    p.communicate(os.linesep.join(["y\n"])) #NOTE: text=True here
    time.sleep(0.1)
    p.communicate(os.linesep.join(["y\n"])) 
    # ps.stdin.write('y')
    # ps.stdin.write('y')
    # with Popen('./msdm_exe', stdin=PIPE, stdout=DEVNULL, bufsize=1,
    #        universal_newlines=True) as process:
    #     time.sleep(2)
    #     # start program with LOAD and filename:    
    #     print("LOAD " + os.path.abspath(r"input_cases\sample.avl"), file=process.stdin)
    #     time.sleep(2)
    #     print(file=process.stdin) # send newline
    #     time.sleep(5)
    #     print(file=process.stdin) # send newline
    #     time.sleep(0.5)
    #     print("QUIT", file=process.stdin)

def runPrintagt(directory):
    os.chdir('./printagt')
    os.system('gfortran printagt.f ')
    os.system('./a.out')
    directory = make_directories('..', directory)
    shutil.copy('spectr.out', directory )
    os.chdir('../')

def fileWithSeed(seed):


    file = open("for033.dat", "w")
    # Write somye text to the file
    file.write('''     Input file FOR033.DAT for hA- and AA-generator of the SHIELD code
IXFIRS, FORMAT(I12):    '''+str(seed)+'''
JPART,  FORMAT(I12):          25  ! JPART= 1-4, 6-11, 21-24, and 25=HI
TINT, FORMAT(F12.3):       400.0
NUCLID, FORMAT(I12):           8
NSTAT,  FORMAT(I12):      100000
LANTI,  FORMAT(I12):           0  ! 0 - Lab. frame; 1 - AntiLab. frame
LSTAR,  FORMAT(I12):           0  ! 0 - no print; 1/2 - short/total print
LCASC,  FORMAT(I12):           0  ! 0 - AGT; 1 - CASCAD
     Incident heavy ion (if JPART.eq.25 only)
APROJ,FORMAT(F12.3):        12.0
ZPROJ,FORMAT(F12.3):         6.0
<------- 20X ------><-- DATA --><---------------- Comments ----------------->
''')
    # Close the file
    file.close()

def create_datafames(FILE_NAME):

    input_file = open(FILE_NAME)

    lines = input_file.readlines()

    column_names = [
        "DELTA T",
        "POLAR ANGLE 0-5",
        "POLAR ANGLE 0-30",
        "POLAR ANGLE 0-90",
        "POLAR ANGLE 0-180",
    ]

    # Definition of DataFrames
    neutrons_spectra_df = pd.DataFrame(columns=column_names)
    protons_spectra_df = pd.DataFrame(columns=column_names)
    gamma_spectra_df = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    piminus_spectra_df = pd.DataFrame(columns=column_names)
    piplus_spectra_df = pd.DataFrame(columns=column_names)
    pi0_spectra_df = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    antineutrons_spectra_df = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    antiprotons_spectra_df = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    kminus_spectra_df = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    kplus_spectra_df = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    k0_spectra_df = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    antik0_spectra_df = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])




    line_i = 0

    # Boolean control variables
    is_neutron = False
    is_piminus = False
    is_antineutrons = False
    is_seed = False 


    # Strips the newline character
    for l in lines:

        line_i += 1

        # Searching for the header name of a given table
        # If tables are in parallel, they are found as the same, as the parser processes line by line
        if re.search("NEUTRONS SPECTRA", l) is not None:
            is_neutron = True
            continue

        if re.search("PI-  SPECTRA", l) is not None:
            is_piminus = True
            continue

        if re.search("ANTINUCLEON AND KAON SPECTRA",l) is not None:
            is_antineutrons = True
            continue

        if re.search("Initial state of the RANLUX random generator.",l) is not None:
            is_seed = True
            continue

        
        if is_seed :

            if re.search("LUX,INT,K1,K2:",l) is not None:
                search = re.findall(r'\d+', l)
                LUX = search[2]
                INT = search[3]
                K1 = search[4]
                K2 = search[5]
                is_seed = False
        # Processing the first group of parallel tables: NEUTRONS SPECTRA, PROTONS SPECTRA, GAMMA spectra
        if is_neutron:

            # Search for the .- character that starts each value
            if re.search("\.-", l) is not None:

                # Finding all numbers with the structure DIGITS.DIGITS
                search = re.findall("[[0-9]+.[0-9]*", l)

                # Extracting values for each
                delta_value = search[0]
                neutrons_spectra_values = search[1:5]
                protons_spectra_values = search[5:9]
                gamma_value = search[9]

                # Dictionaries for appending values to the DataFrame
                neutrons_spectra_append = {
                    "DELTA T": float(delta_value),
                    "POLAR ANGLE 0-5": float(neutrons_spectra_values[0]),
                    "POLAR ANGLE 0-30": float(neutrons_spectra_values[1]),
                    "POLAR ANGLE 0-90": float(neutrons_spectra_values[2]),
                    "POLAR ANGLE 0-180": float(neutrons_spectra_values[3]),
                }
                protons_spectra_append = {
                    "DELTA T": float(delta_value),
                    "POLAR ANGLE 0-5": float(protons_spectra_values[0]),
                    "POLAR ANGLE 0-30": float(protons_spectra_values[1]),
                    "POLAR ANGLE 0-90": float(protons_spectra_values[2]),
                    "POLAR ANGLE 0-180": float(protons_spectra_values[3]),
                }

                # Appending dictionaries to DataFrames
                neutrons_spectra_df = neutrons_spectra_df.append(
                    neutrons_spectra_append, ignore_index=True
                )

                protons_spectra_df = protons_spectra_df.append(
                    protons_spectra_append, ignore_index=True
                )

                gamma_spectra_df = gamma_spectra_df.append(
                    {
                        "DELTA T": float(delta_value),
                        "POLAR ANGLE 0-180": float(gamma_value),
                    },
                    ignore_index=True,
                )
        #Processing the second group of parallel tables: PI- SPECTRA, PI+ SPECTRA, PI0 SPECTRA
        
        if is_piminus:

            # Search for the .- character that starts each value
            if re.search("\.-", l) is not None:

                # Finding all numbers with the structure DIGITS.DIGITS
                search = re.findall("[[0-9]+.[0-9]*", l)

                # Extracting values for each
                delta_value = search[0]
                piminus_spectra_values = search[1:5]
                piplus_spectra_values = search[5:9]
                pi0_value = search[9]

                # Dictionaries for appending values to the DataFrame
                piminus_spectra_append = {
                    "DELTA T": float(delta_value),
                    "POLAR ANGLE 0-5": float(piminus_spectra_values[0]),
                    "POLAR ANGLE 0-30": float(piminus_spectra_values[1]),
                    "POLAR ANGLE 0-90": float(piminus_spectra_values[2]),
                    "POLAR ANGLE 0-180": float(piminus_spectra_values[3]),
                }
                piplus_spectra_append = {
                    "DELTA T": float(delta_value),
                    "POLAR ANGLE 0-5": float(piplus_spectra_values[0]),
                    "POLAR ANGLE 0-30": float(piplus_spectra_values[1]),
                    "POLAR ANGLE 0-90": float(piplus_spectra_values[2]),
                    "POLAR ANGLE 0-180": float(piplus_spectra_values[3]),
                }

                # Appending dictionaries to DataFrames
                piminus_spectra_df = piminus_spectra_df.append(
                    piminus_spectra_append, ignore_index=True
                )

                piplus_spectra_df = piplus_spectra_df.append(
                    piplus_spectra_append, ignore_index=True
                )

                pi0_spectra_df = pi0_spectra_df.append(
                    {
                        "DELTA T": float(delta_value),
                        "POLAR ANGLE 0-180": float(pi0_value),
                    },
                    ignore_index=True,
                )        
            
        if is_antineutrons:

            # Search for the .- character that starts each value
            if re.search("\.-", l) is not None:

                # Finding all numbers with the structure DIGITS.DIGITS
                search = re.findall("[[0-9]+.[0-9]*", l)

                # Extracting values for each
                delta_value = search[0]
                antineutrons_spectra_values = search[1]
                antiprotons_spectra_values = search[2]
                kminus_spectra_value = search[3]
                kplus_spectra_value = search[4]
                k0_spectra_value = search[5]
                antik0_spectra_value = search[6]

                # Dictionaries for appending values to the DataFrame


                antineutrons_spectra_df = antineutrons_spectra_df.append(
                    {
                        "DELTA T": float(delta_value),
                        "POLAR ANGLE 0-180": float(antineutrons_spectra_values),
                    },
                    ignore_index=True,
                )  
                antiprotons_spectra_df = antiprotons_spectra_df.append(
                    {
                        "DELTA T": float(delta_value),
                        "POLAR ANGLE 0-180": float(antiprotons_spectra_values),
                    },
                    ignore_index=True,
                )   
                kminus_spectra_df = kminus_spectra_df.append(
                    {
                        "DELTA T": float(delta_value),
                        "POLAR ANGLE 0-180": float(kminus_spectra_value),
                    },
                    ignore_index=True,
                )        
                kplus_spectra_df = kplus_spectra_df.append(
                    {
                        "DELTA T": float(delta_value),
                        "POLAR ANGLE 0-180": float(kplus_spectra_value),
                    },
                    ignore_index=True,
                )  
                k0_spectra_df = k0_spectra_df.append(
                    {
                        "DELTA T": float(delta_value),
                        "POLAR ANGLE 0-180": float(k0_spectra_value),
                    },
                    ignore_index=True,
                )

                antik0_spectra_df = antik0_spectra_df.append(
                    {
                        "DELTA T": float(delta_value),
                        "POLAR ANGLE 0-180": float(antik0_spectra_value),
                    },
                    ignore_index=True,
                )
        # Once we have processed the group of parallel tables, we have to set the boolean control variable to false
        if (
            is_neutron is True
            and re.search("\.-", l) is None
            and re.search("DELTA T", l) is None
            and re.search("(MEV)", l) is None
        ):
            is_neutron = False

        if (
            is_piminus is True
            and re.search("\.-", l) is None
            and re.search("DELTA T", l) is None
            and re.search("(MEV)", l) is None
        ):
            is_piminus = False
    
        if (
            is_antineutrons is True
            and re.search("\.-", l) is None
            and re.search("DELTA T", l) is None
            and re.search("(MEV)", l) is None
        ):
            is_antineutrons = False     

    input_file.close()

    return neutrons_spectra_df, protons_spectra_df, gamma_spectra_df, piminus_spectra_df, piplus_spectra_df, pi0_spectra_df, antineutrons_spectra_df, antiprotons_spectra_df, kminus_spectra_df, kplus_spectra_df, k0_spectra_df, antik0_spectra_df

def make_directories(parentdirect, directory):
    dirpath = os.path.join(parentdirect, directory)
    try:
        os.mkdir(dirpath)
    except FileExistsError:
        print('Directory {} already exists'.format(dirpath))
    else:
        print('Directory {} created'.format(dirpath))
    return dirpath

def append_dataframes():

    # Definition of DataFrames
    neutrons_spectra_df_total = pd.DataFrame(columns=column_names)
    protons_spectra_df_total = pd.DataFrame(columns=column_names)
    gamma_spectra_df_total = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    piminus_spectra_df_total = pd.DataFrame(columns=column_names)
    piplus_spectra_df_total = pd.DataFrame(columns=column_names)
    pi0_spectra_df_total = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    antineutrons_spectra_df_total = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    antiprotons_spectra_df_total = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    kminus_spectra_df_total = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    kplus_spectra_df_total = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    k0_spectra_df_total = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    antik0_spectra_df_total = pd.DataFrame(columns=["DELTA T", "POLAR ANGLE 0-180"])
    printagt_spectra_df_total = pd.DataFrame(columns=['Energy', 'Energy_Bin_Start', 'Energy_Bin_End', 'Spectrum', 'Unnormalized_Spectrum'])

    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == "for034":
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    neutrons_spectra_df, protons_spectra_df, gamma_spectra_df, piminus_spectra_df, piplus_spectra_df, pi0_spectra_df, antineutrons_spectra_df, antiprotons_spectra_df, kminus_spectra_df, kplus_spectra_df, k0_spectra_df, antik0_spectra_df = create_datafames(file_path)
                    neutrons_spectra_df_total = neutrons_spectra_df_total.append(neutrons_spectra_df, ignore_index=True)
                    protons_spectra_df_total = protons_spectra_df_total.append(protons_spectra_df, ignore_index=True)
                    gamma_spectra_df_total = gamma_spectra_df_total.append(gamma_spectra_df, ignore_index=True)
                    piminus_spectra_df_total = piminus_spectra_df_total.append(piminus_spectra_df, ignore_index=True)
                    piplus_spectra_df_total = piplus_spectra_df_total.append(piplus_spectra_df, ignore_index=True)
                    pi0_spectra_df_total = pi0_spectra_df_total.append(pi0_spectra_df, ignore_index=True)
                   
            if file == "spectr.out":
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    printagt_spectra_df = printagt_spectra(file_path)
                    printagt_spectra_df_total = printagt_spectra_df_total.append(printagt_spectra_df, ignore_index=True)
    return neutrons_spectra_df_total, protons_spectra_df_total, gamma_spectra_df_total, piminus_spectra_df_total, piplus_spectra_df_total, pi0_spectra_df_total, antineutrons_spectra_df_total, antiprotons_spectra_df_total, kminus_spectra_df_total, kplus_spectra_df_total, k0_spectra_df_total, antik0_spectra_df_total, printagt_spectra_df_total

def printagt_spectra(file_path):
    # Initialize lists to store data
    energies = []
    energy_bins_start = []
    energy_bins_end = []
    spectra = []
    unnormalized_spectra = []

    # Read the data from the file and process it
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'Eaver(MeV/A)' in line:
                continue  # Skip the header line

            # Use regular expression to extract numeric values from the line
            values = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line)
            if len(values) < 5:
                continue 
            # Extract data from the line
            energy = float(values[0])
            energy_bin_start = float(values[1])
            energy_bin_end = float(values[2])
            spectrum = float(values[3])
            unnormalized_spectrum = float(values[4])

            # Append data to the lists
            energies.append(energy)
            energy_bins_start.append(energy_bin_start)
            energy_bins_end.append(energy_bin_end)
            spectra.append(spectrum)
            unnormalized_spectra.append(unnormalized_spectrum)

    # Create a DataFrame from the lists
    data = {
        'Energy': energies,
        'Energy_Bin_Start': energy_bins_start,
        'Energy_Bin_End': energy_bins_end,
        'Spectrum': spectra,
        'Unnormalized_Spectrum': unnormalized_spectra
    }
    df = pd.DataFrame(data)

    return df

def calculate_std_mean(df, column):

    grouped_data = pd.DataFrame()
    

    grouped_data = df.groupby("DELTA T")[column].agg(['mean', 'std'])
    grouped_data['count'] = df.groupby("DELTA T")[column].count()

        # Calculate standard deviation of the mean
    grouped_data['std_mean'] = grouped_data['std'] / np.sqrt(grouped_data['count'])
    
    return grouped_data

def calculate_std_mean_printagt(df, column):

    grouped_data = pd.DataFrame()

    grouped_data = df.groupby("Energy")[column].agg(['mean', 'std'])
    grouped_data['count'] = df.groupby("Energy")[column].count()

        # Calculate standard deviation of the mean
    grouped_data['std_mean'] = grouped_data['std'] / np.sqrt(grouped_data['count'])

    return grouped_data

def plot_spectra(df_total,name):
    figure_features()
    column_dataframe = df_total.columns
    particle = name.split('_')
    for column in column_dataframe[1:]:
        df_total[column] = df_total[column]/100000
        grouped_df =  calculate_std_mean(df_total, column)
        plt.step(grouped_df.index, grouped_df['mean'], where='post', marker='o', linestyle='-', label=column)
        plt.errorbar(grouped_df.index, y = grouped_df['mean'], yerr = grouped_df['std_mean'], fmt='none')
        # plt.fill_between(grouped_df.index, grouped_df['mean'])
        # plt.hist(grouped_df.index, grouped_df['mean'], histtype='step', stacked=True, fill=False, label=column)
        # grouped_df.hist(column='mean', bins=20, histtype='step', stacked=True, fill=False, label=column)
        # Set plot title and labels
    plt.title("Scatter Plot")

    #Change legend font and size
    plt.legend(fontsize=11)
    particle_hizt = {'neutrons' : '$n$',
                    'protons' : '$p$',
                    'gamma' : '$\gamma$',
                    'piminus' : '$\pi_{-}$',
                    'piplus' : '$\pi_{+}$',
                    'pi0' : '$\pi_{0}$'
                    }
    #Chnage names axis
    plt.xlabel("$E_{kin}  (MeV)$")
    plt.ylabel(''''''+particle_hizt[particle[0]]+''' spectra''')

    plt.savefig('''figure'''+name+'''.svg''', format='svg', dpi=1200)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()  

def plot_printagt(df_total):
    figure_features()
    column_dataframe = df_total.columns
    column = column_dataframe[4]
    grouped_df =  calculate_std_mean_printagt(df_total, column)
    plt.step(grouped_df.index, grouped_df['mean'], where='post', marker='o', linestyle='-', label=column, color='red')
    plt.errorbar(grouped_df.index, y = grouped_df['mean'], yerr = grouped_df['std_mean'], fmt='none')
    plt.title("Scatter Plot")

    #Change legend font and size
    plt.legend(fontsize=11)

    #Chnage names axis
    plt.xlabel("$E_{kin}  (MeV)$")
    plt.ylabel('$n$ spectra')
    plt.xlim(0.0,1000.0)
    plt.savefig('neutronprintagt.svg', format='svg', dpi=1200)



    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()  

if __name__ == "__main__":
    main()