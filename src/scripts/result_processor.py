import sys
import glob
import shutil
import numpy as np
sys.path.append('../')
import raven.src.config as config
import matplotlib.pyplot as plt
import seaborn as sns


    

class DataStruct:
    def __init__(self, eps = None,
    individual_percentage = None,
    IO_percentage = None,
    raven_no_diff_percentage = None,
    raven_percentage = None,
    times = None, MILP_times = None):
        self.eps = eps
        self.individual_percentage = individual_percentage
        self.IO_percentage = IO_percentage
        self.raven_no_diff_percentage = raven_no_diff_percentage
        self.raven_percentage = raven_percentage
        self.times = np.array(times)
        self.MILP_times = np.array(MILP_times)


class DataStructList:
    def __init__(self, thersholds):
        self.data_struct_list = []
        self.thersholds = thersholds

    def append(self, data_struct):
        self.data_struct_list.append(data_struct)

    def generate_plot(self, eps, individual, io_formulation, 
                    raven, raven_no_diff=None, net_name='', domain=''):
        sns.set_style("darkgrid")
        # Plot the three line plots
        plt.figure(figsize=(6, 4.5))  # Optional: set the figure size
        ax = plt.axes()
        ax.grid(axis='x')
        
        # Setting the background color of the plot 
        # using set_facecolor() method
        ax.set_facecolor("lightgrey")
        if 'cifar' in net_name:
            eps = [255 * e for e in eps]
        binary = 'binary' in net_name
        if binary:
            individual = [20 - e/10 for e in individual]
            io_formulation = [20 - e/10 for e in io_formulation]
            raven = [20 - e/10 for e in raven]
            if raven_no_diff is not None:
                raven_no_diff = [20 - e/10 for e in raven_no_diff]
        cls = 'cls' in net_name
        if cls:
            plt.plot(eps, individual, marker='D', label='Interval', linestyle='-', color='cornflowerblue')
            plt.plot(eps, io_formulation, marker='s', label='Zonotope', linestyle='-', color='indianred')
            if raven_no_diff is not None:
                plt.plot(eps, raven_no_diff, marker='+', label='Pasado', linestyle='-', color='purple')
            plt.plot(eps, raven, marker='o', label='RaVeN', linestyle='-', color='darkseagreen')

        else:
            plt.plot(eps, individual, marker='D', label='Individual', linestyle='-', color='cornflowerblue')
            plt.plot(eps, io_formulation, marker='s', label='I/O Formulation', linestyle='-', color='indianred')
            plt.plot(eps, raven, marker='o', label='RaVeN', linestyle='-', color='darkseagreen')
            if raven_no_diff is not None:
                plt.plot(eps, raven_no_diff, marker='+', label='RaVeN Layerwise', linestyle='-', color='black')
        plt.legend(loc=4, fontsize="10")
        # Add labels and a legend
        #plt.gca().yaxis.label.set(rotation='horizontal', ha='left');
        bbox = ax.get_yticklabels()[-1].get_window_extent()
        x,_ = ax.transAxes.inverted().transform([bbox.x0, bbox.y0])
        if binary:
            ax.set_ylim(0.0, 20.0)
        elif cls:
            ax.set_ylim(0.0, 40.0)
        else:
            ax.set_ylim(0.0, 100.0)
        if 'cifar' in net_name:
            plt.xlabel('Epsilon (*/255)', fontsize=15)
        else:
            plt.xlabel('Epsilon', fontsize = 15)
        if binary:
            #plt.ylabel('Average Worst Case Hamming Distance', fontsize=15)
            ax.set_title('Average Worst Case Hamming Distance', fontsize=15, ha='left', x=x)
        elif cls:
            #plt.ylabel('# Verifiably Monotonic Features', fontsize=15)
            ax.set_title('# Verifiably Monotonic Features', fontsize=15, ha='left', x=x)
        else:
            #plt.ylabel('Average Worst Case Accuracy (%)', fontsize=15)
            ax.set_title('Average Worst Case Accuracy (%)', fontsize=15, ha='left', x=x)
        if cls:
            plt.legend(loc=4, fontsize="12")
        else:
            plt.legend(loc=3, fontsize="12")
        plt.tight_layout(pad=0.5)
        dir_name = 'plots'
        diff = '' if raven_no_diff is None else '_diff'
        # plt.show()    
        plot_name = f'{dir_name}/{net_name}_{domain}{diff}.png'
        #print(plot_name)
        plt.savefig(plot_name, dpi=600,  bbox_inches='tight')

    def get_threshold(self, net_name):
        for key in self.thersholds.keys():
            if key in net_name or net_name in key:
                return self.thersholds[key]
        return None

    def plot(self, net_name, domain):
        individual = []
        io_formulation = []
        raven = []
        raven_no_diff = []
        epsilons = []
        temp_dict = {}
        threshold = self.get_threshold(net_name=net_name)
        for data in self.data_struct_list:
            epsilons.append(data.eps)
            # individual.append(data.individual_percentage)  
            # io_formulation.append(data.IO_percentage)
            # raven_no_diff.append(data.raven_no_diff_percentage)
            # raven.append(data.raven_percentage) 
            temp_list = [data.individual_percentage, data.IO_percentage,
                         data.raven_no_diff_percentage, data.raven_percentage]
            temp_dict[data.eps] = temp_list
        epsilons.sort()
        length = 0
        for epsilon in epsilons:
            if threshold is not None:
                if epsilon > threshold[1] or epsilon < threshold[0]:
                    continue
            length += 1
            individual.append(temp_dict[epsilon][0])  
            io_formulation.append(temp_dict[epsilon][1])
            raven_no_diff.append(temp_dict[epsilon][2])
            raven.append(temp_dict[epsilon][3])
        epsilons = epsilons[:length]
        self.generate_plot(eps=epsilons, individual=individual, io_formulation=io_formulation,
                           raven=raven, raven_no_diff=None, net_name=net_name, domain=domain)
        self.generate_plot(eps=epsilons, individual=individual, io_formulation=io_formulation,
                           raven=raven, raven_no_diff=raven_no_diff, net_name=net_name, domain=domain)

    def compute_avg_time(self, net_name):
        avg_timings = np.array([0.0, 0.0, 0.0, 0.0])
        avg_MILP_times = np.array([0.0, 0.0])
        threshold = self.get_threshold(net_name=net_name)
        list_length = 0
        for data in self.data_struct_list:
            eps = data.eps
            if eps > threshold[1] or eps < threshold[0]:
                continue
            avg_timings = avg_timings + data.times
            avg_MILP_times = avg_MILP_times + data.MILP_times
            list_length += 1

        if list_length > 0: 
            avg_timings = avg_timings / list_length
            avg_MILP_times = avg_MILP_times / list_length
        return avg_timings, avg_MILP_times

def copy_files():
    # Specify the source and destination directories
    source_directory = '../cifar_results/'  # Replace with the source directory path
    destination_directory = './raw_results/'  # Replace with the destination directory path

    # Use shutil.copy() to copy all files from the source directory to the destination directory
    try:
        shutil.copytree(source_directory, destination_directory)
        print("All files copied successfully.")
    except FileNotFoundError:
        print("Source directory not found.")
    except FileExistsError:
        print("Destination directory already exists.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def read_timings(tokens):
    timings = []
    for i in range(2, len(tokens)):
        start = 0
        end = -1
        if i == 2:
            start = 1
        if i == len(tokens) -1:
            end = -2
        timings.append(float(tokens[i][start : end]))
    assert len(timings) == 4
    return timings


def process_file(file):
    eps = None
    individual_percentage = None
    IO_percentage = None
    raven_no_diff_percentage = None
    raven_percentage = None
    times = None
    MILP_times = [0.0, 0.0] 
    for line in file:
        tokens = line.split(' ')
        if len(tokens) == 0:
            continue
        if tokens[0] == 'Eps':
            eps = float(tokens[-1])
        elif tokens[0] == 'Individual':
            if eps is None:
                raise ValueError(f'eps is None')
            individual_percentage = float(tokens[-1])
        elif tokens[0] == 'Baseline':
            if eps is None or individual_percentage is None:
                raise ValueError(f'eps or individual is None')
            IO_percentage = float(tokens[-1])   
        elif tokens[0] == 'Uap':
            if tokens[1] == 'no':
                if eps is None or individual_percentage is None or IO_percentage is None:
                    raise ValueError(f'eps or individual or I/O is None')
                raven_no_diff_percentage = float(tokens[-1])
            else:
                if eps is None or individual_percentage is None or IO_percentage is None \
                    or raven_no_diff_percentage is None:
                    raise ValueError(f'eps or individual or I/O or no diff is None')
                raven_percentage = float(tokens[-1])
                # print(raven_percentage)
                # Some cases like conv big diff percentages may be missing.
                # Overwrite with No diff formulation.
                if raven_percentage == 0.0 and raven_percentage < raven_no_diff_percentage:
                    raven_percentage = raven_no_diff_percentage
        elif tokens[0] == 'Avg.':
            times = read_timings(tokens)
        elif len(tokens) >= 2 and tokens[2] in ['optimization', 'constraint']:
            if tokens[0] == 'No':
                MILP_times[0] += float(tokens[-1])
            else:
                MILP_times[1] += float(tokens[-1])
    data_struct = DataStruct(eps = eps, individual_percentage = individual_percentage, IO_percentage = IO_percentage,
                            raven_no_diff_percentage = raven_no_diff_percentage, raven_percentage = raven_percentage,
                            times = times, MILP_times = MILP_times)
    return data_struct

def get_domain(domains, file_path):
    for domain in domains:
        if domain in file_path:
            return domain
    return None

def read_files(net_name, domains, thersholds, ans_dict):
    # Copy reults if empty.
    copy_files()
    # Define the pattern you want to search for
    pattern = f'{net_name}*.dat'  # For example, open all files with a .txt extension

    # Specify the directory where you want to search for files
    directory = './raw_results/'  # Replace with the actual directory path

    # Use the glob.glob() function to find files that match the pattern
    file_list = glob.glob(directory + '/' + pattern)
    #print(file_list)

    avg_timings = {}

    # Loop through the list of matching files and open them
    for file_path in file_list:
        with open(file_path, 'r') as file:
            data_struct = process_file(file)
            domain = get_domain(domains=domains, file_path=file_path)
            if domain not in avg_timings.keys():
                avg_timings[domain] = DataStructList(thersholds=thersholds)
            avg_timings[domain].append(data_struct)
        
    for key in avg_timings.keys():
        avg_times, avg_MILP_times = avg_timings[key].compute_avg_time(net_name=net_name)
        temp_name = file_path.split('/')[-1]
        temp_name = temp_name.split('.')[0]
        avg_timings[key].plot(net_name=temp_name, domain=key)
        print(f'{temp_name} {key} : {avg_times}')
        print(f'{temp_name} {key} MILP times : {avg_MILP_times}')


def main():
    net_names = {config.MNIST_CROWN_IBP: 'Crown IBP',
                 config.MNIST_CROWN_IBP_MED,
                 config.MNIST_CONV_SMALL_DIFFAI,
                 config.MNIST_CONV_BIG,
                 config.CIFAR_CROWN_IBP,
                 config.CIFAR_CONV_SMALL_DIFFAI,
                 config.CIFAR_CROWN_IBP_MEDIUM,
                 config.CIFAR_CONV_BIG,
                 'mnist_binary_sigmoid_pgd',
                 'mnist_binary_tanh_pgd',
                 'mnist_binary_relu_pgd',
                 'cls_tanh'}
    thersholds = {}
    thersholds[config.MNIST_CROWN_IBP] = 0, 0.1503
    thersholds[config.MNIST_CONV_SMALL_DIFFAI] = 0, 0.1503
    thersholds[config.MNIST_CROWN_IBP_MED] = 0, 0.203
    thersholds[config.MNIST_CONV_BIG] = 0, 0.251
    thersholds[config.CIFAR_CROWN_IBP] = 0, 100#0.1503
    thersholds[config.CIFAR_CONV_SMALL_DIFFAI] = 0, 100#0.1503
    thersholds[config.CIFAR_CROWN_IBP_MEDIUM] = 0, 3.01/255#0.203
    thersholds[config.CIFAR_CONV_BIG] = 0, 2.51/255#0.251
    thersholds['mnist_binary_sigmoid_pgd'] = 0, 100000
    thersholds['mnist_binary_tanh_pgd'] = 0, 100000
    thersholds['mnist_binary_relu_pgd'] = 0, 100000
    thersholds['cls_tanh'] = 0.0999, 100000
    domains = ['DEEPZ', 'DEEPPOLY']
    ans_dict = {}
    for net_name in net_names:
        read_files(net_name=net_name, domains=domains, thersholds=thersholds, ans_dict=ans_dict) 
    

    sns.set_style("darkgrid")
    sns.set_context('paper')
    # Plot the three line plots
    plt.figure(figsize=(6, 4.5)) 
    verified_amount = {
        'Individual': [29.25, 34.5, 26.75, 29.75, 28.5, 30.5, 31.5, 30.25, 34.0, 29.25],
        'I/O Formulation': [29.75, 34.75, 27.0, 30.0, 29.0, 31.0,31.75, 30.5, 34.0, 29.75],
        'RaVeN': [54.5, 63.0, 49.0, 54.0, 55.5, 58.5, 62.0, 56.0, 69.5, 58.0],
    }

    x = np.arange(10)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize = (12, 4))
    ax.grid(axis='x')
    for attribute, measurement in verified_amount.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color='cornflowerblue' if attribute == 'Individual' else 'indianred' if attribute == 'I/O Formulation' else 'darkseagreen')
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Worst-Case Targeted Accuracy (%)')
    ax.set_xlabel('Target Label')
    ax.set_xticks(x + width, x)
    ax.legend(loc='upper left', ncols=3)
    #ax.bar_label(False)
    #ax.set_ylim(0, 30)

    plt.savefig('plots/targeted.png', dpi=600,  bbox_inches='tight')


    
if __name__ == "__main__":
    main()
