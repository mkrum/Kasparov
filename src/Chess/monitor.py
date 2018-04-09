
import os
import os.path
import time


def get_experiments():
    return [ 'res/' + x for x in os.listdir('res') ]

def check_finished(path):
    return os.path.isfile(path+'/final')

def read_iteration(path):
    f = open(path, 'r')
    try:
        gamma = float(f.readline().split()[-1])
        loss = float(f.readline().split()[-1])
        time = float(f.readline().split()[-1])

        return gamma, loss, time 

    except IndexError:
        return None

def read_settings(path):
    f = open(path, 'r')
    attrs = {}
    for line in f.read().splitlines():
        spl = line.split()
        if spl[0] not in ['path', 'quiet']:
            attrs[spl[0]] = spl[1]
    
    return attrs

def get_current_data(path):
    
    iterations = []
    for f in os.listdir(path):
        if f.isdigit():
            iterations.append(int(f))

    iterations.sort() 
    vals = []
    for it in iterations:
        val = read_iteration(path + '/' + str(it))
        if val:
            vals.append(val)
        else:
            print('Currently Testing...')

        
    if len(vals) >= 1:
        best_loss = min( [x[1] for x in vals] )

        current_iteration = iterations[-1]
        c_gamma, c_loss, c_time = vals[-1]
        return current_iteration, best_loss, c_gamma, c_loss, c_time
    else:
        return 0, None, None, None, None



if __name__ == '__main__':
    while True:
        os.system('clear')
        print(time.ctime())
        print()
        for x in get_experiments():
            if not check_finished(x):
                print(x.split('/')[-1])
                settings = read_settings(x + '/settings.txt')
                
                info = ['model', 'epoch', 'gamma', 'decay', 'history', 'threads']
                for val in info:
                    print('{0:>10}'.format(val), end='')

                print() 
                print('----------' * len(info))

                for val in ['model', 'epoch', 'gamma', 'decay', 'history', 'threads']:
                    print('{0:>10}'.format(settings[val]), end='')

                print('\n\nProgress')
                
                current_iteration, best_loss, curr_gamma, curr_loss, curr_time = get_current_data(x)

                print('{}/{}\n'.format(current_iteration, settings['iter']))
                
                if current_iteration > 0:
                    for val in ['Loss', 'Best Loss', 'Gamma', 'Time']:
                        print('{0:>10}'.format(val), end='')

                    print() 
                    print('----------' * 4)

                    for val in [curr_loss, best_loss, curr_gamma, curr_time]:
                        print('{0:>10}'.format(round(val, 4)), end='')
                    print()
                print()
                              
        time.sleep(1)
