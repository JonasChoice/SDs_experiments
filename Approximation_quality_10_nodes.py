import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import helper_functions as fcs


seed = 421


from codebase import metrics

random_state = np.random.RandomState(seed)

N= 10
p=0.3
max_order = 1
experiment_dict = {}
for i in range(50):

    graph1 = fcs.generate_ER_random_DAG(N, p, random_state=random_state)
    graph2 = fcs.generate_ER_random_DAG(N, p, random_state=random_state)

    full = metrics.metric_DAGs(graph1,graph2,max_order=N-2,randomize_higher_order =0, random_state=random_state)
    helper_dict = {}
    interval = [100,200,300,400,500]
    for test_number in interval:
        bootstrap_list = []
        for bootstrap_number in range(100):
            res = metrics.metric_DAGs(graph1,graph2, max_order=max_order, randomize_higher_order=test_number,
                                                     random_state=random_state)
            bootstrap_list.append(res)
            print(res)
            print('bootstrap ' + str((i,test_number,bootstrap_number)) + ' done')
        helper_dict[test_number] = np.std(bootstrap_list)
    experiment_dict[i] = helper_dict
    print('experiment %d done',i)

#print(experiment_list)

average_dict = {}
deviation_dict = {}
upper_errors_dict = {}
lower_errors_dict = {}
for test_number in interval:
    diff_A = []
    for j in experiment_dict.keys():
        diff_A.append(experiment_dict[j][test_number])
    average_dict[test_number] = np.mean(diff_A)
    deviation_dict[test_number] = np.std(diff_A)
    upper_errors_dict[test_number] = average_dict[test_number] + deviation_dict[test_number]
    lower_errors_dict[test_number] = average_dict[test_number] - deviation_dict[test_number]

mpl.style.use('seaborn-v0_8-colorblind')

fig, ax1 = plt.subplots(1,1,sharex=True, sharey=True)

ax1.grid(True)
plt.yticks(np.arange(0.0,0.025,0.005))
plt.xticks(np.arange(0,600,100))
#axs.xaxis.set_ticks(np.linspace(0,1100,12))
#axs.yaxis.set_ticks(np.linspace(-0.2,0.2,8))
ax1.plot(average_dict.keys(),average_dict.values())
ax1.plot(lower_errors_dict.keys(),lower_errors_dict.values(),color = 'lavender',alpha= 0.8)
ax1.plot(upper_errors_dict.keys(),upper_errors_dict.values(),color = 'lavender',alpha =0.8)
ax1.fill_between(upper_errors_dict.keys(),lower_errors_dict.values(),upper_errors_dict.values(),color = 'lavender',alpha =0.8)
#axs.plot(max_dict.keys(),max_dict.values(),label = 'max')
ax1.set_xlabel('L', fontsize = 12)
ax1.set_ylabel('difference with full s/c-metrix', fontsize = 12)
#ax1.legend(loc='upper right')

#plt.show()

filename = 'Plots/Approximation/'+ 'approx_quality_10_nodes_bootstrap' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()

#out_file = open('Results/Approximation/12_nodes'+ ".p", "wb")

#pickle.dump(experiment_dict, out_file)

#out_file.close()