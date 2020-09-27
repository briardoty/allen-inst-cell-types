import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

acc_df = pd.read_csv("max_acc_df.csv")
with open("case_dict.json", "r") as json_file:
    case_dict = json.load(json_file)

def plot_by_parameter_diff_within(net, acc_df, vscale_min=-0.05, vscale_max=0.05,
    metric='rel_acc',colormap='bwr'): 


    net_df = acc_df.query('dataset == "cifar10" & train_scheme == "adam" & net_name == @net & (not cross_fam) & (is_mixed)').copy() 
    net_df['rel_acc'] = net_df['max_val_acc'] - net_df['max_pred']
    matrix={'tanh':np.zeros((6,6)),'swish':np.zeros((7,7))}
    matrix['tanh'][:] = 0
    matrix['swish'][:] = 0
    tanh_indexes={'2':0,'1':1,'0.5':2,'0.1':3,'0.05':4,'0.01':5}
    swish_indexes={'10':0,'7.5':1,'5':2,'2':3,'1':4,'0.5':5,'0.1':6}
    cases = net_df['case'].unique()
    
    for case in cases:
        if case.startswith('tanh'):
            i,j =case_dict[case]['act_fn_params']
            matrix['tanh'][tanh_indexes[i],tanh_indexes[j]] = net_df.query('case == @case')[metric].mean()
            matrix['tanh'][tanh_indexes[j],tanh_indexes[i]] = net_df.query('case == @case')[metric].mean() 
        else:
            i,j =case_dict[case]['act_fn_params']
            matrix['swish'][swish_indexes[i],swish_indexes[j]] = net_df.query('case == @case')[metric].mean()   
            matrix['swish'][swish_indexes[j],swish_indexes[i]] = net_df.query('case == @case')[metric].mean()    
    
    plt.figure()
    im = plt.imshow(matrix['tanh'],cmap=colormap,vmin=vscale_min,vmax=vscale_max)
    cbar = plt.gcf().colorbar(im,ax=plt.gca())
    if metric == 'rel_acc':
        cbar.ax.set_ylabel('Relative Accuracy (%)')
    else:
        cbar.ax.set_ylabel(metric+' (%)')
    plt.ylabel(r'Tanh $\beta$',fontsize=12)
    plt.xlabel(r'Tanh $\beta$',fontsize=12)
    tlabels=tanh_indexes.keys()
    tticks = [tanh_indexes[x] for x in tlabels]
    plt.xticks(tticks,labels=tlabels,fontsize=12)
    plt.yticks(tticks,labels=tlabels,fontsize=12)
    plt.title('Within Family - '+net)
    plt.savefig('within_tanh_'+net+'_'+metric+'.png')

    plt.figure()
    im = plt.imshow(matrix['swish'],cmap=colormap,vmin=vscale_min,vmax=vscale_max)
    cbar = plt.gcf().colorbar(im,ax=plt.gca())
    if metric == 'rel_acc':
        cbar.ax.set_ylabel('Relative Accuracy (%)')
    else:
        cbar.ax.set_ylabel(metric+' (%)')
    plt.ylabel(r'Swish $\beta$',fontsize=12)
    plt.xlabel(r'Swish $\beta$',fontsize=12)
    slabels=swish_indexes.keys()
    sticks = [swish_indexes[x] for x in slabels]
    plt.xticks(sticks,labels=slabels,fontsize=12)
    plt.yticks(sticks,labels=slabels,fontsize=12)
    plt.title('Within Family - '+net)
    plt.savefig('within_swish_'+net+'_'+metric+'.png')


def plot_by_parameter_diff_across(net,acc_df,vscale_min=-0.05,vscale_max=0.05,metric='rel_acc',colormap='bwr'): 
    net_df = acc_df.query('dataset == "cifar10" & train_scheme == "adam" & net_name == @net & cross_fam & (is_mixed)').copy() 
    net_df['rel_acc'] = net_df['max_val_acc'] - net_df['max_pred']

    tanh_indexes={'2':0,'1':1,'0.5':2,'0.1':3,'0.05':4,'0.01':5}
    swish_indexes={'10':0,'7.5':1,'5':2,'2':3,'1':4,'0.5':5,'0.1':6}
    matrix=np.zeros((len(tanh_indexes.keys()),len(swish_indexes.keys())))
    matrix[:] = 0
    cases = net_df['case'].unique()
    for case in cases:
        swish,tanh = case.split('-') 
        j =case_dict[swish]['act_fn_params'][0]
        i =case_dict[tanh]['act_fn_params'][0]
        matrix[tanh_indexes[i],swish_indexes[j]] = net_df.query('case == @case')[metric].mean()
        

    plt.figure()
    im = plt.imshow(matrix,cmap=colormap,vmin=vscale_min,vmax=vscale_max)
    cbar = plt.gcf().colorbar(im,ax=plt.gca())
    cbar.ax.set_ylabel('Relative Accuracy (%)')
    if metric == 'rel_acc':
        cbar.ax.set_ylabel('Relative Accuracy (%)')
    else:
        cbar.ax.set_ylabel(metric+' (%)')
    plt.ylabel(r'Tanh $\beta$',fontsize=12)
    plt.xlabel(r'Swish $\beta$',fontsize=12)
    tlabels=tanh_indexes.keys()
    tticks = [tanh_indexes[x] for x in tlabels]
    slabels=swish_indexes.keys()
    sticks = [swish_indexes[x] for x in slabels]
    plt.xticks(sticks,labels=slabels,fontsize=12)
    plt.yticks(tticks,labels=tlabels,fontsize=12)
    plt.title('Across Family - '+net)
    plt.savefig('across_'+net+'_'+metric+'.png')

plt.close('all')
plot_by_parameter_diff_within('vgg11',acc_df)
plot_by_parameter_diff_within('sticknet8',acc_df)
plot_by_parameter_diff_across('vgg11',acc_df)
plot_by_parameter_diff_across('sticknet8',acc_df)
plot_by_parameter_diff_within('vgg11',acc_df,metric='max_val_acc',vscale_min=0.83,vscale_max=.9,colormap='Reds')
plot_by_parameter_diff_within('sticknet8',acc_df,metric='max_val_acc',vscale_min=0.62,vscale_max=.71,colormap='Reds')
plot_by_parameter_diff_across('vgg11',acc_df,metric='max_val_acc',vscale_min=0.83,vscale_max=.9,colormap='Reds')
plot_by_parameter_diff_across('sticknet8',acc_df,metric='max_val_acc',vscale_min=0.62,vscale_max=.71,colormap='Reds')

