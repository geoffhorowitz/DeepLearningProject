import numpy as np
import matplotlib.pyplot as plt

def plot_complex_learning_curve(results_dict, logx_scale=False):
    best_fh = open('experiments/best_results.txt'.format(), 'w')
    figure_count = -1
    for key, val in results_dict.items():
        if key == 'defaults':
            continue

        figure_count += 1

        for exp_count, (exp_val, all_runs) in enumerate(val.items()):
            best_median = results_dict[key][exp_val]['best_median']
            best_run = results_dict[key][exp_val]['best_run']

            best_res_text = '\nexperiment {}={}:'.format(key, exp_val) + \
                            'Run {} with train {}, validation {} median'.format(best_run, np.round(results_dict[key][exp_val][best_run]['train_median'][-1], 4), np.round(best_median, 4))
            best_fh.write(best_res_text)

            train_loss_mean = results_dict[key][exp_val]['train_loss_mean']
            train_loss_std = results_dict[key][exp_val]['train_loss_std']
            train_acc_mean = results_dict[key][exp_val]['train_median_mean']
            train_acc_std = results_dict[key][exp_val]['train_median_std']
            val_loss_mean = results_dict[key][exp_val]['val_loss_mean']
            val_loss_std = results_dict[key][exp_val]['val_loss_std']
            val_acc_mean = results_dict[key][exp_val]['val_median_mean']
            val_acc_std = results_dict[key][exp_val]['val_median_std']
            train_accim_mean = results_dict[key][exp_val]['train_imacc_mean']
            train_accim_std = results_dict[key][exp_val]['train_imacc_std']
            val_accim_mean = results_dict[key][exp_val]['val_imacc_mean']
            val_accim_std = results_dict[key][exp_val]['val_imacc_std']
            train_accrec_mean = results_dict[key][exp_val]['train_recacc_mean']
            train_accrec_std = results_dict[key][exp_val]['train_recacc_std']
            val_accrec_mean = results_dict[key][exp_val]['val_recacc_mean']
            val_accrec_std = results_dict[key][exp_val]['val_recacc_std']
            
            # Plot learning curve
            fig, ax = plt.subplots(1, 1)
            fig.tight_layout(pad=3.5)
            alpha = .2
            train_color = 'r'
            test_color = 'g'
            n_epochs = results_dict['defaults']['epochs'] if key != 'epochs' else exp_val
            training_range = np.arange(1, n_epochs+1)
            save_path = 'experiments/{}_{}_learning_curves.png'.format(key, exp_val)
            title_suffix = ' for {}={}'.format(key, exp_val)


            # Loss plot
            #ax = axs[0]
            ax.fill_between(training_range, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                            alpha=alpha, color=train_color)
            ax.fill_between(training_range, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std,
                            alpha=alpha, color=test_color)
            if logx_scale:
                ax.semilogx(training_range, train_loss_mean, 'o-', color=train_color, label='Training')
                ax.semilogx(training_range, val_loss_mean, 'o-', color=test_color, label='Validation')
            else:
                ax.plot(training_range, train_loss_mean, 'o-', color=train_color, label='Training')
                ax.plot(training_range, val_loss_mean, 'o-', color=test_color, label='Validation')
            ax.set_title("Figure {}.{}: Loss Learning Curve".format(2*figure_count+1, exp_count)+title_suffix)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend(loc="best")
            ax.grid(linestyle='dotted')
            plt.savefig("Loss_curve.png")
            ax.clear()

            # accuracy plot
            #ax = axs[-1]
            ax.fill_between(training_range, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std,
                            alpha=alpha, color=train_color)
            ax.fill_between(training_range, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                            alpha=alpha, color=test_color)
            if logx_scale:
                ax.semilogx(training_range, train_acc_mean, 'o-', color=train_color, label='Training')
                ax.semilogx(training_range, val_acc_mean, 'o-', color=test_color, label='Validation')
            else:
                ax.plot(training_range, train_acc_mean, 'o-', color=train_color, label='Training')
                ax.plot(training_range, val_acc_mean, 'o-', color=test_color, label='Validation')
            ax.set_title("Figure {}.{}: Median Rank Learning Curve".format(2*figure_count+2, exp_count)+title_suffix)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Median Rank")
            ax.legend(loc="best")
            ax.grid(linestyle='dotted')
            plt.savefig(save_path[:-4]+'_median'+save_path[-4:])
            ax.clear()
            
            # accuracy plot - img
            #ax = axs[-1]
            ax.fill_between(training_range, train_accim_mean - train_accim_std, train_accim_mean + train_accim_std,
                            alpha=alpha, color=train_color)
            ax.fill_between(training_range, val_accim_mean - val_accim_std, val_accim_mean + val_accim_std,
                            alpha=alpha, color=test_color)
            if logx_scale:
                ax.semilogx(training_range, train_accim_mean, 'o-', color=train_color, label='Image Training')
                ax.semilogx(training_range, val_accim_mean, 'o-', color=test_color, label='Image Validation')
            else:
                ax.plot(training_range, train_accim_mean, 'o-', color=train_color, label='Image Training')
                ax.plot(training_range, val_accim_mean, 'o-', color=test_color, label='Image Validation')
            ax.set_title("Figure {}.{}: Image Accuracy Learning Curve".format(2*figure_count+2, exp_count)+title_suffix)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            ax.legend(loc="best")
            ax.grid(linestyle='dotted')
            plt.savefig(save_path[:-4]+'_imacc'+save_path[-4:])
            ax.clear()
            
            # accuracy plot - recipe
            #ax = axs[-1]
            ax.fill_between(training_range, train_accrec_mean - train_accrec_std, train_accrec_mean + train_accrec_std,
                            alpha=alpha, color=train_color)
            ax.fill_between(training_range, val_accrec_mean - val_accrec_std, val_accrec_mean + val_accrec_std,
                            alpha=alpha, color=test_color)
            if logx_scale:
                ax.semilogx(training_range, train_accrec_mean, 'o-', color=train_color, label='Recipe Training')
                ax.semilogx(training_range, val_accrec_mean, 'o-', color=test_color, label='Recipe Validation')
            else:
                ax.plot(training_range, train_accrec_mean, 'o-', color=train_color, label='Recipe Training')
                ax.plot(training_range, val_accrec_mean, 'o-', color=test_color, label='Recipe Validation')
            ax.set_title("Figure {}.{}: Recipe Accuracy Learning Curve".format(2*figure_count+2, exp_count)+title_suffix)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            ax.legend(loc="best")
            ax.grid(linestyle='dotted')
            plt.savefig(save_path[:-4]+'_recacc'+save_path[-4:])
            ax.clear()


            #fig.savefig(save_path)
            plt.close(fig)

    best_fh.close()


def plot_complexity_curve(results_dict, logx_scale=False):
    figure_count = 2
    for key, val in results_dict.items():
        if key == 'defaults':
            continue

        figure_count += 1
        training_range = np.zeros(len(val.keys()))
        train_loss_mean = np.zeros(len(val.keys()))
        train_loss_std = np.zeros(len(val.keys()))
        train_acc_mean = np.zeros(len(val.keys()))
        train_acc_std = np.zeros(len(val.keys()))
        val_loss_mean = np.zeros(len(val.keys()))
        val_loss_std = np.zeros(len(val.keys()))
        val_acc_mean = np.zeros(len(val.keys()))
        val_acc_std = np.zeros(len(val.keys()))
        train_accim_mean = np.zeros(len(val.keys()))
        train_accim_std = np.zeros(len(val.keys()))
        val_accim_mean = np.zeros(len(val.keys()))
        val_accim_std = np.zeros(len(val.keys()))
        train_accrec_mean = np.zeros(len(val.keys()))
        train_accrec_std = np.zeros(len(val.keys()))
        val_accrec_mean = np.zeros(len(val.keys()))
        val_accrec_std = np.zeros(len(val.keys()))


        for exp_count, (exp_val, all_runs) in enumerate(val.items()):

            train_loss_mean[exp_count] = results_dict[key][exp_val]['train_loss_mean'][-1]
            train_loss_std[exp_count] = results_dict[key][exp_val]['train_loss_std'][-1]
            train_acc_mean[exp_count] = results_dict[key][exp_val]['train_median_mean'][-1]
            train_acc_std[exp_count] = results_dict[key][exp_val]['train_median_std'][-1]
            val_loss_mean[exp_count] = results_dict[key][exp_val]['val_loss_mean'][-1]
            val_loss_std[exp_count] = results_dict[key][exp_val]['val_loss_std'][-1]
            val_acc_mean[exp_count] = results_dict[key][exp_val]['val_median_mean'][-1]
            val_acc_std[exp_count] = results_dict[key][exp_val]['val_median_std'][-1]
            train_accim_mean[exp_count] = results_dict[key][exp_val]['train_imacc_mean'][-1]
            train_accim_std[exp_count] = results_dict[key][exp_val]['train_imacc_std'][-1]
            val_accim_mean[exp_count] = results_dict[key][exp_val]['val_imacc_mean'][-1]
            val_accim_std[exp_count] = results_dict[key][exp_val]['val_imacc_std'][-1]
            train_accrec_mean[exp_count] = results_dict[key][exp_val]['train_recacc_mean'][-1]
            train_accrec_std[exp_count] = results_dict[key][exp_val]['train_recacc_std'][-1]
            val_accrec_mean[exp_count] = results_dict[key][exp_val]['val_recacc_mean'][-1]
            val_accrec_std[exp_count] = results_dict[key][exp_val]['val_recacc_std'][-1]

            training_range[exp_count]  = exp_val

        # Plot learning curve
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout(pad=3.5)
        alpha = .2
        train_color = 'r'
        test_color = 'g'

        save_path = 'experiments/{}_complexity_curve.png'.format(key)
        title_suffix = ' for {}'.format(key)

        if key == 'learning_rate':
            label='Learning Rate'
        else:
            label = key

        # Loss plot
        #ax = axs[0]
        ax.fill_between(training_range, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                        alpha=alpha, color=train_color)
        ax.fill_between(training_range, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std,
                        alpha=alpha, color=test_color)
        if logx_scale:
            ax.semilogx(training_range, train_loss_mean, 'o-', color=train_color, label='Training')
            ax.semilogx(training_range, val_loss_mean, 'o-', color=test_color, label='Validation')
        else:
            ax.plot(training_range, train_loss_mean, 'o-', color=train_color, label='Training')
            ax.plot(training_range, val_loss_mean, 'o-', color=test_color, label='Validation')
        ax.set_title("Figure {}.a: Loss Complexity Curve".format(2*figure_count+1)+title_suffix)
        ax.set_xlabel(label)
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        ax.grid(linestyle='dotted')
        #plt.savefig("Loss_curve.png")
        #ax.clear()

        # accuracy plot
        #ax = axs[-1]
        ax.fill_between(training_range, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std,
                        alpha=alpha, color=train_color)
        ax.fill_between(training_range, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                        alpha=alpha, color=test_color)
        if logx_scale:
            ax.semilogx(training_range, train_acc_mean, 'o-', color=train_color, label='Training')
            ax.semilogx(training_range, val_acc_mean, 'o-', color=test_color, label='Validation')
        else:
            ax.plot(training_range, train_acc_mean, 'o-', color=train_color, label='Training')
            ax.plot(training_range, val_acc_mean, 'o-', color=test_color, label='Validation')
        ax.set_title("Figure {}.b: Median Rank Complexity Curve".format(2*figure_count+2)+title_suffix)
        ax.set_xlabel(label)
        ax.set_ylabel("Median Rank")
        ax.legend(loc="best")
        ax.grid(linestyle='dotted')
        #plt.savefig("Perplexity_curve.png")
        #ax.clear()
        
        # accuracy plot - img
        #ax = axs[-1]
        ax.fill_between(training_range, train_accim_mean - train_accim_std, train_accim_mean + train_accim_std,
                        alpha=alpha, color=train_color)
        ax.fill_between(training_range, val_accim_mean - val_accim_std, val_accim_mean + val_accim_std,
                        alpha=alpha, color=test_color)
        if logx_scale:
            ax.semilogx(training_range, train_accim_mean, 'o-', color=train_color, label='Image Training')
            ax.semilogx(training_range, val_accim_mean, 'o-', color=test_color, label='Image Validation')
        else:
            ax.plot(training_range, train_accim_mean, 'o-', color=train_color, label='Image Training')
            ax.plot(training_range, val_accim_mean, 'o-', color=test_color, label='Image Validation')
        ax.set_title("Figure {}.{}: Image Accuracy Complexity Curve".format(2*figure_count+2, exp_count)+title_suffix)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="best")
        ax.grid(linestyle='dotted')
        plt.savefig(save_path[:-4]+'_imacc'+save_path[-4:])
        ax.clear()

        # accuracy plot - recipe
        #ax = axs[-1]
        ax.fill_between(training_range, train_accrec_mean - train_accrec_std, train_accrec_mean + train_accrec_std,
                        alpha=alpha, color=train_color)
        ax.fill_between(training_range, val_accrec_mean - val_accrec_std, val_accrec_mean + val_accrec_std,
                        alpha=alpha, color=test_color)
        if logx_scale:
            ax.semilogx(training_range, train_accrec_mean, 'o-', color=train_color, label='Recipe Training')
            ax.semilogx(training_range, val_accrec_mean, 'o-', color=test_color, label='Recipe Validation')
        else:
            ax.plot(training_range, train_accrec_mean, 'o-', color=train_color, label='Recipe Training')
            ax.plot(training_range, val_accrec_mean, 'o-', color=test_color, label='Recipe Validation')
        ax.set_title("Figure {}.{}: Recipe Accuracy Complexity Curve".format(2*figure_count+2, exp_count)+title_suffix)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="best")
        ax.grid(linestyle='dotted')
        plt.savefig(save_path[:-4]+'_recacc'+save_path[-4:])
        ax.clear()


        fig.savefig(save_path)
        plt.close(fig)
