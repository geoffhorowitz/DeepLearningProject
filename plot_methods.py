import numpy as np
import matplotlib.pyplot as plt

def plot_simple_learning_curve(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, exp=None):
    n_epochs = np.arange(1, len(train_loss_history)+1) #range(1, len(train_loss_history)+1)

    if exp:
        key, exp_val, i, j, k = exp
        title_suffix = ' for {}={}'.format(key, exp_val)
        save_path = 'experiments/runs/{}_{}_{}_learning_curves.png'.format(key, exp_val, k)
    else:
        title_suffix = ''
        save_path = './learning_curves.png'

    # Plot
    #fig, axs = plt.subplots(2, 1, sharex='all')
    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)

    # Loss plot
    ax = axs[0]
    ax.plot(n_epochs, train_loss_history, label='Training')
    ax.plot(n_epochs, valid_loss_history, label='Validation')
    ax.set_title("Figure {}.{}.{}: Loss Learning Curve".format(1*i+1, j, k)+title_suffix)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    #ax.grid(linestyle='dotted')
    #plt.savefig("Loss_curve.png")
    #ax.clear()

    # accuracy plot
    ax = axs[-1]
    ax.plot(n_epochs, train_acc_history, label='Training')
    ax.plot(n_epochs, valid_acc_history, label='Validation')
    ax.set_title("Figure {}.{}.{}: Perplexity Learning Curve".format(1*i+2, j, k)+title_suffix)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Perplexity")
    ax.legend(loc="lower right")
    ax.grid(linestyle='dotted')
    #plt.savefig("Perplexity_curve.png")
    #ax.clear()

    # output
    #plt.show()
    plt.savefig(save_path)
    plt.close(fig)


def plot_complex_learning_curve(results_dict, logx_scale=False):
    best_fh = open('experiments/best_results.txt'.format(), 'w')
    figure_count = -1
    for key, val in results_dict.items():
        if key == 'defaults':
            continue

        figure_count += 1

        for exp_count, (exp_val, all_runs) in enumerate(val.items()):
            best_perp = results_dict[key][exp_val]['best_perp']
            best_run = results_dict[key][exp_val]['best_run']

            best_res_text = '\nexperiment {}={}:'.format(key, exp_val) + \
                            'Run {} with train {}, validation {} perplexity'.format(best_run, np.round(results_dict[key][exp_val][best_run]['train_perp'][-1], 4), np.round(best_perp, 4))
            best_fh.write(best_res_text)

            train_loss_mean = results_dict[key][exp_val]['train_loss_mean']
            train_loss_std = results_dict[key][exp_val]['train_loss_std']
            train_acc_mean = results_dict[key][exp_val]['train_perp_mean']
            train_acc_std = results_dict[key][exp_val]['train_perp_std']
            val_loss_mean = results_dict[key][exp_val]['val_loss_mean']
            val_loss_std = results_dict[key][exp_val]['val_loss_std']
            val_acc_mean = results_dict[key][exp_val]['val_perp_mean']
            val_acc_std = results_dict[key][exp_val]['val_perp_std']

            # Plot learning curve
            fig, axs = plt.subplots(2, 1)
            fig.tight_layout(pad=3.5)
            alpha = .2
            train_color = 'r'
            test_color = 'g'
            n_epochs = results_dict['defaults']['epochs'] if key != 'epochs' else exp_val
            training_range = np.arange(1, n_epochs+1)
            save_path = 'experiments/{}_{}_learning_curves.png'.format(key, exp_val)
            title_suffix = ' for {}={}'.format(key, exp_val)


            # Loss plot
            ax = axs[0]
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
            #plt.savefig("Loss_curve.png")
            #ax.clear()

            # accuracy plot
            ax = axs[-1]
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
            ax.set_title("Figure {}.{}: Perplexity Learning Curve".format(2*figure_count+2, exp_count)+title_suffix)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Perplexity")
            ax.legend(loc="best")
            ax.grid(linestyle='dotted')
            #plt.savefig("Perplexity_curve.png")
            #ax.clear()


            fig.savefig(save_path)
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


        for exp_count, (exp_val, all_runs) in enumerate(val.items()):

            train_loss_mean[exp_count] = results_dict[key][exp_val]['train_loss_mean'][-1]
            train_loss_std[exp_count] = results_dict[key][exp_val]['train_loss_std'][-1]
            train_acc_mean[exp_count] = results_dict[key][exp_val]['train_perp_mean'][-1]
            train_acc_std[exp_count] = results_dict[key][exp_val]['train_perp_std'][-1]
            val_loss_mean[exp_count] = results_dict[key][exp_val]['val_loss_mean'][-1]
            val_loss_std[exp_count] = results_dict[key][exp_val]['val_loss_std'][-1]
            val_acc_mean[exp_count] = results_dict[key][exp_val]['val_perp_mean'][-1]
            val_acc_std[exp_count] = results_dict[key][exp_val]['val_perp_std'][-1]

            training_range[exp_count]  = exp_val

        # Plot learning curve
        fig, axs = plt.subplots(2, 1)
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
        ax = axs[0]
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
        ax = axs[-1]
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
        ax.set_title("Figure {}.b: Perplexity Complexity Curve".format(2*figure_count+2)+title_suffix)
        ax.set_xlabel(label)
        ax.set_ylabel("Perplexity")
        ax.legend(loc="best")
        ax.grid(linestyle='dotted')
        #plt.savefig("Perplexity_curve.png")
        #ax.clear()


        fig.savefig(save_path)
        plt.close(fig)
