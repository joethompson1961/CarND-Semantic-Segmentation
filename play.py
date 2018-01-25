import os.path
import time

def make_hparam_string(kprob, lrate, l2_const):
    return "kp_%.0E,lr_%.0E,l2_%.0E" % (kprob, lrate, l2_const)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    # Larger batch sizes might speed up the training but can degrade the quality of the model at the same time.
    # Good results obtained using small batch sizes of 2 or 4.
    # It important to note that batch size and learning rate are linked. If the batch size is too small then
    # the gradients will become more unstable and learning rate would need to be reduced (~1e-4 or 1e-5).
    epochs = 50
    batch_size = 4

    # hyperparameter search
    for reg in [0.002,0.005]:
        for lr in [0.0001,0.00005]:
            for kp in [0.8,0.9]:
                param_str = "kp-%.2f_lr-%.5f_reg-%.3f" % (kp, lr, reg)

                tmp =  param_str + "_" + str(time.time())
                output_dir = os.path.join(runs_dir, tmp)
                
                print(output_dir)

if __name__ == '__main__':
    run()
