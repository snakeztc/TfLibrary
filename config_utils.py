class RnnVAEConfig(object):
    # VAE
    latent = "gaussian"
    latent_size = 200
    latent_class = 5
    latent_category_var = 100
    full_kl_step = 1
    use_bow = False

    # Network general
    cell_type = "gru"
    grad_clip = 5.0
    init_w = 0.08
    batch_size = 40
    embed_size = 200
    cell_size = 400
    num_layer = 1

    # SGD
    op = "adam"
    init_lr = 0.001
    lr_hold = 1
    lr_decay = 0.6
    keep_prob = 1.0
    dec_keep_prob = 1.0
    improve_threshold = 0.996
    patient_increase = 2.0
    early_stop = True
    max_epoch = 50
    grad_noise = 0.0






