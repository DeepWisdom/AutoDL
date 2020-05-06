sample_strategy = {

    "sample_iter_incremental_with_train_split": {'add_val_to_train': False,
                                                 'update_train': True,
                                                 'use_full': False
                                                 },

    "sample_iter_incremental_no_train_split": {'add_val_to_train': True,
                                               'update_train': True,
                                               'use_full': False},


    "sample_from_full_data": {'add_val_to_train': False,
                              'update_train': False,
                              'use_full': True},


    "sample_from_full_train_data": {'add_val_to_train': False,
                                    'update_train': False,
                                    'use_full': False}
}
