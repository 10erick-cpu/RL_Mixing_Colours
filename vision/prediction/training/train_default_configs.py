from utils.datasets.calcein_ds import CalceinCropsTrain, COMBINED_STATS_CALC_HOE_TRAIN, CalceinCropsVal, COMBINED_STATS_CALC_HOE_VAL, \
    CalceinCropsTest, COMBINED_STATS_CALC_HOE_TEST
from utils.datasets.hoechst_ds import HoechstCropsTrain, HoechstCropsVal, HoechstCropsTest
from vision.prediction.settings.model_configurations import default_configs


def setup_data_sets(train=True, val=True, test=False, mean_std_normalize=False):
    if train:
        train_calc = CalceinCropsTrain()
        train_hoe = HoechstCropsTrain()

        train_calc.update_transforms(COMBINED_STATS_CALC_HOE_TRAIN if mean_std_normalize else None)
        train_hoe.update_transforms(COMBINED_STATS_CALC_HOE_TRAIN if mean_std_normalize else None)
        train_calc.init()
        train_hoe.init()
        training_set = train_calc + train_hoe
    else:
        training_set = None

    if val:
        val_calc = CalceinCropsVal()
        val_hoe = HoechstCropsVal()

        val_calc.update_transforms(COMBINED_STATS_CALC_HOE_VAL if mean_std_normalize else None)
        val_hoe.update_transforms(COMBINED_STATS_CALC_HOE_VAL if mean_std_normalize else None)

        val_calc.init()
        val_hoe.init()
        val_set = val_calc + val_hoe
    else:
        val_set = None

    if test:
        test_calc = CalceinCropsTest()
        test_hoe = HoechstCropsTest()

        test_calc.update_transforms(COMBINED_STATS_CALC_HOE_TEST if mean_std_normalize else None)
        test_hoe.update_transforms(COMBINED_STATS_CALC_HOE_TEST if mean_std_normalize else None)

        test_calc.init()
        test_hoe.init()

        test_set = test_calc + test_hoe
    else:
        test_set = None

    return training_set, val_set, test_set


configs = default_configs()

if __name__ == '__main__':

    setup_data_sets(True, True, True)
    raise EOFError

    num_workers = multiprocessing.cpu_count()

    print(f"Available CPUs: {num_workers}")

    configs = configs[-2:]

    for config_idx, config in enumerate(configs):
        print(f"Train configuration {config_idx + 1}/{len(configs)}")

        # config.training = DotDict({'batch_size': 11, 'num_epochs': 50, 'steps_per_epoch': 500})
        # config.validation = DotDict({'batch_size': 12, 'steps_per_epoch': 20, 'metrics': ['l1', 'ssim', 'l2']})

        if not config.already_trained():
            print("Training", config.get_tensor_board_id())
            set_seeds(1)

            ds_train, ds_val, ds_test = setup_data_sets(train=True, val=True, test=False)
            train(config,
                  ds_train=ds_train,
                  ds_test=ds_val,
                  preprocessor=None,
                  postprocessor=None,
                  save_interval=10,
                  num_workers=num_workers)



        else:
            print("already trained:", config.get_tensor_board_id())
        print("Validation", config.get_tensor_board_id())
        try:
            eval_steps_per_epoch = -1
            Validator.evaluate(config, CalceinCropsTrain(auto_init=True),
                               dataset_identifier="CalceinCropsTrain",
                               steps_per_epoch=eval_steps_per_epoch, batch_size=config.validation.batch_size)
            Validator.evaluate(config, HoechstCropsTrain(auto_init=True),
                               dataset_identifier="HoechstCropsTrain",
                               steps_per_epoch=eval_steps_per_epoch, batch_size=config.validation.batch_size)
            Validator.evaluate(config, CalceinCropsVal(auto_init=True),
                               dataset_identifier="CalceinCropsVal",
                               steps_per_epoch=eval_steps_per_epoch, batch_size=config.validation.batch_size)
            Validator.evaluate(config, HoechstCropsVal(auto_init=True),
                               dataset_identifier="HoechstCropsVal",
                               steps_per_epoch=eval_steps_per_epoch, batch_size=config.validation.batch_size)
            Validator.evaluate(config, HoechstCropsTest(auto_init=True),
                               dataset_identifier="HoechstCropsTest",
                               steps_per_epoch=eval_steps_per_epoch, batch_size=config.validation.batch_size)
            Validator.evaluate(config, CalceinCropsTest(auto_init=True),
                               dataset_identifier="CalceinCropsTest",
                               steps_per_epoch=eval_steps_per_epoch, batch_size=config.validation.batch_size)
        except ValueError as e:
            print("Skipping config with non-existing checkpoint", config.get_tensor_board_id())
            continue
