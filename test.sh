
# evaling without optimization
python main.py --test 1 --batch_size 128 --model model_Gaussian --reload_model --layers 3 --gpu 0 --model_path "./pre_trained_models/Model_Gaussian_p1_4941.pth"

# evaling with optimization
python main.py --test 1 --test_time_optimization --opt_iter_num 4 --batch_size 128 --model model_Gaussian --reload_model --layers 3 --pad 0 --gpu 1 --model_path "./pre_trained_models/Model_Gaussian_p1_4941.pth"