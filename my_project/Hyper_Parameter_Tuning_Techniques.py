# --coding:utf-8--
from nni.experiment import Experiment
import nni

search_space = {
    'lr': {'_type': 'loguniform', '_value': [0.001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 0.2]},
    'features': {'_type': 'choice', '_value': [128, 256]},
}

optimized_params = nni.get_next_parameter()

search_space.update(optimized_params)
print(search_space)

experiment = Experiment('local')

experiment.config.trial_command = 'python train2_file_string.py'  # 运行命令
experiment.config.trial_code_directory = '.'  # py 文件所在文件夹

experiment.config.search_space = search_space  # 搜索空间

experiment.config.tuner.name = 'TPE'  # 配置调优算法
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 10  # 设置最大实验次数，即总共要进行多少次不同的实验
experiment.config.trial_concurrency = 2  # 设置单次同时实验的个数

experiment.run(8081)  # 启动实验
# experiment.run(8080)

# input('Press enter to quit')
experiment.stop()  # 试验结束

# 重启网页
nni.experiment.Experiment.view()
