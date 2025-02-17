## 数据生成

我们准备了若干个单臂、双臂的操作任务，分别位于`discoverse/examples/tasks_airbot_play`和`discoverse/examples/tasks_mmk2`中，要自动收集数据，请运行

```python
cd scripts
python os_run.py --robot_name <ROBOT_NAME> --task_name <TASK_NAME> --track_num <NUM_TRACK> --nw <NUM_OF_WORKERS>
e.g. python os_run.py --robot_name airbot_play --task_name kiwi_place --track_num 100 --nw 8
# 表示使用airbot_play机械臂，任务为放置猕猴桃，总共生成100条任务轨迹，使用8个进程来同时生成数据。
```

## 数据转换

## act

将仿真采集的原始数据格式转换为ACT算法中用到的hdf5格式，命令如下：

```bash
python3 policies/act/data_process/raw_to_hdf5.py -md mujoco -dir data -tn <task_name> -vn <video_names>
```

- `-md`: 转换模式，mujoco表示转换由discoverse仿真器采集的数据
- `-dir`: 数据存放的根目录，默认为data
- `-tn`: 任务名，程序将根据任务名从data目录中寻找相同名称的数据集文件夹
- `-vn`: 视频名，指定需要转换的视频文件名（无后缀），多个名称用空格隔开

转换后的数据存放于`discoverse/data/hdf5`文件夹中。

## dp

将仿真采集的原始数据格式转换为DP算法中用到的zarr格式，命令如下：

```bash
python3 policies/dp.py  -dir data -tn <task_name> 
```

- `-dir`: 数据存放的根目录，默认为data
- `-tn`: 任务名，程序将根据任务名从data目录中寻找相同名称的数据集文件夹

转换后的数据存放于`discoverse/data/zarr`文件夹中。