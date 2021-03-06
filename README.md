# 需要修正
state abstraction
[减少item | 将frog设置为墙壁]

featurize 
[step 要加入进去]

progressive neural network
[加入lateral connection 以及可学习的参数alpha]

hierarchies with hindsight
[目标设定]

_djikstra


# 注意
不要使用中文输出，在服务器上跑会出错

不要更改 stable_baselines 和 pommerman 中的文件

# 安装环境依赖包
可以 conda 初始化一个纯净环境，使用清华源或者豆瓣源安装

```pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt```

# 后台运行
```nohup python -u *.py > logs/filename 2>&1 &```

# 预训练
pretrain v0
```python pretrain.py --env=PommeRadioCompetition-v2 --num_timesteps=1000 --policy_type=resnet --expert_path=dataset/hako_v0/agent0 --save_path=models/hako_v0```

pretrain v1
```python pretrain.py --env=PommeRadioCompetition-v2  --num_timesteps=1000 --policy_type=resnet --expert_path=dataset/hako_v1/ --save_path=models/hako_v1 --load_path=models/hako_v0_e29.zip --pretrain_version=v1```

# 训练
参数解析可以参考 my_common.cmd_utils # learn

```python run.py --env=PommeRadioCompetition-v2 --policy_type=pgn --model_type=ppo --num_timesteps=1e7 --log_path=./logs/ --save_path=./models/test/ --save_interval=1e2```

# 演示 或 评测
参数解析可以参考 my_common.cmd_utils # play and evaluate

model 0 --> 4 | using_prune


```python play.py --env=PommeRadioCompetition-v2 --model_type=ppo --using_prune  --model0_path=./models/hako_v0_e29.zip```

```python evaluate.py --env=PommeRadioCompetition-v2 --using_prune --model0=hako_v0_e29.zip+prune --model1=hit18Agent+prune --model2=hako_v0_e29.zip+prune --model3=hit18Agent+prune --model0_path=models/hako_v0_e29.zip --model2_path=models/hako_v0_e29.zip```

# 环境信息
较有可能用到的
活着的智能体编号
'alive': [10, 11, 12]

棋盘编号
'board':
array([[ 0, 1, 2, 1, 2, 1, 5, 5, 5, 5, 5],
[ 1, 0, 3, 0, 2, 2, 5, 5, 5, 5, 5],
[ 0, 0, 0, 1, 1, 2, 5, 5, 5, 5, 5],
[ 1, 10, 1, 0, 0, 0, 5, 5, 5, 5, 5],
[ 2, 2, 1, 0, 0, 2, 5, 5, 5, 5, 5],
[ 1, 2, 2, 0, 2, 0, 5, 5, 5, 5, 5],
[ 1, 2, 0, 1, 0, 2, 5, 5, 5, 5, 5],
[ 1, 0, 1, 1, 1, 2, 5, 5, 5, 5, 5],
[ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]], dtype=uint8),

炸弹范围（视野外视为无炸弹，即为0）
'bomb_blast_strength':
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),

炸弹生命值（视野外视为无炸弹，即为0）
'bomb_life':
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),

炸弹运动方向（视野外视为无炸弹，即为0）
'bomb_moving_direction':
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),

爆炸火花生命值（视野外视为无炸弹，即为0）
'flame_life':
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),

本智能体位置
'position': (3, 1),

本智能体炸弹范围
'blast_strength': 3,

本智能体是否可踢
'can_kick': False,

本智能体剩余炸弹量
'ammo': 0,

队友传来的信息（队友死或刚开始为0，否则范围是[1, 8]）
'message': (0, 0)}

不太会用到的
走了几步
'step_count': 20,

对局类型
'game_type': 3,

对局环境
'game_env': 'pommerman.envs.v2:Pomme',

队友
'teammate': <Item.Agent2: 12>,

敌人
'enemies': [<Item.Agent1: 11>, <Item.Agent3: 13>, <Item.AgentDummy: 9>],