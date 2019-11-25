# 注意
不要使用中文输出，在服务器上跑会出错

不要更改 stable_baselines 和 pommerman 中的文件

# 安装环境依赖包
可以 conda 初始化一个纯净环境，使用清华源或者豆瓣源安装

```pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt```

# 后台运行
```nohup python *.py > logs/filename 2&>1 &```

# 预训练
```python pretrain.py --env=PommeRadioCompetition-v2 --num_timesteps=1000 --policy_type=resnet --expert_path=dataset/hako_v0/agent0 --save_path=models/```

# 训练
参数解析可以参考 my_common.cmd_utils # learn

```python run.py --env=PommeRadioCompetition-v2 --policy_type=pgn --num_timesteps=1e7 --log_path=./logs/ --save_path=./models/test/ --save_interval=1e2```

# 演示 或 评测
参数解析可以参考 my_common.cmd_utils # play and evaluate

model 0 --> 4 | using_prune


```python play.py --env=PommeRadioCompetition-v2 --model0=hit18Agent_prune --model1=SimpleAgent_prune --model2=hit18Agent_prune --model3=SimpleAgent_prune --using_prune```

```python evaluate.py --env=PommeRadioCompetition-v2 --model0=hit18Agent_prune --model1=SimpleAgent_prune --model2=hit18Agent_prune --model3=SimpleAgent_prune --using_prune```