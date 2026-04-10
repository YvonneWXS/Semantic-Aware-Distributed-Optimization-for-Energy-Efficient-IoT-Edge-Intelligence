# 代码修改计划

> 最新修改日期：26/4/10 16/52

## 1. 基本要求
+ 添加带宽
+ 说明文档
    + README.md：项目介绍、运行环境、输入文件路径、输出文件路径、参数说明、结果分析
    + Modification.md：修改内容、修改原因、修改影响
+ config.py
    + 批量实验参数 在 config.py 中直接填入【离散参数列表】，例如：
        - DataSize_list（KB） = [100, 200, 300, 400]
        - num_UEs_list = [5, 10, 15, 20]
        - bandwidth_list（kHz） = [750, 1000, 1500, 2000]
        - mec_capacity_list（Giga Cycles/s） = [10.0, 12.5, 15.0, 17.5, 20.0]
        - min_semantic_factor_list = [0.2, 0.3, 0.4, 0.5]
    + 可以手动修改批量实验时非对比条件的其他的参数



## 2. 代码修改

- [x] GA
- [ ] SA-GA
- [ ] IPPO
- [ ] SA-IPPO
- [ ] SA-MAPPO
- [x] Greedy
- [x] ALE(All-Local-Execution)
- [x] ROE(Random-Execution)

