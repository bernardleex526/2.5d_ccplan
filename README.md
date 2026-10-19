# 2.5d_ccplan — 山地巡检机器狗 SLAM 导航系统

面向云深处 M20 Pro 轮足机器狗（上位机 NVIDIA Orin NX 16GB）的山地巡检系统：
遥控建图（带回环）→ 2.5D 高程图 → 重定位 → 多航点自主巡逻，台阶区自动切换楼梯步态。
第一阶段在本地 Gazebo 中用运动学替身验证全部算法链路，算法收敛后再上实机。

技术栈：Ubuntu 22.04 · ROS 2 Humble · Gazebo Fortress · Lightning-LM · elevation_mapping_cupy · Nav2

## 当前状态（2026-10）

| 阶段 | 状态 |
|---|---|
| M1 基础模块（协议、桥接、看门狗、步态状态机） | ✅ 单元 + 集成测试 |
| M1 仿真环境（山体生成、替身模型、地形跟随、启动） | ✅ |
| M2 建图与重定位（T17–T25） | ✅ 小场景实测，基线报告已生成 |
| M3 巡逻闭环（T26–T30） | ✅ 小场景一圈巡逻通过 |
| M3 前端（T31–T35） | ⏸ 暂缓 |
| M4 验收（T36–T37） | ⏳ 待第 2 检查点冻结阈值、完整场景运行 |

### 小场景基线（`hill_world_small`，只报告实测值）

| 指标 | 实测 |
|---|---|
| 建图轨迹 ATE RMSE（SE3 对齐） | 0.13 m，5 次回环 |
| 全局 2.5D 高程图步道区 RMSE / 覆盖率 | 0.062 m / 81% |
| 重定位误差（3 个起点最差） | 0.19 m / 0.54° |
| 巡逻 1 圈 | 4 次步态切换全部正确，违规 0 次，成功率 1.0 |

详见 [`reports/baseline_small_world.md`](reports/baseline_small_world.md)。验收阈值
（`src/hill_patrol_bringup/config/thresholds.yaml`）全部为 `null`，由用户在第 2 检查点冻结。

### 仿真的局限

- 抬脚上台阶由运动学替身"贴地滑行"代替，不含腿部/接触动力学；真实台阶动作留给实机检查点。
- 仿真雷达为均匀扫描，不是 Mid-360 非重复扫描；IMU 由平滑轨迹合成。
- 精度结论以第二阶段实机数据包回放为准（计划风险 K1/K2）。
- 完整场景（上下各 500 级）尚未运行，需原生 Ubuntu + GPU 机器（计划 K7）。

## 系统架构

```
Gazebo Fortress (hill_patrol_sim 生成的山体 + M20 替身 + 运动学地形跟随插件)
   │  ros_gz_bridge: /points_raw /imu /d435i/depth/points /gnss/fix /clock /sim/ground_truth/pose
   ▼
cloud_adapter ──/points_lio──► Lightning-LM（建图 / 定位）──► TF map→base_link（localization_monitor）
   │                                                          └► /localization/ok
   └─► cloud_merger(+D435i) ──► elevation_mapping_cupy ──► traversability_maps ──► Nav2 代价地图
global_elevation_builder: global.pcd → 全局 2.5D 高程图（Nav2 全局代价图 / 网页底图）
patrol_manager ──NavigateToPose──► Nav2 (SmacPlanner2D + MPPI Omni) ──/cmd_vel──► m20_bridge
gait_manager: 台阶区进出 → /m20/hold + /gait/request（1 Basic / 14 Stair），等待 M20 回报确认
m20_bridge ──UDP（M20 协议）──► mock_m20_server（仿真） / M20 实机
```

仿真与实机走同一条 UDP 协议路径：仿真中对端是 `mock_m20_server`，它把轴指令和步态转发给 Gazebo 插件。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/m20_bridge` | M20 UDP 协议编解码、速度换算、看门狗、桥接节点、模拟 M20 服务端 |
| `src/gait_manager` | 台阶区几何、步态切换状态机（停稳→切换→确认→继续）、ROS 节点 |
| `src/hill_patrol_sim` | 山体世界生成器、M20 替身模型生成器、运动学跟随插件（C++）、点云适配/合并、步态规则监测 |
| `src/traversability_maps` | 局部/全局可通行性 GridMap（台阶区设为可通行）供 Nav2 |
| `src/patrol_manager` | 多圈多航点巡逻（开始/暂停/继续/停止、软停） |
| `src/geo_alignment` | 地图 ↔ WGS84 配准 |
| `src/hill_patrol_testing` | 阈值助手、轨迹/高程评估、全局高程图生成、基线报告 |
| `src/hill_patrol_bringup` | launch、配置（安全时序、限速、Nav2、Lightning、高程图）、端到端测试 |
| `src/third_party` | submodule：Lightning-LM、elevation_mapping_cupy、nav2_gridmap_traversability_layer |
| `deps/` | 依赖版本、第三方构建说明（`third_party_build.yaml`）、上游补丁 |
| `reports/` | 基线报告 |

## 安装与构建

```bash
git clone --recursive https://github.com/bernardleex526/2.5d_ccplan.git ~/hill_patrol_ws
cd ~/hill_patrol_ws
bash scripts/setup_env.sh                     # ROS 2 Humble / Fortress / Nav2 / grid_map 等
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

第三方包的构建要点（Pangolin、cupy/NVRTC、补丁、已知坑）见
[`deps/third_party_build.yaml`](deps/third_party_build.yaml)。elevation_mapping_cupy 需要
`deps/patches/elevation_mapping_cupy-get_position.patch`。

WSL2 下：Gazebo 需 Mesa 软件渲染（`sim.launch.py software_gl:=auto` 自动设置），
DDS 建议 `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。

## 使用

```bash
# 1. 仿真 + 遥控（teleop 模式，无需定位）
ros2 launch hill_patrol_bringup sim.launch.py world:=small teleop:=true

# 2. 遥控建图（drive:=true 用路线驾驶脚本代替手柄），结束后保存地图
ros2 launch hill_patrol_bringup mapping.launch.py world:=small drive:=true
ros2 service call /lightning/save_map lightning/srv/SaveMap "{map_id: hill_world_small}"

# 3. 由点云地图生成全局 2.5D 高程图
ros2 run hill_patrol_testing global_elevation_builder \
    maps/data/hill_world_small/global.pcd maps/data/hill_world_small/elevation

# 4. 重定位 + 高程图 + Nav2 + 步态管理
ros2 launch hill_patrol_bringup navigation.launch.py world:=small

# 5. 巡逻（台阶区与航点默认由世界真值生成，代替网页标注）
ros2 launch hill_patrol_bringup patrol.launch.py world:=small laps:=1
ros2 service call /patrol/start std_srvs/srv/Trigger

# 数据录制
ros2 launch hill_patrol_bringup record.launch.py output:=~/bags/run1
```

## 测试

```bash
# 纯逻辑单元测试（CI 中运行）
pytest -m "not gpu and not sim" src/m20_bridge/test src/gait_manager/test ...

# 仿真集成 / 端到端（需 Fortress、GPU，按顺序运行）
pytest -s src/hill_patrol_bringup/test/test_mapping_loop.py         # T20 建图 + 回环
pytest -s src/hill_patrol_bringup/test/test_global_elevation_map.py # T22 全局高程图
pytest -s src/hill_patrol_bringup/test/test_relocalization.py       # T23 重定位
pytest -s src/hill_patrol_bringup/test/test_single_goal.py          # T27 单航点
pytest -s src/hill_patrol_bringup/test/test_gait_patrol.py          # T29 巡逻 + 步态
python3 -m hill_patrol_testing.baseline_report                      # T25 基线报告
```

## 安全

- 安全时序在 `src/hill_patrol_bringup/config/m20_safety.yaml`，当前为**仿真占位值**，
  实机首次运动（CP3）前必须由用户确认。
- `m20_bridge`：指令中断只输出零速度并自动恢复；软停、链路断开、步态切换超时、定位丢失锁死，
  需调用 `/m20/reset_fault` 复位。全程常规模式（Mode=0），安全员持手柄。
- `gait_manager`：进入台阶区先停稳再切 Stair，确认后才放行；离开前按速度外推提前停车切回 Basic；
  超时进入 FAULT 并保持 hold。
- m20_bridge、gait_manager 的修改合并前需人工审查。

## 许可证

自写代码 Apache-2.0；第三方 submodule 遵循各自许可证（许可证核对为计划任务 T36，尚未完成）。
