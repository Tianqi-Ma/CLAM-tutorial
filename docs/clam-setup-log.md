# CLAM 环境搭建实录

> 目标：在 Windows 11 上从零跑通 CLAM（mahmoodlab/CLAM）WSI 分析流水线，供后续整理成 GitHub 文章。
> 起始日期：2026-10-01

## 0. 机器环境

| 项目 | 配置 |
|---|---|
| OS | Windows 11 Home China (26200) |
| CPU | Intel i7-10875H @ 2.30GHz（8 核 16 线程） |
| RAM | 16 GB |
| GPU | NVIDIA RTX 2060, 6GB 显存 |
| 磁盘 | E: 盘剩余 ~364 GB |
| Python | 3.12.2（`C:\Users\34708\AppData\Local\Programs\Python\Python312\`） |
| conda | Miniconda3-PBE 26.7.1.beta0-0，装在 `D:\Software\Miniconda3`（base 为 Python 3.14） |

CLAM 仓库位置：`E:/Projects/DP/CLAM`（git clone 自 `https://github.com/mahmoodlab/CLAM.git`，最新提交含 CONCH v1.5 支持）

**环境方案决策**：官方推荐 `conda env create -f env.yml`。起初检查本机无 conda，曾计划改用 venv 方案；随后用户决定自行安装 Miniconda，**最终走官方 conda 路线**（对 Windows 上 openslide 动态库依赖最友好）。
已有的 `DP/.venv`（其他 DP 项目在用）不动。

---

## 步骤记录

### 1. 定位 conda 安装位置（小坑：装完找不到）

用户装好 Miniconda 后，当前 shell 里 `conda` 不在 PATH，常见默认路径（`C:\Users\34708\miniconda3`、`C:\ProgramData\miniconda3` 等）也都不存在。

排查过程：
- `Get-Command conda` → 无
- 注册表 Uninstall 项找到 `Miniconda3-PBE 26.7.1.beta0-0 (Python 3.14.7 64-bit)`，但 InstallLocation 字段为空
- 最终通过解析开始菜单快捷方式参数定位到：**`D:\Software\Miniconda3`**（用户自定义安装路径）

> 教训：Windows 上装完 conda 后老终端不会自动刷新 PATH；自定义路径安装时，最快的定位办法是查开始菜单里 Anaconda Prompt 快捷方式的 Arguments。

**注意**：装的是 PBE（预览版）Miniconda，base 环境 Python 3.14。不影响使用——CLAM 的 env.yml 会在独立环境里装 Python 3.10。

### 2. 创建 conda 环境

```bash
cd E:/Projects/DP/CLAM
D:/Software/Miniconda3/Scripts/conda.exe env create -f env.yml
```

env.yml 内容（官方原版）：Python 3.10 + conda-forge 的 openslide + pip 安装 torch/torchvision、timm==0.9.8、h5py、opencv、scikit-learn、smooth-topk（git 源码）、tensorboardX 等。

**报错 1：`CondaToSNonInteractiveError`**（新版 conda 特有）
```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels:
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r
    - https://repo.anaconda.com/pkgs/msys2
```
解决：按提示逐条接受
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

**报错 2：pip 阶段 git clone GitHub 超时**
conda 部分（52.5MB，含 openslide 4.0.1、python 3.10.0）全部装好，但 pip 安装阶段在克隆 `git+https://github.com/oval-group/smooth-topk.git` 时 `Failed to connect to github.com port 443`（国内网络对 GitHub 不稳定）。**注意：此时环境已建好但只有 conda 包，pip 依赖一个都没装上**。

### 3. 手动补装 pip 依赖（分两步）

策略：与 env.yml 等价的 pip 依赖手动装；torch 等大包走清华镜像；smooth-topk 单独用 `--no-build-isolation` 装（跳过 PEP 517 构建隔离，避免它再去网上拉 setuptools——第一次重试时就是挂在构建子进程访问镜像 SSL 断连上）。

```bash
PY=/d/Software/Miniconda3/envs/clam_latest/python.exe
# 第一步：主依赖（torch 约 2.5GB，走清华镜像）
$PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    timm==0.9.8 torch torchvision h5py pandas PyYAML opencv-python \
    matplotlib scikit-learn scipy tqdm openslide-python tensorboardX
# 第二步：smooth-topk（--no-build-isolation 用环境自带 setuptools）
$PY -m pip install --no-build-isolation "git+https://github.com/oval-group/smooth-topk.git"
```

> 踩坑记录：`pip install ... 2>&1 | tail -30` 这种写法会让管道吃掉 pip 的退出码（pipeline 返回 tail 的退出码 0），失败被掩盖成"成功"。排查时要看完整输出文件。

**报错 3：清华镜像持续 SSL 断连（SSLEOFError）**
pypi.tuna.tsinghua.edu.cn 对本机网络持续握手失败，5 次重试全挂，导致第一次主依赖安装"零安装"（pip 在解析 smooth-topk 的构建依赖时整个任务就中止了——git+ 依赖会最先处理，它失败则后面所有包都不会装）。

实测各镜像可达性（curl 探测 `/simple/timm/`）：

| 镜像 | 结果 |
|---|---|
| pypi.org 官方 | ✅ 200, 0.51s |
| 阿里云 mirrors.aliyun.com | ✅ 200, 0.41s |
| 腾讯云 mirrors.cloud.tencent.com | ✅ 200, 0.42s |
| 清华 tuna | ❌ SSLEOFError |
| 中科大 ustc | ❌ 无响应 |
| 北外 bfsu | ❌ 无响应 |

最终主依赖走**阿里云镜像**，smooth-topk 仍从 GitHub clone（git 协议当时是通的）。

**报错 4（本次最硬的坑）：pip 全线 TLS 握手失败，但 curl/urllib/git 全都正常**

现象：环境内 pip（26.2.1）访问**任何** PyPI 源（清华、阿里、pypi.org 官方）都在 TLS 握手阶段被断开：
```
SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:997)')
```

排查链（对照实验）：

| 客户端 | 目标 | 结果 |
|---|---|---|
| curl | tuna / aliyun / pypi.org | ✅ 200 |
| Python urllib（同环境 py3.10） | aliyun | ✅ 200 |
| git clone | github.com | 时通时断 |
| **pip（任意源）** | 全部 | ❌ SSLEOFError |

即：同一解释器、同一目标，pip 的 TLS 指纹被中间设备精准掐断，其他客户端放行。本机无系统代理（`netsh winhttp show proxy` 为直接访问），疑似运营商/网关层面对 pip 的 TLS ClientHello 指纹做阻断。

**解法：换 TLS 栈**——改用 `uv`（Rust 编写，用 rustls 而非 OpenSSL，指纹完全不同）：
```bash
uv pip install --python "D:\Software\Miniconda3\envs\clam_latest\python.exe" \
    timm==0.9.8 torch torchvision h5py pandas PyYAML opencv-python \
    matplotlib scikit-learn scipy tqdm openslide-python tensorboardX
```
uv 直连 pypi.org 立即成功（验证：本机已有 uv 0.11.7，`C:\Users\34708\.local\bin\uv.exe`）。

注意：uv 一次性解析所有依赖时，若含 `git+...` 依赖而 GitHub 恰好断连，会整体失败（uv 的 git fetch 报 `Connection was reset`）。**git 依赖务必拆出来单独装**，可重试：
```bash
uv pip install --python <env_python> "git+https://github.com/oval-group/smooth-topk.git"
```

**报错 5：uv 下载 torch 中途被断流 + PyPI 的 torch 是 CPU 版**

两个叠加的问题：
1. **PyPI 上的 Windows 版 torch 是 CPU 版**（wheel 仅 ~118MB）。CUDA 版必须从 `https://download.pytorch.org/whl/cu124` 装（文件名带 `+cu124` 后缀）。
2. uv/rustls 虽然能过 TLS 握手，但**下载大 wheel 时连接被中途掐断**（`peer closed connection without sending TLS close_notify`），重试 5 次都在中途死——本网络对"长连接大流量"有阻断行为。

另外注意 timm 依赖 torch，即使不显式列 torch，uv 也会自动拉 CPU 版 torch——所以 timm 也要等 CUDA torch 就位后再装。

**解法：curl 断点续传 + 离线安装**
curl 是全场唯一被放行的客户端（169MB 的 CMU-1.svs 一次下完）。用 `-C -` 断点续传 + 重试循环下载 CUDA 版 wheel（约 2.5GB），再本地安装：
```bash
for i in $(seq 1 60); do
  curl -sSL -C - --retry 10 --retry-all-errors --retry-delay 3 \
    -o "torch-2.6.0+cu124-cp310-cp310-win_amd64.whl" \
    "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp310-cp310-win_amd64.whl" \
    && break
  sleep 3
done
uv pip install --python <env_python> ./torch-2.6.0+cu124-...whl
```
选 `torch==2.6.0+cu124` 是因为本机另一个 venv 已在用这个版本，RTX 2060（Turing, sm_75）验证可用。
实测 download.pytorch.org 的 CDN 直连极快：**2.5GB 约 1 分钟下完**（~40MB/s），之前的"断流"只针对 pip/uv 的 TLS 指纹。torchvision 同理下载 `torchvision-0.21.0+cu124` 后本地装。

**报错 6：smooth-topk（topk 包）缺 `future`**
`import topk` → `ModuleNotFoundError: No module named 'future'`。smooth-topk 是 2018 年的老包，依赖 future 但没在 setup.py 里声明。解决：`uv pip install future`（装的是 future==1.0.0）。

**报错 7：openslide 找不到 DLL（Windows + Python 3.8+ 经典问题）**
conda 装的 `libopenslide-1.dll` 在 `envs/clam_latest/Library/bin/`，但 Python ≥3.8 的 ctypes 不再搜 PATH，`cdll.LoadLibrary('libopenslide-1.dll')` 直接 FileNotFoundError。
解决：`uv pip install openslide-bin`（openslide-python ≥1.4 会自动发现 openslide-bin 打包的 DLL）。装完后验证通过：
```
slide dims: (46000, 32914) | levels: 3 | mpp-x: 0.499 | vendor: aperio
```

### 4. 流水线第一步：分割 + 切块 ✅

```bash
python create_patches_fp.py --source E:/Projects/DP/DATA_DIRECTORY \
    --save_dir E:/Projects/DP/RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch
```
CMU-1.svs 结果：**分割 0.33s + 切块 7.15s + 拼接 0.46s**，共提取 4679 个 patch。masks/ 里绿色轮廓准确圈出 4 块组织，stitches/ 重建图背景全黑无杂质——分割质量完美。

### 5. 特征提取（GPU 步骤，坑最密集）

**坑 A：csv 要去扩展名**
`process_list_autogen.csv` 里 slide_id 是 `CMU-1.svs`，而提取脚本会自己拼 `--slide_ext`。需准备只含 `CMU-1` 的 csv：
```bash
printf 'slide_id\nCMU-1\n' > process_list_feat.csv
```

**坑 B：`--data_h5_dir` 传的是上级目录**
脚本内部自己拼 `'patches'` 子目录（`extract_features_fp.py:88`），所以应传 `RESULTS_DIRECTORY` 而非 `RESULTS_DIRECTORY/patches`，否则报 `patches\patches\CMU-1.h5 不存在`。

**坑 C：timm 0.9.8 要求装 huggingface_hub，但又与新版不兼容**
- 不装：`RuntimeError: Hugging Face hub model specified but package not installed`
- 装最新版（2.0.0）：同样报错——timm 0.9.8 用 try/except 包住 `from huggingface_hub import cached_download` 来判断是否可用，而 cached_download 在 huggingface_hub 0.26 被移除，导入失败被误判为"没装"
- 解决：**`uv pip install huggingface_hub==0.25.2`**（cached_download 存在的最后版本区间）

**坑 D：系统代理截杀 huggingface 请求**
Windows 注册表 `ProxyEnable=1, ProxyServer=127.0.0.1:10809`（且无绕过列表），但代理客户端没在运行。requests 库默认读取系统代理 → huggingface.co 和 hf-mirror.com 全部 `ProxyError: Unable to connect to proxy`。
而 curl/git/uv 不读注册表代理——这解释了为什么之前它们能直连。

**坑 D 的解法：离线播种 HF 缓存（零改 CLAM 代码）**
timm 的 `resnet50.tv_in1k` 就是 torchvision 的 IMAGENET1K_V1 权重（"tv"=torchvision），而 download.pytorch.org 直连飞快：
```bash
# 1. 下载 torchvision 权重（98MB）
curl -sSL -C - -o resnet50-0676ba61.pth https://download.pytorch.org/models/resnet50-0676ba61.pth
```
```python
# 2. 转成 safetensors，按 HF 缓存结构摆放
import torch, os
from safetensors.torch import save_file
sd = torch.load('resnet50-0676ba61.pth', map_location='cpu', weights_only=True)
sd = {k: v.contiguous() for k, v in sd.items()}
sha = 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2'  # 任意 40 位 hex
repo = os.path.expanduser('~/.cache/huggingface/hub/models--timm--resnet50.tv_in1k')
os.makedirs(f'{repo}/snapshots/{sha}', exist_ok=True)
os.makedirs(f'{repo}/refs', exist_ok=True)
save_file(sd, f'{repo}/snapshots/{sha}/model.safetensors')
open(f'{repo}/refs/main', 'w').write(sha)
```
```bash
# 3. 离线模式运行（hf_hub_download 直接命中缓存）
HF_HUB_OFFLINE=1 python extract_features_fp.py ...
```

**坑 E：Windows 下 DataLoader num_workers=8 崩溃**
`ValueError: ctypes objects containing pointers cannot be pickled`——Windows 的 multiprocessing 是 spawn 模式，dataset 里的 OpenSlide 对象（ctypes 指针）无法 pickle 给子进程。Linux 的 fork 没这问题，所以官方代码没处理。
解法：给 `extract_features_fp.py:83` 打一行 Windows 适配补丁：
```python
# 原代码：
loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
# 改为：
loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" and os.name != 'nt' else {'num_workers': 0, 'pin_memory': True}
```

**最终结果** ✅
```
computing features for CMU-1.h5 took 35.8 s
features size:  (4679, 1024)
coordinates size:  (4679, 2)
```
RTX 2060 + batch_size=64，35.8 秒完成 4679 个 patch 的特征提取。输出 `FEATURES_DIRECTORY/pt_files/CMU-1.pt`（19MB）+ `h5_files/CMU-1.h5`（含 coords+features）。抽查特征向量模长 1.78~2.13，分布正常。

---

## 当前状态总结

| 流水线环节 | 状态 |
|---|---|
| 环境（conda clam_latest + CUDA torch 2.6.0） | ✅ |
| openslide 读 WSI | ✅ |
| 分割 + 切块（CPU） | ✅ 8.4s/张 |
| 特征提取（GPU, ResNet50） | ✅ 35.8s/张 |
| 训练 main.py | ⬜ 需要带标签的多切片数据集 |
| 热图 create_heatmaps.py | ⬜ 需要训练好的模型权重 |

---

## 6. 项目重组 + TCGA 数据下载（2026-10-01 下午）

**项目结构整理**：应用户要求，所有产出集中到 `E:/Projects/DP/CLAM-tutorial/`（data/scripts/notebooks/results/wheels/docs），本日志移至 `docs/`。CLAM 官方仓库 clone（`DP/CLAM`）保持干净，仅有一处补丁。

**下载脚本**：`scripts/download_tcga.py`（详见 notebooks 第 4 节）。
- 目标：TCGA-LUAD/LUSC 诊断切片各 50 张（≥0.3GB 过滤后随机抽样，seed=42），共 100 张 / 88.7GB
- 附带：同名患者临床数据（clinical.json）+ RNA-seq STAR counts（metadata/expression/）→ 多组学扩展备用

**报错 9（新）：git-bash curl 与 Windows curl 行为分裂**
- git-bash 的 curl（OpenSSL）：连 GDC 直接 TLS 握手被杀（和 pip 同款指纹阻断）
- Windows 版 curl（`C:\Windows\System32\curl.exe`，schannel）：报 `CRYPT_E_REVOCATION_OFFLINE`——连接其实通了，只是查不了证书吊销列表
- 解法：统一用 Windows curl + `--ssl-no-revoke`（本项目所有 GDC 流量走它）

### 7. Jupyter kernel 注册 + notebook 无头执行（2026-10-01 下午）

- 给 clam_latest 装 `ipykernel nbclient nbformat nbconvert`（uv），注册内核：
  ```bash
  python -m ipykernel install --user --name clam_latest --display-name "CLAM (clam_latest)"
  ```
  → VSCode/Jupyter 里选 **"CLAM (clam_latest)"** 内核即可。notebook 的 kernelspec 元数据已固定为该内核，打开自动匹配。
- 无头执行（产出可复现的结果 cell）：`nbclient` API 加载 notebook → 执行前 14 个 cell（TCGA 批量处理等数据下载完再补）→ 输出写回同一文件。全部 ✅，含 mask/stitch 内联图片。

### 8. Notebook 小白化改写 + 数据预览增强（2026-10-01 下午）

按用户反馈两轮迭代 `notebooks/CLAM_workflow.ipynb`：
1. **小白化**：每步四件套（🎯目的/📥输入/⚙️原理/📤输出）+ 开篇科普（WSI/MIL/注意力名词表 + "5000页的书"比喻）+ 代码逐行注释
2. **数据预览**：每个产物都加"看数据"cell——process_list.csv 前几行、h5 坐标形状、特征矩阵统计、抠出一个真实 patch 显示、切片缩略图、splits/标签表/评估结果的守卫式预览（数据未就绪时打印提示不报错）
3. **第 7 节深挖**：对照 `model_clam.py` 源码画出 CLAM_SB 真实结构图（fc1→门控注意力→加权平均→分类器 + 实例级聚类支线），参数量实测 790,791；附全参数选择指南表
4. **模型演示 cell**：实例化 CLAM_SB + 随机数据前向，亲眼看到 A/logits/Y_prob 的形状

**新坑（顺手修的）**：子进程中文输出 GBK 乱码 → run() 里统一注入 `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`。

---

## 9. 教学化改造第二轮：CLAM 原理玩具版 notebook（2026-10-01 下午）

**起因**：主 notebook 第 7 节（训练 CLAM）对零基础读者仍然太硬——一次堆了形式化、架构图、门控、实例级损失 5 个概念。
用户决定：新开一个玩具版 notebook 讲原理，主版第 7 节精简成直觉版。

**产出 `notebooks/CLAM_原理_玩具版.ipynb`**（CPU 几分钟跑完，已嵌入全部输出）：
- 合成数据：200 张假切片，每张 100 个 4 维 patch；肿瘤切片藏 5 个 N(2,1) 的肿瘤 patch（有标准答案可验证）
- 实验 1 笨办法 mean pooling：准确率卡在 77%
- 实验 2 手写 30 行注意力 MIL（`A=softmax(w·h); M=ΣA·h; ŷ=clf(M)`，共 10 个参数）：99.5%
- 见证时刻：注意力 Top-5 patch 命中真肿瘤 4/5（没给过 patch 级标签）
- 实验 3 肿瘤占比扫描（10%→1%）：mean pooling 从 97% 崩到 63%，注意力保持 85~99% → 为什么真实病理必须注意力
- 收尾：玩具版 ↔ CLAM_SB 逐层对照表（门控/实例级支线各自加了什么、为什么）

**主 notebook 第 7 节精简**：删掉形式化数学和架构图（搬到玩具版的对照表里），保留三分钟直觉 + 完整参数表 + 训练观察要点 + 模型实例化演示 cell。

**踩坑记录（本轮新增）**：
- 坑 11：nbconvert `--inplace` 执行疑似因前台超时转后台而没落盘（退出码 0 但输出为空）。解法：改用 `--output 新文件` 再覆盖，可靠。
- 坑 12：matplotlib 默认 DejaVu Sans 无 CJK 字形，图里中文全是方块（stderr 刷 Glyph missing 警告）。解法：`plt.rcParams['font.sans-serif']=['Microsoft YaHei']` + `axes.unicode_minus=False`。
- 后台任务有 2 小时上限：88.7GB 的 TCGA 下载在 35/100 处被杀。好在 download_tcga.py 有 manifest 断点续传，重启即续（需分段跑完）。

## 10. TCGA 数据下载完成（2026-10-01 19:56）

- **100/100 张切片全部下载完成**，共 83GB（LUAD 50 + LUSC 50，每张 ≥0.3GB，种子 42 可复选）
- 实际耗时约 5.9 小时纯下载（14:03 起，三次后台任务接力：后台命令有 2h 上限，靠 manifest 断点续传无缝衔接）
- 附带产物：`tcga_luad_lusc.csv`（100 行标签表）、`clinical.json`（97 个病例的 demographic/diagnoses/exposures）、`metadata/expression/`（114 个 RNA-seq STAR-Counts 文件）
- 随即启动第 5 步批量预处理（后台）：100 张切片的分割+切块+拼接 → 特征提取（batch_size 64，HF_HUB_OFFLINE=1）

## 11. 教学化改造第三轮：玩具版系列扩展到四个模型（2026-10-01 晚）

**起因**：用户反馈 ① 要 double check CLAM 原理正确性 ② softmax 还是看不懂（无 DL 背景）③ 用户还用过 CONCH/TITAN/UNI2，各做一个玩具版。

**原理核对（全部通过）**：
- CLAM_SB 直接对照 `models/model_clam.py` 源码逐行核实：fc1(1024→512)+ReLU+Dropout → Attn_Net_Gated(a=tanh 路, b=sigmoid 路, A=a⊙b, c:256→1) → softmax(A, dim=1 沿 patch 维) → M=A@h → Linear(512→2)；实例支线 top/bottom 各 k_sample=8（subtyping 时类外分支只取 top-p 当负样本）；默认实例损失 CE，训练时 --inst_loss svm 换 SmoothTop1SVM。玩具版对照表与源码一致。
- 网络检索核实：UNI2-h = ViT-H/14 (681M), 1536 维, DINOv2 配方, 200M+ tiles / Mass-350K；CONCH = CoCa(对比+captioning), ViT-B/16, 512 维, 1.17M 图文对, 视觉端 iBOT 预热(16M patches)；TITAN = 三阶段(iBOT 遮住重建 → ROI-caption 423K → WSI-report 183K, CoCa), CONCHv1.5 768 维输入 + ALiBi, Mass-340K=335,645 张。来源：arXiv 2411.19666 / 2307.12914、HF 模型卡、Nature Medicine。

**CLAM 玩具版加固**：新增 1.5 节"两个数学零件"（加权平均=成绩总评比喻；softmax 三步手算 [2.0,0.5,-1.0]→[0.79,0.17,0.04]，解释为何不能"直接除以总和"、为何总和=1 是预算约束、为何 e^x 放大差距）+ 4.5 节"5 个 patch 手算完整前向"（打分→softmax→加权平均→分类，每步打印真实数字 + 注意力条形图）。

**三个新玩具 notebook**（均已执行嵌入输出）：
- UNI2：3 类组织 blob + 扫描仪增益/偏色污染；SimCLR-lite（1800 里认另一半）；kNN 原始 vs 学习后对比图
- CONCH：12 词玩具词典（5 肿瘤词/5 正常词/2 干扰词），图 Linear(4→2) 文 Linear(12→2)，InfoNCE 双向；零样本分类 + 以文搜图演示
- TITAN：[SLIDE] 令牌 + 1 层 Transformer(d=16)，遮 25% patch 重建(MSE)；指纹上线性探测 vs 取平均、PCA 可视化、相似病例检索
- 每个都有"玩具版 vs 真实模型"诚实对照表 + 系列全景图（四模型在流水线中的位置）

### 附：玩具实验的"实验设计审查"（教学 demo 也会踩的坑）

第一版玩具实验自身有三个设计错误，被验证环节抓出后修正：
1. **有序标签 + 连续切分 = 泄漏评估**：`labels = [0]*300+[1]*300+...` 配 `前2/3训练` 的切分 → 测试集全是训练里不存在的类，kNN 准确率 0% 的假灾难。修正：`train_test_split(stratify=)`。
2. **UNI2 玩具的"叙事错位"**：kNN 是局部方法，批次效应破坏的是全局对齐——随机划分下原始特征 92% 根本没问题。正确考点是**留一扫描仪（LOSO）**：原始特征 32%（崩）→ 对比学习后 91%（修好）。这才讲对"换医院就崩"的故事。
3. **TITAN 玩具的"重建不可学"**：patch 若彼此独立（iid 噪声），遮住重建的最优解就是猜均值，学不到上下文。修正：给切片加**区域结构**（10 区域×10 patch，同区域同组织类型），上下文才真正包含可学习的信息（MSE 2.15→0.95，噪声下限 0.16）；同时补"闭眼猜 0"基线替代原来错误的"随机≈2.0"说法。
4. CONCH 玩具的"图找对文 top-1"指标无意义（12 词词典图注定大量重复），改为**类别级配对准确率**。

教训：玩具实验和真实实验一样需要审查评估协议——标签有序、泄漏、指标无意义，样样都会骗人。

### 玩具版系列最终结果（已执行嵌入）

| notebook | 核心实验 | 结果 |
|---|---|---|
| CLAM_原理_玩具版 | softmax 手算 + 5-patch 全流程手算 + 注意力 vs 取平均 + 占比扫描 | 注意力 99.5% vs 取平均 77%；Top-5 命中真肿瘤 4/5；肿瘤 patch 获得 99.2% 注意力预算 |
| UNI2_原理_玩具版 | 3组织×3扫描仪，随机划分 vs 留一扫描仪（LOSO） | 原始特征：随机 92% / LOSO 32%（崩）；对比学习后 LOSO 90.2%，扫描仪可识别度 100%→58% |
| CONCH_原理_玩具版 | 12 词词典图文对齐（InfoNCE 双向） | 类别级配对 99.8%；零样本分类 96.5%；以文搜图 8/8 |
| TITAN_原理_玩具版 | 区域结构切片 + 遮住重建 + 切片级一致性（iBOT 两零件） | 重建 1.07 vs 瞎猜 2.16；指纹线性探测 100%（取平均 96%）；相似病例检索 5/5 |

第二轮修正的 bug：UNI2 变量名笔误（SCAN→SCANNERS）、TITAN 补丁丢了类定义（NameError）、UNI2 反向考试误用 LOSO（标签不存在导致恒 0）→ 改随机分层划分。

## 12. 特征提取提速 6 倍：Windows 多进程加载补丁 v2（2026-10-02 凌晨）

**问题**：batch=128 后吞吐仍只有 ~118 patch/s，全量 563 万 patch 预估 13h+。瓶颈排查：
- `Whole_Slide_Bag_FP.__getitem__` 每取一个 patch 开关一次 .h5 文件（batch=128 → 128 次开关）
- Windows 下 num_workers=0（v1 补丁），OpenSlide 读取单线程
- 一次 2h 窗口只完成 2 张切片，且最后一张疑似卡死（tqdm 停在 66% 后 89 分钟无输出，原因未复现）

**补丁 v2（extract_features_fp.py，仅 Windows 分支生效）**：
- 新增 `Whole_Slide_Bag_FP_Win` 子类：不持有 OpenSlide 对象（存路径，worker 内惰性打开，解决 pickle）；
  `__init__` 一次性把 coords 读进内存（消灭每 patch 的 h5 开关）
- loader_kwargs 的 Windows 分支改为 `num_workers=4, pin_memory=True`

**坑**：首次以 batch=128 + 4 workers 运行时 CUDA OOM——桌面应用（抖音等）占去 ~2.4GB 显存，只剩 3.6GB。
解法：batch 降回 64（多进程后吞吐瓶颈在 I/O 不在 batch 大小）+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**效果**：~60 patch/s → ~350 patch/s（6×），单张 14 万 patch 的大切片 379s 完成。全量 ETA 13h+ → ~4.5h。

附注：TaskStop 杀后台 bash 时其 python 子进程可能存活，曾观察到 GPU 被占用导致的连锁 OOM——重启前用 nvidia-smi 确认无僵尸。

**坑 12b**：4 workers + pin_memory 跑约 1 小时后 `RuntimeError: Couldn't open shared file mapping, error code 1455`——
Windows 的 DataLoader worker→主进程共享内存走页面文件，16GB 内存 + 小页面文件撞 commit limit。
解法（保守稳定）：workers 4→2、pin_memory 关、prefetch_factor 2。速度 ~350→~200 patch/s 换取不崩。
（若把 Windows 页面文件手动调大，可再回到 4 workers。）

### 补充（特征提取后半程）
- workers=2 跑了约 2 小时后再次撞 **error 1455**（共享内存 commit 上限，这次是 collate 阶段 `_new_shared`）——16GB 内存 + 桌面程序常驻时，双 worker 的在途 batch 共享映射余量不足
- 最终稳定配置：`num_workers=1, pin_memory=False, prefetch_factor=2`，~176 patch/s（batch 64），与 workers=2 的 ~200 相差不大但全程无共享内存压力
- 中途崩溃无损失：`--no_auto_skip` 默认关闭，重启自动跳过已完成 .pt，从断点续跑

## 13. 训练阶段踩坑实录（2026-10-02）

1. **scipy≥1.11 不兼容**：`create_splits_seq.py` 里 `stats.mode(字符串数组)` 直接 TypeError。改 `dataset_generic.py` 的 maj 投票为 `np.unique(return_counts=True)` + argmax（行为等价）。
2. **numpy 2.0 不兼容**：`np.Inf`（core_utils.py EarlyStopping）和 `np.NaN`（batch_process_utils.py）已删除，改 `np.inf` / `np.nan`。
3. **expandable_segments 与 WDDM 冲突**：训练时 `CUDA error: unknown error`（ReLU 处随机崩）。Windows 显示卡上这个选项是雷，训练不要开（特征提取时倒是稳定）。冒烟测试（最大切片 14.7 万 patch 前向+反向，峰值 2.7GB）证明模型和数据没问题，是分配器的问题。
4. **训练 DataLoader 硬编码 num_workers=4**（utils/utils.py get_split_loader）：每 batch 一整张切片的特征（最大 573MB）走共享内存映射 → 立刻 error 1455。Windows 下改 0。
5. **多折连跑段错误（exit 139）**：fold 0 完成后崩，疑似 tensorboard 事件文件多轮写。对策：去掉 `--log_data`，逐折独立进程（`--k_start i --k_end i+1`），配合接力模式。
6. **后台跑训练必须 `python -u`**：否则 stdout 全缓冲，日志一片空白（差点误判成"启动即崩"）。
7. **dataset_generic.py 会自己拼 `pt_files/`**：补丁里的 data_dir 要给特征根目录，给到 pt_files 会路径叠加。
8. 第 0 折实测：~35s/epoch（2060），第 5 个 epoch val AUC 已到 0.76。
9. **段错误真凶 = pagefile commit 耗尽**（不是 tensorboard）：16GB RAM + 系统管理 pagefile 仅 7.5GB，每次 torch.load 一张 600MB 特征就吃掉一大块 commit 配额，积累到上限进程直接被秒（exit 139，无 traceback）。修复：`torch.load(..., mmap=True)`——特征改由磁盘文件背书，完全不占 pagefile。此招对一切"Windows 上跑大文件深度学习"通用。若想根治可手动加大 pagefile 到 E 盘（需管理员）。
10. **段错误的精确案发现场**（faulthandler 抓到）：`collate_MIL` 里 `torch.cat`——batch_size=1 时对每张切片做一次全量拷贝（最大 600MB），mmap 省下的内存又被它吃回去，commit 耗尽即 access violation。修复：batch_size=1 时直接透传不 cat。**教训：崩溃点与 epoch 无关，与"当时内存水位"有关，所以看着像随机。**

## 14. 训练与评估结果（2026-10-03）

- 任务：LUAD vs LUSC 二分类（50/50），CLAM-SB，ResNet50 特征（1024 维），5 折交叉验证（按患者分层，79/10/11）
- 超参：lr 2e-4, dropout 0.25, bag_loss ce, inst_loss svm, bag_weight 0.7, weighted_sample, early_stopping（各折 30~51 epoch 收工）
- **5 折 test AUC：0.880 / 0.833 / 0.960 / 0.920 / 1.000 → 均值 0.919 ± 0.065**
- 训练时长：每折约 25~40 分钟（RTX 2060，~35s/epoch）
- 训练期坑（已修复）：collate_MIL 的 batch1 全量拷贝（faulthandler 定位）、torch.load mmap、scipy mode、np.Inf、expandable_segments、DataLoader workers——详见第 13 节

## 15. 注意力热图（轻量自研版）
- 没用官方 `create_heatmaps.py`（它要按 overlap=0.5 重切重提特征，数小时/张）；自研 `scripts/make_heatmap.py`：`attention_only=True` 拿分数 → patching 的 coords 画回缩略图，**每张几秒**
- fold-4 模型测试集 11/11 全对；已渲染 LUAD（TCGA-55-8302）与 LUSC（TCGA-22-4613）示例于 `results/heatmaps/`，高注意力区集中在致密肿瘤区
- 注意 checkpoint 含 SmoothTop1SVM buffer，重建模型要 `instance_loss_fn=SmoothTop1SVM(2)` 才能 load_state_dict

## 16. 收尾：批量热图 + notebook 定稿（2026-10-03）
- 热图从 2 张示例扩到 **fold-4 测试集全部 11 张**（results/heatmaps/），每张几秒
- `CLAM_workflow.ipynb` 定稿为 GitHub 主文档：标题加完成状态、§8 补 5 折成绩表+诚实声明、评估/划分 cell 注入实际输出、§9 热图 cell 内嵌 2 张示例图（共 4.5MB）、踩坑速查表扩到 19 条、修 eval 守卫路径、nbformat 校验通过
- README 同步更新进度清单与结果入口

## 17. notebook 瘦身定稿（2026-10-03）
- 应要求删除全部 CMU 冒烟测试段落（原 §2/§3，共 9 cells）：workflow 只保留 TCGA 主线
- 下载 section 改为"已下载完成，仅核验"：删除演习查询/进度检查 cells，替换为磁盘核验（100 svs/83GB/manifest/标签分布）+ TCGA 缩略图预览
- 环境验证、预处理、训练、评估各 cell 注入真实输出（版本信息/5.6M patch/44GB/模型结构/splits/成绩/热图），GitHub 上不执行也能看到全部关键证据
- 36 → 25 cells（2.5MB），节号重排：0 路径 / 1 环境 / 2 数据核验 / 3 预处理 / 4 划分 / 5 训练 / 6 评估 / 7 热图 / 8 踩坑 / 9 后续

## 18. 预处理可视化补强：QC 对比图 + 特征预览（2026-10-03）
- 应要求把冒烟测试期的"分割掩膜 vs 拼接重建"对比图接回 workflow：新图 `results/tcga/qc_mask_vs_stitch.png`（左=官方 masks 绿轮廓，右=stitches 坐标重建），插入 notebook §3c
- 新增 §3d 特征预览（小白向）：① 一个 patch ↔ 它的 1024 维特征向量长什么样（`feat_preview_vector.png`，均值≈0、大量 0 值=ReLU 特性）；② 随机 2000 patch 的 PCA 散点（`feat_preview_pca.png`）——两类癌的 patch 在特征空间里已经隐约成团，说明 ResNet50 特征含形态信息，CLAM 的注意力只是"再加一层筛选"
- 教学要点写进 notebook：特征提取 ≠ 黑盒魔法，是"地图翻译"——相似的 patch 坐标相近

## 19. 多组学分析：形态 × 基因表达（2026-10-03）
- 脚本 `scripts/analysis_multiomics.py`；数据 `data/metadata/expression/*.tsv`（GDC STAR counts，114 文件 → 匹配标签 97 患者 × 19,938 蛋白编码基因，log1p TPM）
- **Marker 体检**：NKX2-1 LUAD 高（4.87 vs 2.15, p=2e-11）；TP63/KRT5/KRT6A LUSC 高（p 最低 3e-14）→ 数据健康
- **重大发现（也是大坑 #11）**：CLAM 官方 `create_splits_seq` 不是教科书 k 折！每折独立分层随机 80/10/10（蒙特卡洛 CV），fold 间测试集可重叠。实测：100 张里仅 45 张进过测试集（7 张两次）、26 张仅 val、**29 张 5 折全在 train**。原"out-of-fold 嵌入"脚本直接 KeyError
- 选折政策：test 折（45, 零泄漏）→ val 折（26, 近似无泄漏）→ 借用 fold-0（29, 有泄漏需复核）
- **结果**：4 个 marker 全部最强相关于形态 PC1（|r|=0.49~0.58, p 最低 5e-10）；岭回归形态→NKX2-1 R²=0.11；干净子集（n=69）复核 r=+0.55 vs 全体 +0.58、R² 反升 0.17 → 结论非泄漏造成
- 产物：`results/multiomics/{markers_boxplot.png, morph_vs_expr.png, console_output.txt}`；已插入 notebook §8

## 20. 临床生存分析：KM 曲线 + 缺失数据陷阱（2026-10-03）
- 脚本 `scripts/analysis_survival.py`（lifelines）；解析 `clinical.json`：Dead→days_to_death（event=1），Alive→days_to_last_follow_up（diagnoses 兜底），stage 取首个非空
- **缺失陷阱（教学核心）**：97 患者中 26 个完全无随访时间，**全是 Alive 的 LUAD** → 朴素 LUAD vs LUSC KM 给出 0.9 年 vs 3.7 年 p<0.001 的"显著差异"纯属假象；图中明确标注"此对比无效"作反面教材
- **主分析（分期 KM）**：I/II/III = 39/17/11 例（IV 仅 2 例不画），I vs III log-rank p=0.15，方向符合临床常识
- 坑：matplotlib 雅黑无 ⚠️ 字形（标题变豆腐块□□）→ 改【警告】文字
- 产物：`results/survival/{survival_table.csv, km_by_stage.png, km_luad_vs_lusc.png, console_output.txt}`；已插入 notebook §9

## 21. notebook 终审修正（2026-10-03）
- **修掉两个会 SyntaxError 的隐患格**：cell 24（评估预览）和 cell 26（热图）注入输出时 `\n` 转义被写成了真实换行，单引号字符串断裂——全部重写源码并 ast.parse 全量体检
- **全面更正"5 折 CV = 每张恰好考一次"的错误表述**：cell 0 名词表/目录表、cell 15 k 折定义、cell 22 诚实声明、cell 24 预览文字同步改为蒙特卡洛 CV 口径；踩坑表新增 #11
- 目录表重写为真实节号（§0~§10，含耗时与产出）；§11 后续更新：多组学/生存分析标记已完成，新增 Cox 预后模型与严格 k 折改造两条
- 最终 35 cells：0 路径 / 1 环境 / 2 数据核验 / 3 预处理(3c QC, 3d 特征预览) / 4 划分 / 5 训练 / 6 评估 / 7 热图 / 8 多组学 / 9 生存 / 10 踩坑 / 11 后续，nbformat 校验通过

## 22. 严格 k 折 + 深入分析原型（2026-10-04 上午，GDC 维护期间的练兵）
- **动机**：坑 #11 发现 CLAM 官方划分是蒙特卡洛 CV。用户拍板：数据扩到 150 张，官方抽样和严格真 5 折**两套都做**；150 结果取代 100 成为主文档口径
- **GDC 维护**：api/portal 503（BigIP 官方维护页），主站正常；下载循环挂后台自动续（每 3~5 分钟重试）；**与 VPN 无关**
- **严格 100 练兵**（GPU 空闲利用）：`scripts/make_strict_splits.py`（StratifiedGroupKFold，按患者分组）→ `splits/task_2_strict100`；训练 5 折（exp `luad_lusc_CLAM_sb_strict100`，会话重启杀后台后 fold-4 无损续训）
- **严格 100 成绩**：AUC 0.689/0.838/0.820/0.964/0.919，均值 **0.846**（对照 MC-100 的 0.919）——MC 更"亮"部分因为 55 张从未进考场；严格折每张都考、每折考 20 张，估计更保守诚实
- **深入分析三件套全部跑通**（严格100零泄漏嵌入：100/100 张 test 折来源）：
  - `analysis_de.py`：全基因组 DE（15,193 基因，BH-FDR）→ 1,340 显著（LUAD高592/LUSC高748）；火山图 marker 自检全在顶部（DSG3/SFTPB/NKX2-1/KRT5/6A）；Enrichr 通路富集（curl 直连 maayanlab 正常）
  - `analysis_morph_predict.py`：形态 PC1-10 → 多输出岭回归 + 帽子矩阵闭式 LOO，14,857 基因一次算完；KRT5 全基因组第 5（R²=0.215）；**top200 可预测基因 ∩ DE 显著 = 121/200，Fisher OR=17.4 p=1.1e-76**（形态≅分子的独立证据）
  - `analysis_cox.py`：单因素 Cox（分期 HR=1.36 p=0.057；"亚型 HR=0.32 p<0.001" 是缺失假象已标注）→ OOF C-index：临床 0.575 / 形态 0.493 / 联合 0.505 → 风险分组 KM p=0.93——**诚实的阴性结果**（教学价值：严谨的"没信号" vs 缺失陷阱的"假显著"）
- **泄漏的实锤教训**：MC 时代岭回归 NKX2-1 R²=+0.11，严格零泄漏后 R²=-0.09——正 R² 有泄漏水分；相关方向不变（KRT5 r=+0.46 vs +0.58）但幅度下降
- 坑：lifelines `predict_partial_hazard` 强惩罚下 exp 下溢 → log(0) 崩 KM；改 `predict_log_partial_hazard` 直接拿线性预测子
- 坑：`Stage IIIx` 会被 `startswith('Stage II')` 误吞——分期解析必须先判 IV→III→II→I 的顺序

## 23. 150 张时代收官（2026-10-05）
- **扩容下载**：`download_tcga.py` 加"kept+fill"采样（已在磁盘的切片永远保留进数据集，不够的份额再从剩余候选抽）——100→150 只增不换，老切片全部保留；GDC 晚高峰直连限速 ~20KB/s，脚本加 `GDC_PROXY` 环境变量走本地代理后 6MB/s
- **数据定格**：150 张 svs（75+75，137.9GB，144 患者）｜842 万 patch｜ResNet50 特征 69.1GB｜clinical 144 例｜expression 167 文件
- **特征提取提速**：双进程各啃一半 todo csv（进程级并行避开共享内存 1455 风险），VRAM 2×~2.2GB 在 6GB 卡内安全，合计约 2.8 分钟/张
- **双协议训练**：MC150（exp `luad_lusc_CLAM_sb`，5 折 11:55→15:0x）+ strict150（exp `luad_lusc_CLAM_sb_strict --split_dir task_2_strict150`，5 折 20:28→22:30，中间跨过一次会话重启无损接力）
- **坑 #13**：MC150 fold 3 启动 45 秒后静默消失（Windows 提交内存抖动），循环继续跑完 fold 4，靠"checkpoint 只在折末保存"的守卫发现缺失；补跑直接重发同一循环（幂等跳过已完成折），教训：后台连跑时 `tail -6` 会把报错截没，改 `tail -40`
- **坑 #12（eval.py 三机关）**：`--save_exp_code` 会被自动加 `EVAL_` 前缀（传 `EVAL_xxx` 会得到 `EVAL_EVAL_xxx`）；`--splits_dir` 断言 `isdir`，要给 `splits/task_2_strict150` 这种含 splits/ 的路径而非目录名；eval.py **没有** `--subtyping` 参数（那是 main.py 的），多传直接 argparse 报错
- **双评估成绩**：MC150 **0.788 ± 0.181**（0.918/0.968/0.551/0.643/0.857，fold 2/3 崩塌）｜strict150 **0.834 ± 0.090**（0.855/0.788/0.936/0.705/0.889）。对照 100 时代 MC 0.919 / strict 0.846——**同一模型同一批数据，仅评估协议不同结论就从 0.92 晃到 0.79**；严格折在两个规模下都稳在 0.83~0.85，是可信水平
- **五项分析全部在 strict150 零泄漏嵌入上重跑**（150/150 张嵌入全部来自各自 test 折模型，100 时代的"借用折"彻底消失）：
  - survival：分期 KM I vs III p=0.26；缺失陷阱演示 LUAD 1.2y vs LUSC 3.0y p<0.001（26 例 Alive LUAD 无随访的假象）
  - multiomics：marker 体检 p 最低 3e-20；形态 PC×marker |r|=0.32~0.35（比 100 时代弱，更大更杂队列的正常稀释）；岭回归 R²=-0.09（零泄漏口径与 strict100 一致）
  - de：19,938 蛋白编码基因 → **1,526 显著**（FDR<0.05 且 |log2FC|>1，LUAD 高 638 / LUSC 高 888）；Enrichr 富集方向符合生物学常识（LUSC 侧 p53/KRAS Dn/Apical Junction）
  - morph_predict：14,857 基因 hat 矩阵闭式 LOO；R² 前五 CACFD1 0.240/GLOD5 0.238/SERPINB5 0.232/DSC3 0.204/BLM 0.203；**Top200 ∩ DE = 141/200，Fisher OR=23.5 p=5.0e-95**（比 100 时代 OR=17.4 更强）
  - cox：分期 HR=1.29 p=0.084；OOF C-index 临床 0.561/形态 0.500/联合 0.517；风险分层 KM p=0.92——再次确认**分型特征 ≠ 预后特征**
- **热图**：MC150 fold-4 模型对 splits_4.csv 测试集出 4 张（2 LUAD + 2 LUSC）→ `results/heatmaps/`
- **文档**：notebook 35→42 cells（新增严格划分、§8b DE、§8c 形态预测、§9b Cox，15 张新图全部嵌入），README 重写为 150 口径，100 时代产物全部归档（`*_mc100`/`*_strict100`/`survival_100`）

## 24. 2026-10 审计勘误（2026-10-09）

审计只读代码、仓库里已提交的输入和输出，没有重新下载数据、没有重跑真实分析。下面更正前文的说法，前文原样保留作记录。

**前期流程**
- 第 0 节名词与 §5 的"ResNet50 在 1400 万张图上预训练"有误：CLAM 用的是 torchvision 的 ImageNet-1k 权重（`resnet50.tv_in1k`，约 128 万张、1000 类）；新版 CLAM 默认先把 256 像素的 patch 缩到 224 再提特征。
- 切块在 level 0 做，没有统一倍率：TCGA 切片混有 40×（约 0.25 µm/像素）和 20×（约 0.5 µm/像素），同样 256 像素覆盖面积差 4 倍。新增 `scripts/check_slide_mpp.py` 查分布。
- 43 个组织来源中心（TSS）每个只贡献一种亚型；严格 5 折下 138/150 张测试切片的中心在训练集出现过。新增 `make_strict_splits.py --group-by site`（Howard et al. 2021 Nat Commun 的 site-preserved CV）和 `eval_cv_summary.py`。
- 严格 5 折每折 AUC 均值 0.834，但池化 AUC 0.783，患者级 AUC 0.775（bootstrap 95% CI 0.70–0.85）；阈值 0.5 的准确率 0.713，概率偏向 LUSC。
- 用 sklearn 1.9.1、同一 seed 重新跑 `make_strict_splits.py`，得到的划分和本项目的不一致——划分文件需要入库。
- `make_heatmap.py` 把 `Y_prob`（已是 softmax 概率）又做了一次 softmax，标题概率被压向 0.5。4 张示例热图里 TCGA-98-A53H 是 LUSC 却被判为 LUAD（p=0.504）。
- §18 / notebook §3d 的特征 PCA 只用了 1 张 LUAD + 1 张 LUSC，不能说明"特征带癌种信息"。

**多组学（§19、§22、§23）**
- 167 个 RNA-seq 文件对应 144 个患者（原发 01 / 癌旁正常 11 / 复发 02），脚本按患者覆盖写入，可能用了癌旁正常样本。`download_tcga.py` 现在只下原发肿瘤并保存 `rna_files.csv`。
- 嵌入"零泄漏"只解决了测试集问题，却引入了新问题：每张切片用自己测试折的模型提嵌入，5 个模型的空间不同。v1 嵌入的 PC2–PC5 有 81–97% 方差可由折号解释（`scripts/audit_v1_embedding_folds.py`）。
- §22"泄漏的实锤教训"与 §19 自相矛盾：§19 的干净子集 R² 反而从 0.11 升到 0.17。R² 从 +0.11 变成 −0.09 来自嵌入来源和样本的变化，不是泄漏被去掉。
- "Top200 ∩ DE（OR 17.4 / 23.5）是形态≅分子的独立证据"是循环论证：嵌入为分 LUAD/LUSC 训练，自然能预测亚型差异基因。
- DE 的 log2FC 用算术均值计算，会被极端样本拉大（SST log2FC 5.8 而 q=0.29）；Enrichr 只送 top 200、背景是全基因组；"LUSC 侧 p53/KRAS Dn/Apical Junction 符合生物学常识"说过头了。

**生存（§20、§22、§23）**
- "26 例 Alive LUAD 无随访"不是 GDC 数据的非随机缺失，而是下载脚本请求临床数据时没 expand `follow_ups`：150 张时代是 44 例活着的 LUAD 没有随访，留下的 26 例 LUAD 全是死亡病例。KM、Cox、"诚实阴性"的 C-index 全部作废。
- `gender` 已改名 `sex_at_birth`；分期应取原发诊断（本数据集只影响 1 人）。

**玩具版 notebook（§9、§11）**
- TITAN 玩具没有位置编码，被遮 patch 无法利用邻居，重建误差停在约 1.07（只能猜整张切片的平均）；加 1D ALiBi 后 0.29。提取指纹前缺 `model.eval()`。真实 TITAN 切片编码器约 4850 万参数（不是"亿级"），第一阶段是 iBOT 自蒸馏。
- CLAM 玩具原来只报训练准确率；加独立考试卷后，1% 肿瘤占比时注意力在考试卷上接近瞎猜（训练准确率 86.5% 是记住了训练集）。
- UNI2 玩具：MLP 参数 259（不是"约 200"）；iBOT 预测老师网络的分布而非像素；UNI2-h 模型卡没写器官数和增强细节。
- CONCH 玩具：Linear(4→2) 是 10 个参数；损失下限 log(600)。

改动全部在 2026-10-09 的提交里：`scripts/common.py`、`embed_slides.py`、`eval_cv_summary.py`、`check_slide_mpp.py`、`audit_v1_embedding_folds.py` 为新增，五个 analysis 脚本重写为 v2，v1 产物移到 `results/archive_v1/`。v2 脚本在合成数据上跑通，真实数据需在本机重跑。

## 25. v2 真实数据重跑 + 按中心分组重训（2026-10-10）

审计 TODO 在数据机上的执行记录。

**v2 多组学 / 生存（真实数据，已完成）**
- `download_tcga.py --metadata-only`：重取带 follow_ups 的 clinical.json（144/144 有随访，v1 的"44 个活着的 LUAD 没有随访"确认是下载脚本漏字段）；生成 rna_files.csv（167 个文件 = 150 原发肿瘤 + 16 癌旁正常 + 1 复发）；从 GDC 托管的 xlsx 转出 TCGA-CDR.csv（Xena 403，改走 GDC API 直取）。
- `embed_slides.py`：meanpool + 5 折 CLAM 模型两套嵌入（slide/patient 两级共 12 个 csv）。教训：管道接 tail 会吞掉非零退出码，后台任务第一次"成功"实际死在 import timm；改重定向日志 + `echo exit=$?`。embed_slides 依赖 CLAM 的 topk/models，用 clam_latest 的 python 跑（DP venv 里 uv 装 timm 遇网络重置）。
- 多组学 v2：144 个原发肿瘤表达谱，DE 1,366 显著（LUAD 高 592 / LUSC 高 774），阳性对照通过（角化 q=3.1e-26 / 表面活性物质 q=3.2e-08）。marker 总体相关 BH 显著但亚型内大多消失；中心解释 PC 方差 30–58% vs 亚型 0–19%；6 个 marker 的 ΔR² 全 ≤ 0。全基因组 ΔR²：769 个基因超置换零分布 99% 分位（+0.0488，随机预期 ~149），头部全是 T/NK 标志（IL18RAP 0.169 等），CYT 形态 R²=+0.12 vs 亚型 −0.01。
- 生存 v2：TCGA-CDR 终点 142/144 可用，两亚型缺失率各 1.4%。亚型 KM p=0.612（v1 的 p<0.001 是 bug 倒影）；分期 KM p=0.128；Cox（139 例 58 事件，亚型分层，20×5 重复 CV）临床池化 C-index 0.617，加形态 0.588，ΔC=−0.029（bootstrap 95% [−0.100, +0.028]）；唯一显著协变量男性 HR=1.847 p=0.035。

**按中心分组 5 折重训（TODO 第 5 步）**
- 划分：`make_strict_splits.py --group-by site`（task_2_site150）；训练 exp `luad_lusc_CLAM_sb_site`。
- 教训一：两个 GPU 任务并发（embed + 训练）把 Windows commit 限额打爆，fold 0/1 段错误（exit 139）——且 CLAM 在验证集改善时训练中途就存 checkpoint，"有 checkpoint"≠"训练完成"（fold 0 只跑到 epoch 13）。真正的完成标志是 `split_i_results.pkl`。
- 教训二（exit 139 根因深挖）：串行之后 fold 0 又在 epoch 14 准时崩了一次。实测 commit 限额 28.1GB 只剩 2.6GB 空闲，而训练进程自己 commit 了 8.1GB——大头是 **WDDM 下显存占用兑现为 commit charge**：bag 大小悬殊（7k~98k patch），CUDA 缓存分配器的保留块随 epoch 数碎片化增长，直到系统无 commit 可给。修复：`core_utils.py` 的 train/validate 循环每 epoch 结束 `torch.cuda.empty_cache()`（CLAM 的 `torch.load` 此前已打过 `mmap=True` 补丁，所以 bag 读入不占 commit，问题只剩显存侧）。加补丁后重跑通过。同机教训汇总：GPU 任务串行 + 每 epoch empty_cache + torch.load mmap + workers≤1。
- 结果（带补丁重跑 fold 0/1 各 52 epoch、early stopping 正常触发后评估）：每折 AUC 0.738 / 0.640 / 0.582 / 0.764 / 0.778（0.700±0.085）；池化切片级 **0.647**；患者级 **0.647 [0.553, 0.736]**。对比按患者分组（0.783 / 0.775 [0.70, 0.85]）：**约 0.13 个 AUC 来自"认医院"**，剩余 0.647 仍高于随机（CI 下限 0.55）——形态有真实的、不依赖中心的亚型信号，但账面数字虚高明显。

## 26. 第一轮结果复核（2026-10-10）

只读第一轮推上来的结果文件（fold_*.csv、slide_mpp.csv、rna_files.csv、clinical.json、TCGA-CDR.csv、各 run_log）做复核，没有重跑训练或表达分析。

**确认没问题的**
- 补全 follow_ups 后，v2 从 GDC 记录算出的总生存与 TCGA-CDR 在 142 个共同患者上完全一致（事件和天数一个不差），缺失的 2 人两边相同。v1 的生存问题确实只是下载时漏字段。
- 167 个 RNA 文件 = 150 原发 + 16 癌旁正常 + 1 复发。按 Windows 上 glob 的读取顺序推算，v1 给 11 个患者（7 LUAD / 4 LUSC）用的是癌旁正常样本。
- 新热图标题的概率与 `results/eval_strict150/fold_*.csv` 一致，且都用了切片所在测试折的模型。

**需要更正或补充的**
- §25"约 0.13 个 AUC 来自认医院"说过头了。同一批患者上配对 bootstrap，按患者分组比按中心分组高 0.128（95% CI 0.04–0.22，只对患者重抽样）。
  这个差距同时包含"靠认医院拿分"和"新医院染色不同、泛化变差"两种成分，这个实验分不开。
- **倍率捷径**：`slide_mpp.csv` 里 8 张 ~20× 切片有 7 张是 LUSC。两种交叉验证下，8 张全被判成 LUSC，唯一一张 20× 的 LUAD（TCGA-75-7030）得到 p(LUSC)=0.94。
  按中心分组时，20× 的 LUSC 判对 86%，40× 的 LUSC 只有 53%。去掉这 8 张，严格折 / 中心折的患者级 AUC 为 0.768 / 0.631，只比全体低 0.01–0.02。
  §25 选的两张 LUSC 示例热图（TCGA-60-2723、TCGA-60-2726）恰好都是 20× 切片。
- **免疫信号还没过中心混杂检验**：v2 的 morph_predict 用随机分折 + 亚型内打乱，而 meanpool 主成分 30–58% 的方差可由中心解释。
  合成数据实测：40 个只受中心影响的基因被这个检验全部判为"超过零分布"；改用中心内打乱后仍有 17 个漏网，原因是全基因组共用一个阈值，
  而受中心影响的基因自己的零分布本来就高。改为"每个基因的实际 ΔR² 减去它自己的置换均值"后，降到 2/40，全基因组阳性约 1%（与随机预期一致）。
- **早停与 checkpoint**：§23 和 notebook §5.4 的"checkpoint 只在折末保存，有 checkpoint = 这折完成"不对。开了 `--early_stopping` 时 CLAM 在验证集 loss 创新低时就写 checkpoint（§25 已发现）。
  所以 MC150 和 strict150 中如果有折曾中途崩溃又被守卫跳过，用的就是没训练完的模型——需要在数据机上检查每折是否都有 `split_{k}_results.pkl`。
- `eval_cv_summary.py` 在按中心分组的结果上会打印"没见过的中心样本太少…需按中心分组重训"，自相矛盾；已改为按情况输出。
- `results/archive_v1/heatmaps_strict150/` 里其实是 MC150 fold-4 模型的旧图，已改名为 `heatmaps_mc150/`。
- notebook 的 8d、6b 两格之前有输出但没执行过（execution_count 为空），输出是日志内容。已改成"默认不重跑、只读日志"，代码与输出一致。

**脚本改动**
- `eval_cv_summary.py`：按倍率拆开报告、去掉 20× 后的 AUC、`--compare` 两套交叉验证的配对 bootstrap。
- `analysis_morph_predict.py`：`--cv site`、`--null site`，判定改为每个基因和自己的零分布比；产物按设置加后缀。
- `analysis_multiomics.py`、`analysis_cox.py`：`--embedding clam` 的产物加 `_clam` 后缀，不再覆盖默认结果。
- 以上改动在合成数据上跑通；真实数据上的第二轮待办见 `TODO.md`。
