# Dense PointMap 点云补全：网络设计与训练流程

本文档描述 **Dense PointMap** 方法的神经网络设计与训练流程，并与 **PointSea**（深度图自投影）做对照。

---

## 1. 方法总览

两者的**主干网络完全相同**（PointNet++ 点编码器 + 缩放 ResNet 图像编码器 + 注意力融合 + 两级 SDG 细化器），
**唯一区别在于自投影（self-projection）的输入表示**：

| | PointSea | Dense PointMap |
|---|---|---|
| 投影产物 | 深度图 `[B,3,H,W]`（3 通道均为同一深度值） | 点坐标图 `[B,3,H,W]`（x,y,z 各不相同） |
| 第 4 通道 | 恒为 1 的占位通道 | 原始/扩充点 mask（1=原始点，0=扩充点） |
| 稀疏性处理 | `points2grid` + `Grid2Image`（MaxPool3d+Gaussian 稠密化） | 高斯混合扩充 / PCA 切平面扩充 |
| 信息量 | 仅深度（丢失 x,y 位置） | 完整 3D 坐标 + mask |

---

## 2. 网络结构（Mermaid）

```mermaid
flowchart TB
    subgraph IN["输入"]
        PC["部分点云 partial<br/>[B, N, 3]  (N=2048)"]
    end

    subgraph PROJ["自投影 self-projection"]
        EXP["高斯混合扩充<br/>以每个原始点为中心、固定半径高斯球<br/>采样 M≈H×W 个扩充点"]
        ALL["原始点 + 扩充点<br/>[B, N+M, 3]"]
        VIEW["3 视角旋转/平移"]
        GRID["投影到 H×W 网格<br/>每个像素存 (x,y,z)"]
        MASK["mask 通道<br/>1=原始点  0=扩充点"]
        PM["pointmap [B·3, 4, H, W]<br/>(x, y, z, mask)"]
        EXP --> ALL --> VIEW --> GRID --> PM
        MASK --> PM
    end

    subgraph ENC["编码器"]
        IMG["缩放 ResNet18 (scale=0.5, conv1 输入 4ch)<br/>输出 [B·3, 256, 7, 7]"]
        PT["PointNet++ 编码器<br/>SA(512)→SA(128)→SA(all)<br/>输出 f_p [B, 128, 1]"]
        PM --> IMG
        PC --> PT
    end

    subgraph FUSE["SVFNet 多视角融合"]
        VA1["viewattn1: 3 视角图像特征 + 点特征<br/>self-attention → max over views"]
        VA2["viewattn2: 视角特征 + 点特征 + 视角位置编码"]
        FG["f_g = [f_p ; f_v]  [B, 256, 1]"]
        IMG --> VA1
        PT --> VA1
        VA1 --> VA2 --> FG
    end

    subgraph DEC["两级细化 (SDG / SDG_l)"]
        PS["ConvTranspose + sa 注意力<br/>生成 coarse 点云 [B, 256, 3]"]
        LE["local_encoder (EdgeConv)<br/>局部特征 832 维"]
        R1["refine1 (SDG, ratio=step1=4)<br/>结构分析 + 相似性对齐 + 路径选择<br/>coarse → fine1"]
        R2["refine2 (SDG_l, ratio=step2=8)<br/>fine1 → fine2"]
        FG --> PS
        PC --> LE
        PS --> R1
        LE --> R1
        R1 --> R2
    end

    subgraph OUT["输出"]
        O1["coarse [B, 256, 3]"]
        O2["fine1  [B, 1024, 3]"]
        O3["fine2  [B, 8192, 3]"]
    end
    PS --> O1
    R1 --> O2
    R2 --> O3

    subgraph LOSS["损失 (原版 PointSea get_loss)"]
        GT["真值 gt [B, 2048, 3]"]
        L1["CD(coarse, fps(gt,256))"]
        L2["CD(fine1, fps(gt,1024))"]
        L3["CD(fine2, gt)"]
        TOT["loss = L1 + L2 + L3<br/>(三层对称 L1 Chamfer, pytorch3d)"]
        L1 --> TOT
        L2 --> TOT
        L3 --> TOT
    end
    O1 --> L1
    O2 --> L2
    O3 --> L3
```

---

## 3. 自投影细节：Dense PointMap

```mermaid
flowchart LR
    A["partial [B,N,3]"] --> B["3 视角旋转/平移<br/>[B·3, N, 3]"]
    B --> C["窗口归一化<br/>仅用原始点范围 → 稀疏点铺满画幅"]
    C --> D{"扩充方式"}
    D -->|"Gaussian (dense_pointmap)"| E1["以每点为中心固定半径高斯球<br/>随机采样 M 个点"]
    D -->|"PCA 切平面 (geom_pointmap)"| E2["kNN 局部 PCA → 切平面 (t1,t2)+法向 n<br/>确定性同心环采样"]
    E1 --> F["原始点 + 扩充点"]
    E2 --> F
    F --> G["投影到 H×W<br/>z-buffer 取每像素一个点<br/>(原始点优先)"]
    G --> H["pointmap [B·3, 3, H, W]"]
    G --> I["mask [B·3, 1, H, W]"]
    H --> J["concat → [B·3, 4, H, W]"]
    I --> J
```

**稀疏性缓解**：原始 2048 点在 224×224 上仅约 0.5% 像素被占据；扩充到 M≈H×W 后填充率提升到 **35–45%**，
使 ResNet 能在大范围像素上做卷积。

---

## 4. 与 PointSea 的差异对照

```mermaid
flowchart TB
    PC["partial [B,N,3]"]

    subgraph PS["PointSea (深度图)"]
        P1["3 视角变换"]
        P2["points2grid: scatter z 到 3D 网格 [B·3,8,224,224]"]
        P3["Grid2Image: MaxPool3d(7×7) + Gaussian Conv3d + max(depth)"]
        P4["深度图 [B·3, 3, 224, 224] (3 通道相同)"]
        P5["补 1 通道 → [B·3, 4, 224, 224]"]
        P1-->P2-->P3-->P4-->P5
    end

    subgraph DP["Dense PointMap (点坐标图)"]
        D1["3 视角变换"]
        D2["高斯/PCA 扩充点"]
        D3["投影: 每像素存 (x,y,z)"]
        D4["pointmap [B·3, 3, 224, 224] (通道各异)"]
        D5["+ mask 通道 → [B·3, 4, 224, 224]"]
        D1-->D2-->D3-->D4-->D5
    end

    PC-->P1
    PC-->D1
    P5-->RN["相同的缩放 ResNet18 + PointNet++ + SDG/SDG_l"]
    D5-->RN
    RN-->OUT["coarse / fine1 / fine2"]
```

**核心区别**：
- **PointSea** 把点云压成**深度标量**（每像素只有 1 个深度值，且 3 通道冗余），丢失了像素内点的 (x,y) 信息；
- **Dense PointMap** 保留**完整 3D 坐标**（每像素 (x,y,z) 各异），并用 **mask** 区分原始点与扩充点。

---

## 5. 训练流程（Mermaid 时序）

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant P as 投影
    participant M as Model
    participant L as Loss
    participant OPT as Optimizer

    loop 每个 epoch (30)
        loop 每个 batch (bz=12)
            DL->>P: partial [B,2048,3]
            P->>M: pointmap+mask [B·3,4,224,224]
            M->>M: ResNet(图像) + PointNet++(点) + 融合 + SDG 两级
            M->>L: (coarse, fine1, fine2)
            L->>L: 三层对称 L1 CD (pytorch3d) vs 下采样 GT
            L-->>OPT: backward + clip_grad_norm(10) + step
            Note over L: NaN/Inf 输出/梯度/权重检测并跳过
        end
        M->>M: eval 模式跑 val(800) → CD/DCD/F1
        Note over M: 保存 best val CD 的 checkpoint
    end
```

**训练超参**（当前实验）：
- 模型：缩放版（5.0M 参数，全模型 64.1M 的 ~1/12）
- batch size 12，lr 1e-4 (Adam)，warmup 300 步
- `merge_points=256`（attention 序列减半以提速）
- 损失：`get_loss`（原版 Completion3D 协议，三层对称 L1 CD）
- 30 epoch，~10–12h/方法（GTX 1070）

---

## 6. 评测结果（val 800 样本，best checkpoint）

| 方法 | CD(×1000) ↓ | DCD ↓ | F1 ↑ | best epoch |
|------|-------------|-------|------|-----------|
| PointSea (深度图) | 26.99 | 0.725 | 0.212 | 11 |
| **Dense PointMap** | **18.72** | **0.677** | **0.318** | 25 |

Dense PointMap 的 CD 低 31%、F1 高 50%，且泛化更稳定（PointSea 第 11 epoch 后过拟合）。
