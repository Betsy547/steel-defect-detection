# 钢材缺陷检测实践报告｜参考文献清单（已逐条核对 DOI）

> 核对方式：每一条都通过 Crossref API 查过 **标题、作者、年份、期刊、卷期、页码/文章号、DOI**，不是凭印象写的。
> 核对日期：2026-11-23。

## 一、先看是否满足课程硬性要求

| 要求 | 本清单 | 结果 |
|---|---|---|
| 参考文献不少于 15 条 | 主推 19 条（另有 3 条备选） | ✅ |
| 2024—2027 年文献占 50% | 2024 年 5 条、2025 年 8 条、2026 年 1 条，共 14 条；14/19 = **73.7%** | ✅ |
| 尽量英文文献 | 19 条全部为英文期刊/会议论文 | ✅ |
| 真实数据集 | 第 [3] 条即 NEU-DET 原始论文 | ✅ |

> 说明：2027 年的论文现在还不存在（尚未到该年份），所以“2024—2027”实际能覆盖的是 **2024、2025、2026** 三年。即使你后面删掉 6 条近年文献，剩下的 8 条仍≥50%。

## 二、主推 19 条（按建议引用顺序）

| # | 文献（第一作者+年份） | 年份 | 期刊/会议 | 在报告里的用途 |
|---|---|---|---|---|
| 1 | Ameri et al. | 2024 | Engineering Applications of Artificial Intelligence | 工业表面缺陷检测深度学习方法**系统综述**——引言、相关工作 |
| 2 | Qiao et al. | 2025 | IEEE Access | 金属表面缺陷检测技术**综述**——工业背景、研究现状 |
| 3 | Song & Yan | 2013 | Applied Surface Science | **NEU-DET 数据集原始论文**——数据来源必须引 |
| 4 | Redmon et al. | 2016 | CVPR | YOLO 原始论文——单阶段检测原理 |
| 5 | Ren et al. | 2017 | IEEE TPAMI | Faster R-CNN——两阶段检测对照 |
| 6 | Lin et al. | 2014 | ECCV | COCO/mAP 评价指标——指标定义 |
| 7 | Lin et al. | 2017 | ICCV | Focal Loss——类别不平衡与难样本 |
| 8 | Luo et al. | 2024 | Procedia CIRP | Mobile-YOLO-SDD：轻量化实时钢材缺陷检测 |
| 9 | Wu et al. | 2024 | Metals | SDD-YOLO：条带钢表面缺陷轻量化、泛化性 |
| 10 | Zhao et al. | 2024 | Measurement Science and Technology | FCS-YOLO：高效钢材表面缺陷检测 |
| 11 | Xu et al. | 2024 | Robotic Intelligence and Automation | 多尺度信息提取：精度—算力—速度平衡 |
| 12 | Jiang et al. | 2025 | Measurement | FPDNet：热轧带钢低对比度/不规则缺陷 |
| 13 | Zhang | 2025 | Engineering Applications of Artificial Intelligence | 门控空间通道重建：轻量化框架 |
| 14 | Xie et al. | 2025 | Neurocomputing | 重参数化特征金字塔：YOLOv8 改进 |
| 15 | Zhou et al. | 2025 | Scientific Reports | 改进 YOLOv5 的高精度轻量化方法 |
| 16 | Wang et al. | 2025 | IEEE Access | CSCP-YOLO：实时钢材缺陷检测 |
| 17 | Dang & Wang | 2025 | IEEE Access | FD-YOLO11：YOLO11 特征增强 |
| 18 | Li et al. | 2025 | Measurement Science and Technology | DSP-YOLO：**直接对标你的 YOLO11 基线** |
| 19 | Lin et al. | 2026 | Scientific Reports | WSO-YOLO11：热轧带钢上的最新 YOLO11 工作 |

## 三、可直接粘贴的 GB/T 7714-2015 参考文献表

> 按“建议引用顺序”编号；若学校要求按作者字母排序，我可以重排。英文作者姓全大写、名缩写。

[1] AMERI R, HSU C C, BAND S S. A systematic review of deep learning approaches for surface defect detection in industrial applications[J]. Engineering Applications of Artificial Intelligence, 2024, 130: 107717. DOI:10.1016/j.engappai.2023.107717.

[2] QIAO Q, HU H, AHMAD A, et al. A review of metal surface defect detection technologies in industrial applications[J]. IEEE Access, 2025, 13: 48380-48400. DOI:10.1109/ACCESS.2025.3544578.

[3] SONG K, YAN Y. A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects[J]. Applied Surface Science, 2013, 285: 858-864. DOI:10.1016/j.apsusc.2013.09.002.

[4] REDMON J, DIVVALA S, GIRSHICK R, et al. You only look once: unified, real-time object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Las Vegas: IEEE, 2016: 779-788. DOI:10.1109/CVPR.2016.91.

[5] REN S, HE K, GIRSHICK R, et al. Faster R-CNN: towards real-time object detection with region proposal networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(6): 1137-1149. DOI:10.1109/TPAMI.2016.2577031.

[6] LIN T Y, MAIRE M, BELONGIE S, et al. Microsoft COCO: common objects in context[C]//Computer Vision – ECCV 2014. Cham: Springer, 2014: 740-755. DOI:10.1007/978-3-319-10602-1_48.

[7] LIN T Y, GOYAL P, GIRSHICK R, et al. Focal loss for dense object detection[C]//Proceedings of the IEEE International Conference on Computer Vision. Venice: IEEE, 2017: 2999-3007. DOI:10.1109/ICCV.2017.324.

[8] LUO S, XU Y, ZHU M, et al. Mobile-YOLO-SDD: a lightweight YOLO for real-time steel defect detection[J]. Procedia CIRP, 2024, 129: 228-233. DOI:10.1016/j.procir.2024.10.040.

[9] WU Y, CHEN R, LI Z, et al. SDD-YOLO: a lightweight, high-generalization methodology for real-time detection of strip surface defects[J]. Metals, 2024, 14(6): 650. DOI:10.3390/met14060650.

[10] ZHAO C, FAN Y, TAN J, et al. FCS-YOLO: an efficient algorithm for detecting steel surface defects[J]. Measurement Science and Technology, 2024, 35(8): 086004. DOI:10.1088/1361-6501/ad4410.

[11] XU W, ZHANG Y, JIANG X, et al. An efficient steel defect detection model based on multi-scale information extraction[J]. Robotic Intelligence and Automation, 2024, 44(6): 817-829. DOI:10.1108/RIA-03-2024-0065.

[12] JIANG F, LU K, WANG W, et al. FPDNet: a fast and high-precision detection network for hot-rolled strip surface defects[J]. Measurement, 2025, 253: 117309. DOI:10.1016/j.measurement.2025.117309.

[13] ZHANG W. Gate-guided spatial-channel reconstruction network: an efficient lightweight framework for steel surface defect detection[J]. Engineering Applications of Artificial Intelligence, 2025, 162: 112537. DOI:10.1016/j.engappai.2025.112537.

[14] XIE W, MA W, SUN X. An efficient re-parameterization feature pyramid network on YOLOv8 to the detection of steel surface defect[J]. Neurocomputing, 2025, 614: 128775. DOI:10.1016/j.neucom.2024.128775.

[15] ZHOU M, WANG H, WANG Y. A high precision and lightweight method for steel surface defect detection based on improved YOLOv5[J]. Scientific Reports, 2025, 15: 44492. DOI:10.1038/s41598-025-28022-5.

[16] WANG C, WANG H, JIANG Y, et al. CSCP-YOLO: a lightweight and efficient algorithm for real-time steel surface defect detection[J]. IEEE Access, 2025, 13: 113517-113528. DOI:10.1109/ACCESS.2025.3581024.

[17] DANG Z, WANG X. FD-YOLO11: a feature-enhanced deep learning model for steel surface defect detection[J]. IEEE Access, 2025, 13: 63981-63993. DOI:10.1109/ACCESS.2025.3559733.

[18] LI X, FAN Z, LIU Q, et al. DSP-YOLO: an improved YOLO11-based method for steel surface defect detection[J]. Measurement Science and Technology, 2025, 36(9): 096201. DOI:10.1088/1361-6501/adfc88.

[19] LIN J, CHEN L, GUO C, et al. WSO-YOLO11: a wavelet-strip-oriented YOLO11 framework for hot-rolled steel strip surface defect detection[J]. Scientific Reports, 2026, 16: 30129. DOI:10.1038/s41598-026-60355-7.

## 四、备选 3 条（需要凑数或加强某一部分时再加）

[20] JI W, LIU S, DENG L, et al. WDI-YOLO: a lightweight steel bridge weld defect detection algorithm using UAV images[J]. Journal of Constructional Steel Research, 2025, 235: 109833. DOI:10.1016/j.jcsr.2025.109833.

[21] SONG H. RSTD-YOLOv7: a steel surface defect detection based on improved YOLOv7[J]. Scientific Reports, 2025, 15: 19649. DOI:10.1038/s41598-025-04811-w.

[22] WANG C Y, YEH I H, LIAO H Y M. YOLOv9: learning what you want to learn using programmable gradient information[C]//Computer Vision – ECCV 2024. Cham: Springer, 2024: 1-21. DOI:10.1007/978-3-031-72751-1_1.

## 五、正文引用位置建议（只给位置，不给整段文字）

| 报告位置 | 建议引用 | 说明 |
|---|---|---|
| 1 背景与意义 | [1][2] | 工业表面缺陷检测的深度学习方法与金属场景综述，建立“为什么值得做” |
| 2 数据集 | [3] | NEU-DET 的原始出处，必须引；不要只写“网上下的” |
| 2 技术路线 | [4][5] | 单阶段 YOLO 与两阶段 Faster R-CNN 的取舍 |
| 3 评价指标 | [6] | mAP50 / mAP50-95 的定义与计算口径 |
| 3 难样本问题 | [7] | 类别不平衡、难样本对损失函数的影响 |
| 2/4 轻量化与实时性 | [8][9][16][17] | 与你“CPU 约 27 ms、19 MB 权重”的部署选择对话 |
| 4 多尺度与注意力 | [10][11][12][13][14][15] | 说明近年主流改进方向，反衬你“不堆模块、先做基线对比”的取舍 |
| 4 YOLO11 基线 | [17][18][19] | 直接对标你的 YOLO11s；注意不同论文的划分/增强/轮数不同 |
| 6 讨论与局限 | [8]—[19] | 用文献中的 NEU-DET 结果解释你的 0.739 处在什么区间，以及为什么不能直接横比 |

## 六、四条注意事项（与课程要求直接相关）

1. **“避免使用软件生成的文档”**：这份清单可以用于查文献、核对 DOI、安排引用位置；**正文段落请你自己写**。不要把我生成的整段文字直接粘进报告，查重平台会把“文档助手生成内容”算进去。
2. **引用要转述，不要抄摘要**：每篇文献用你自己的话概括“做了什么、在什么数据上、结论是什么”，再标注引用。直接抄英文摘要会同时抬高重复率和被查出 AI 生成的风险。
3. **代码许可说明**：仓库根目录**已添加 `LICENSE`（MIT）**，README 的“许可”一节写明了自写代码、依赖库（Ultralytics AGPL-3.0 / Flask BSD-3 / OpenCV Apache-2.0）、数据集与权重的边界。报告附录可直接写“代码采用 MIT 许可（见仓库 LICENSE），数据来自 NEU-DET（仅用于研究）”。
4. **数据提交**：报告里写清楚数据来源（NEU-DET 官方页面或你实际使用的下载链接），仓库里已有 `03_数据集/`，可以直接作为“数据随代码提交”的说明；如果学校要求单独数据包，附上仓库链接即可。

## 七、关于“你的 0.739 和文献里的 0.8+”怎么解释

近年论文在 NEU-DET 上常报告 mAP50 0.78–0.83，但它们的**划分方式、增强策略、训练轮数、输入尺寸、是否使用额外模块**都不同。你的实验条件是 `imgsz=320、150 epochs、batch=32、单 seed、YOLO11s 原版`，属于“干净基线”。报告里应写成：

> 本文不追求在 NEU-DET 上刷到最高 mAP，而是固定一套可复现的超参，比较 YOLOv8s / YOLO11s / YOLO26s 三种骨干，并据此做部署选型；与文献结果的差异主要来自实验协议不同，不能直接横向比较。

这句话既诚实，又能把“精度不是最高”转化为“实验设计有边界”的优点。
