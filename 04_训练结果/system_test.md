# Flask 检测系统端到端测试

- 用例数：21；通过：21；失败：0
- 方式：Flask `test_client()` 进程内请求，临时数据库与临时上传目录，不污染项目数据
- 权重：`best.pt`

| # | 用例 | 请求 | 预期 | 实际 | 结果 |
|---:|---|---|---|---|---|
| 1 | 首页 HTML | `GET /` | 200 + HTML | 200（HTML 页面） | PASS |
| 2 | 历史页重定向 | `GET /history` | 302 -> /#/history | 302（Location=/#/history） | PASS |
| 3 | 模型信息接口 | `GET /api/model` | 200 + YOLO11s | 200（name=YOLO11s, mAP50=0.739, ok=True） | PASS |
| 4 | 示例列表 | `GET /api/samples` | 200 + 6 张 | 200（samples=6, ok=True） | PASS |
| 5 | 内置示例检测 | `POST /api/demo/crazing.jpg` | 200 + record_id | 200（record_id=1, status=fail, ok=True） | PASS |
| 6 | 示例不存在 | `POST /api/demo/nope.jpg` | 404 | 404（示例图不存在, ok=False, error=示例图不存在） | PASS |
| 7 | 单张图片检测 | `POST /api/predict (multipart)` | 200 + 判定 | 200（status=fail, count=3, 25.7 ms, ok=True） | PASS |
| 8 | JSON + base64 检测 | `POST /api/predict (json)` | 200 + 判定 | 200（status=fail, ok=True） | PASS |
| 9 | 自定义阈值生效 | `POST /api/predict conf=0.20 conf_high=0.50` | 200 + 参数回传 | 200（conf=0.2, conf_high=0.5, ok=True） | PASS |
| 10 | 缺少图片字段 | `POST /api/predict (empty)` | 400 | 400（没有选择图片, ok=False, error=没有选择图片） | PASS |
| 11 | 扩展名不支持 | `POST /api/predict x.txt` | 400 | 400（不支持的格式：.txt, ok=False, error=不支持的格式：.txt） | PASS |
| 12 | 图片文件损坏 | `POST /api/predict bad.jpg` | 400 | 400（图片解码失败，请换一张试试, ok=False, error=图片解码失败，请换一张试试） | PASS |
| 13 | 批量 2 张 | `POST /api/predict-batch (2)` | 200 + total=2 | 200（total=2, ok=True） | PASS |
| 14 | 批量超过上限 | `POST /api/predict-batch (21)` | 400 | 400（一次最多 20 张, ok=False, error=一次最多 20 张） | PASS |
| 15 | 历史记录列表 | `GET /api/history` | 200 + total>=1 | 200（total=6, ok=True） | PASS |
| 16 | 单条记录查询 | `GET /api/record/1` | 200 | 200（record_id=1, ok=True） | PASS |
| 17 | 记录不存在 | `GET /api/record/99999999` | 404 | 404（记录不存在, ok=False, error=记录不存在） | PASS |
| 18 | 检测报告页 | `GET /report/1` | 200 | 200（HTML 报告） | PASS |
| 19 | 报告不存在 | `GET /report/99999999` | 404 | 404 | PASS |
| 20 | 统计看板接口 | `GET /api/stats` | 200 + total>=1 | 200（total=6, pass_rate=0.0, ok=True） | PASS |
| 21 | 静态示例图 | `GET /static/samples/crazing.jpg` | 200 + image/jpeg | 200（image/jpeg） | PASS |

## 说明

- 第 7、8、9 条会真实调用模型推理，首次请求包含 predictor 初始化，耗时约 2–3 s；后续单张约 27 ms。
- 第 13、14 条测试批量上限，21 张在上传阶段就被拒绝，不会进入推理。
- 第 18 条报告页依赖第 5 或第 7 条生成的记录 id。