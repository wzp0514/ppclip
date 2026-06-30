# ppclip

AI 辅助剪辑：把素材目录和一段想法描述交给它，自动生成剪映草稿。你在剪映里接着调。

```
素材目录 + 想法描述
  → 素材扫描 + 场景检测 + 画面分析
  → 生成分镜脚本
  → 匹配素材到分镜
  → 生成剪映草稿
```

## 安装

```bash
cd ppclip
pip install -r requirements.txt
```

依赖：Python 3.9+、FFmpeg、剪映专业版。

## 运行

```python
from ppclip import run

result = run(
    material_dir="D:/我的素材/",
    idea="一段描述，想要什么效果",
    tier="prod",          # dev | test | prod | full
)
```

`tier` 选 `prod` 体验最完整。API Key 配法见下文。

## 配置

需要两类模型：文字模型（生成脚本/匹配素材）和视觉模型（分析画面）。在 `~/.ppclip/config.json` 中填入 Key：

```json
{
  "api": {
    "api_key": "你的Key"
  }
}
```

不配 Key 也能跑 `dev` 档，不调任何模型。

## 降级

每一步都有自动降级，不会中断：

| 环节 | 正常 | 降级 | 保底 |
|------|------|------|------|
| 场景检测 | FFmpeg | PySceneDetect | 固定间隔切分 |
| 画面分析 | 视觉模型 | 本地分类 | 文件名推测 |
| 脚本生成 | 文字模型 | 文字模型降级 | 3 镜空模板 |
| 素材匹配 | 文字模型 | 文件名+时长 | 随机 |
| 草稿生成 | 全功能 | 素材+字幕 | 剪辑指导.md |

## License

MIT
