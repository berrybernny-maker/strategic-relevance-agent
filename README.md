# Strategic Relevance Agent

给“市场事件卡（Event Card）”打战略相关性分数，用于把海量公开情报路由到集团/业务/地区不同层级，并为下游预测与建议提供稳定的结构化输入。

## 功能

- 输入：企业战略意图（可通过 API 动态更新）+ 事件卡列表（来自你们的 MCP 清洗结构化输出）
- 输出（每个事件）：
  - relevance_score (0-1)：与战略意图的综合相关性
  - matched_intents[]：命中的战略标签
  - why_relevant：可解释的命中摘要
- 评分方式（可配置）：
  - keyword 命中（加权）
  - BM25（稀疏检索）
  - embeddings 语义相似度（可选；支持 OpenAI 兼容 embedding API）
  - 线性融合（或仅 sparse）

## 安装

```bash
cd strategic-relevance-agent
python -m pip install -e .
```

## 运行 API

```bash
python -m strategic_relevance.api --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

更新战略意图（示例见 `examples/intents.sample.json`）：

```bash
curl -X PUT http://localhost:8000/intents -H "Content-Type: application/json" -d @examples/intents.sample.json
```

对事件卡打分（示例见 `examples/events.sample.json`）：

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d @examples/score.request.sample.json
```

## 运行 CLI

```bash
sr-agent score --intents examples/intents.sample.json --events examples/events.sample.json --out examples/scored.out.json
```

## Embeddings 配置（可选）

默认只跑 sparse（keyword + BM25）。如果要启用 embeddings，设置环境变量：

- `SR_EMBEDDINGS_ENABLED=true`
- `SR_EMBEDDINGS_BASE_URL`：OpenAI 兼容 base url（例如 `https://api.openai.com/v1` 或你们的兼容服务）
- `SR_EMBEDDINGS_API_KEY`
- `SR_EMBEDDINGS_MODEL`：例如 `text-embedding-3-large`

## 事件卡输入约定

该服务对事件卡字段采取“宽松兼容”策略：

- 标识字段：存在 `event_id` / `id` / `canonical_event_key` / `primary_source_id` 任一即可
- 文本字段：建议提供 `summary`（或 `title/content/text/event_text/clean_text` 任一），否则只能用 `topic/strategic_vertical/region` 做弱回退，相关性会很不准
- 结构字段：`region/topic/strategic_vertical/importance_score/confidence_score` 等会透传用于 why_relevant 的解释补全

## License

MIT
