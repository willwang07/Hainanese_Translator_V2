# 海南话字词改写器 · A 方案 V1

默认保留原汉字。词典精确匹配时本地替换；未匹配表达可以通过一次 LLM 语义匹配选择已有词条；没有可靠匹配就保留原文。

这次更新聚焦翻译流程和失败处理。没有读音输出、整句生成、语法调整、verifier 或重试。词典不完整不会导致翻译失败。

## 更新已有项目

可以只用新版 `translator.py` 替换旧文件。它兼容原来的 `lexicon.json`，不需要重新填写 API key。保留您已填写的 `.env` 或 `config.env`，以及您自己扩充的词典。

完整压缩包内的 `lexicon.json` 已转为简单的 `source / target / gloss` 数组，共 83 项，原记录的目标汉字未改变。`dictionary.tsv` 保留最初的数据，译文不使用其中的读音列。

## 运行

在项目目录中执行：

```bash
python3 -m pip install -r requirements.txt
python3 translator.py "你为什么去学校？"
```

示例输出：

```text
汝做勿去学校？
```

这句话中的所有字词都可以本地匹配，API 调用次数为 0。“学校”可以按现有的“学”“校”两个词条保留，无须模型证明每个字的身份。

省略原句进入交互模式：

```bash
python3 translator.py
```

`:clear` 清除此前原句；`:reload` 重新读取词典；`:quit` 退出。

完全本地运行（不需要安装 OpenAI SDK）：

```bash
python3 translator.py "我今天去学校学习计算机" --local-only
python3 translator.py --demo
```

## 配置

项目目录已有可见的 `config.env`：

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5
```

将您的 key 填在第一行的等号后面即可。若同目录有 `.env`，优先读取 `.env`；已设置的环境变量优先于文件。`--model` 可以覆盖模型设置。

程序仅在需要语义匹配时加载 SDK 和建立客户端。SDK 未安装、API 不可用、超时、拒绝、响应未完成或 JSON 不合法时，仍输出本地替换加原文保留，退出状态为成功。普通输出只显示译文，失败回退时会在 stderr 给出简短提示。词典文件不存在或损坏、配置格式错误仍属于启动错误。

## 词典

编辑 `lexicon.json` 即可扩充词汇，无须改程序或维护 aliases：

```json
[
  {"source": "你", "target": "汝", "gloss": "you"},
  {"source": "为什么", "target": "做勿", "gloss": "why"},
  {"source": "什么", "target": "勿", "gloss": "what"}
]
```

`gloss` 可以用中文或英文说明意义。程序使用词典原文，不自行生成任何目标词。空数组也是有效词典，此时完全保留输入。

兼容旧词典时，只提取 `source_forms → source`、`han → target` 和 `sense → gloss`，忽略读音、kind 等旧字段；停用的旧词条不参与匹配。

现有数据已经存在“在”“妈妈”等相同 source、不同 target 的情况。程序将它们留在 unresolved 中，由同一次语义请求选择已有项；无法选择就保留原文。相同 source 的所有 target 一致时，仍可本地替换。这只是处理当前数据冲突，没有引入新的词典结构。

从带有“词义”“文字”表头的 TSV/CSV 导入简单词典：

```bash
python3 migrate_dictionary.py dictionary.tsv imported_dictionary.json
python3 translator.py "你为什么去？" --dictionary imported_dictionary.json
```

读音列可省略或留空。导入结果只含 `source / target / gloss`；完全相同的项合并，不同义项或目标写法保留。括号中的意义说明放入 gloss，斜线分隔的原始词义分别导入。导入输出使用新文件名，便于检查后再更新您的词典。

## 翻译流程与请求限制

1. 从左到右扫描原文，在当前位置优先匹配最长 source。生成 spans，锁定 exact 匹配。生成的目标文字不会再次进入匹配。
2. 将连续未匹配文字组成 unresolved spans。标点、空白独立保留。V1 不做会改变原文的 Unicode、空白或简繁规范化。
3. 仅在存在 unresolved 内容、有可选词条且启用语义匹配时，发送一次请求。包含整句、此前原句/固定语境、锁定匹配、未匹配 spans 和完整 compact dictionary。
4. 模型只返回已有词条的选择，不要求汇报每个词。它不重翻整句，也不修改锁定部分。不确定时返回空 matches。
5. 本地检查位置、词条引用和重叠后合并。个别无效 match 被忽略，不挡住其他合法 match；整个请求失败就使用 exact + passthrough。

每次提交最多一次语义请求，多个未匹配片段一起处理。SDK 设置 `max_retries=0`，程序也没有修复请求或第二次核对。输出 token 上限为 2048；默认 `gpt-5` 使用 `minimal` reasoning。其他模型不强行设置这个参数。

## 结构化匹配与 provenance

内部响应示例：

```json
{
  "matches": [
    {"span_id": 0, "source_span": "咋", "occurrence": 0, "entry_index": 1}
  ]
}
```

`span_id` 指定未匹配片段，`source_span` 是该片段的原文子串，`occurrence` 是从 0 开始的同一子串出现序号。`entry_index` 是本次发送的词典数组下标，用于区分同名词条，不需要手写 ID。目标文字始终由本地程序读取，模型返回的自由译文不会被采用。

查看内部记录：

```bash
python3 translator.py "你咋今天才来？" --json
python3 translator.py "我在学校" --context "这里的在表示位置。" --json
python3 translator.py --inspect 在
```

`provenance` 记录原文位置、原文、输出和 `exact / llm_semantic / passthrough` 方法。`statistics` 以原文非标点、非空白字符计数，包含 identity 匹配；这些比例是处理来源的占比，不是准确率。`semantic_attempts` 统计进入可选语义步骤的次数（0 或 1），SDK 无法加载时也算一次尝试。

可选地将保留的未知表达收集到文件，便于以后补词典：

```bash
python3 translator.py "你今天研究计算机" --gap-file passthrough.jsonl
```

收集不会自动扩充词典，不会把模型推测持久化成 aliases。

## V1 的已知边界

“怎么会 → 为什么”的例子有一个条件：词典里不能先锁定其中的“会”。当前词典包含“会 → 解”，所以“你怎么会今天才来”会先得到“你 → 汝”“会 → 解”；模型不能跨过这个锁定位置重写整个“怎么会”。程序遵守 V1 的 exact 锁定原则。

精确匹配是字符串匹配，没有中文分词或单义词的上下文复核。因此唯一 target 的多义 source 会直接替换，词典组件也可能在复合词中命中。这个版本实现字词改写；整句的自然度和这些匹配边界需要通过实际例句评估，后续再讨论是否调整锁定规则。

## 验证

```bash
python3 -m unittest -v test_translator
```

32 项本地测试覆盖最长匹配、原文保留、组件/identity 匹配、一次请求、锁定边界、重复与重叠匹配、词典冲突、旧格式兼容、provenance 统计，以及 SDK/API/JSON 失败回退。新增检查包含不填读音的导入、简单 JSON 导出，以及从其他工作目录启动时的 API 失败回退。模型响应使用模拟数据；未声称已完成真实 API 的海南话质量或耗时评测。

API 接入参考 [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[OpenAI Python SDK](https://github.com/openai/openai-python) 和 [GPT-5](https://developers.openai.com/api/docs/models/gpt-5) 官方说明。
