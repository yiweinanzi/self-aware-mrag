## 0. 先卡清楚 OK-VQA 的特点（跟 prompt 强相关）

- 约 1.4 万道开放式问题，专门要求**依赖外部知识**，知识类别包括 science & technology、history、sports 等十几个类。([CVF开放获取](https://openaccess.thecvf.com/content_CVPR_2019/papers/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.pdf?utm_source=chatgpt.com))
- 每个问题有 10 个人工答案，官方打分是 soft-accuracy（min(#same answers / 3, 1)），**短的英文名词短语最吃香**。([okvqa.allenai.org](https://okvqa.allenai.org/?utm_source=chatgpt.com))
- S3VQA 对 OK-VQA 又做了标注，大致分成四型：
   1）要先检测物体再查外部知识；2）纯 OCR；3）带主观色彩；4）其它。([s3vqa.github.io](https://s3vqa.github.io/?utm_source=chatgpt.com))

而目前 SOTA 的几条路子都指向同一个结论：

- 把图像变成**“问题相关”的文字描述 + （可选）几个 example**喂给 LLM（PICa、PromptCap、Simple Baseline）；([arXiv](https://arxiv.org/abs/2109.05014?utm_source=chatgpt.com))
- 或者在 prompt 里额外塞 **答案候选 + 答案示例（answer heuristics）**，让 LLM 在一个更“安全”的答案空间里选择（Prophet）。([arXiv](https://arxiv.org/abs/2303.01903?utm_source=chatgpt.com))

所以我们针对 OK-VQA 设计 prompt，核心就是围绕这两点做定制。

------

## 1. 针对 OK-VQA 的“总原则”

你可以把下面当成写 prompt 时的 checklist：

1. **输出约束（非常重要）：**
   - 只允许：**1～3 个英文单词的短语**（大部分官方答案就是这样）；
   - 全小写，不要标点，不要完整句子；
   - 禁止前后多余文字（“the answer is…”，“I think…”）。
2. **输入信息要“问题相关”而不是泛泛的 caption：**
    PromptCap、Simple Baseline 都证明了**问题引导的 caption**比普通 caption 好很多。([ACL Anthology](https://aclanthology.org/2023.emnlp-main.919.pdf?utm_source=chatgpt.com))
3. **尽可能暴露“知识类别”**：
    OK-VQA 本身有类别信息（science & technology, history, sports …），你在文本里显式告诉模型“这道题属于哪类知识”，能帮助它检索对的知识块。([CVF开放获取](https://openaccess.thecvf.com/content_CVPR_2019/papers/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.pdf?utm_source=chatgpt.com))
4. **利用你现有模型作为“答案启发器”**：
    这正好对应 Prophet 的思路——先用你自己的 VQA 模型出几个候选答案 / 例子，再 prompt 一个更强的 LLM 做最终决策。([arXiv](https://arxiv.org/abs/2303.01903?utm_source=chatgpt.com))
5. **不同题型使用轻微不同的子指令**：
   - 是非题 → 限制输出 yes / no；
   - 数字题 → 限制输出阿拉伯数字；
   - 主观 / opinion 题 → 引导回答“commonsense guess”。

------

## 2. 一套可以直接塞进代码的 Prompt 模板

### 2.1 最小的“OK-VQA 专用”模板（无 caption 版）

如果你暂时没有 question-guided caption 模型，可以先用这版对比你现在的 prompt：

```text
You are solving a visual question answering task on the OK-VQA dataset.
Each question requires both understanding of the image and external world knowledge.

Answer with a short English phrase only (1-3 words, all lowercase).
Do not output any explanation, punctuation, or extra words.

question: {QUESTION_TEXT}
answer:
```

中文版说明（不用喂给模型，只是给你看）：

- 强调“OK-VQA”“需要外部知识”，让模型知道这不是普通 VQA；
- 明确输出要求：short phrase + lowercase + no explanation。

你可以直接把你当前 prompt 换成这个，看看 accuracy 有多少提升（一般会有一点 gain）。

------

### 2.2 加入 Caption 的模板（PICa / Simple Baseline 风格）

当你能给每个 (image, question) 生成一两句**问题相关的 caption**（比如用 BLIP / PromptCap 一类的模型）时，推荐改用：

```text
You are a knowledgeable visual question answering system.
You are given a description of an image, a question about the image,
and you must use both the visual information and your world knowledge
to answer.

image_description:
{CAPTION_1}
{CAPTION_2}

question: {QUESTION_TEXT}

Answer with a short English phrase only (1-3 words, all lowercase).
Do not output explanation or punctuation.
answer:
```

- 这个结构基本就是 PICa + Simple Baseline 所用的思路：画像 → caption → LLM few-shot。([arXiv](https://arxiv.org/abs/2109.05014?utm_source=chatgpt.com))
- 如果你能做到 PromptCap 那样“用问题作为 prompt 引导 caption”，就更接近论文里的 SOTA 做法。([CVF开放获取](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_PromptCap_Prompt-Guided_Image_Captioning_for_VQA_with_GPT-3_ICCV_2023_paper.pdf?utm_source=chatgpt.com))

> 建议：先用通用 caption 模型做 baseline，再换成“question-guided caption”，测一下差多少，这个提升在文献里是挺显著的。([ACL Anthology](https://aclanthology.org/2023.emnlp-main.919.pdf?utm_source=chatgpt.com))

------

### 2.3 加入“知识类别”的模板

如果你手里有 OK-VQA 的 category（比如 official metadata 或自己训练了一个分类器），可以在 prompt 里加一个 domain 提示：

```text
You are solving a knowledge-based visual question answering task (OK-VQA).
The question belongs to the knowledge category: {CATEGORY}.

Use the image description, the question, and your world knowledge in this category
to answer.

category: {CATEGORY}  # e.g. "science and technology", "history", "sports"
image_description:
{CAPTION}

question: {QUESTION_TEXT}

Answer with a short English phrase only (1-3 words, lowercase), no explanation.
answer:
```

OK-VQA 论文里本来就按知识类别做过统计分析，所以这个信息对模型是有指导意义的。([CVF开放获取](https://openaccess.thecvf.com/content_CVPR_2019/papers/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.pdf?utm_source=chatgpt.com))

------

### 2.4 利用你自己的模型做“答案候选 + 示例”（Prophet 风格）

你现在已经有一个效果一般的模型，这正好可以被当成 Prophet 里的**stage-1 vanilla VQA**：从它身上挖“答案启发信息”，再交给（同一个 or 更大的）LLM 复查。([arXiv](https://arxiv.org/abs/2303.01903?utm_source=chatgpt.com))

假设你能从自己的模型拿到：

- top-k 候选答案：`cand_1 ... cand_k`
- 若干训练集中“相似问题”的 (caption, question, answer) 作为 few-shot 例子

可以设计第二阶段的 prompt：

```text
You are a knowledgeable assistant for knowledge-based visual question answering (OK-VQA).
Given an image description, a question, and some candidate answers,
please choose the best answer. If all candidates are wrong, output your own better answer.

Follow these rules strictly:
- Output one short English phrase only (1-3 words, lowercase).
- Do not output explanation or punctuation.
- If one of the candidates is correct, prefer that exact wording.

Here are some solved examples:

example 1:
image_description: {ex1_caption}
question: {ex1_question}
candidates: {ex1_cand1}, {ex1_cand2}, {ex1_cand3}
correct_answer: {ex1_answer}

example 2:
image_description: {ex2_caption}
question: {ex2_question}
candidates: {ex2_cand1}, {ex2_cand2}, {ex2_cand3}
correct_answer: {ex2_answer}

Now solve a new case:

image_description: {CAPTION}
question: {QUESTION_TEXT}
candidates: {CAND_1}, {CAND_2}, {CAND_3}, {CAND_4}

answer:
```

这就是 Prophet 的精髓：
 **你自己的模型 = 提供“候选答案 + 例题”的启发器**，LLM 负责用更强的世界知识和语言能力做最后裁决。实验证明，这个套路在 OK-VQA / A-OKVQA 上能把 GPT-3 推到 61% 左右的准确率。([CVF开放获取](https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Prompting_Large_Language_Models_With_Answer_Heuristics_for_Knowledge-Based_Visual_CVPR_2023_paper.pdf?utm_source=chatgpt.com))

你可以先只实现“候选答案 + prompt”，不放 examples，也会有一定提升。

------

### 2.5 In-context Few-shot 模板（Simple Baseline / PICa）

如果你打算做 few-shot（比如每个 prompt 放 4~8 个训练样本），可以用下面这种结构：

```text
You are a visual question answering system for the OK-VQA dataset.
For each example, you are given an image description and a question.
Answer with a short English phrase (1-3 words, lowercase).

example 1:
image_description: {ex1_caption}
question: {ex1_question}
answer: {ex1_answer}

example 2:
image_description: {ex2_caption}
question: {ex2_question}
answer: {ex2_answer}

example 3:
image_description: {ex3_caption}
question: {ex3_question}
answer: {ex3_answer}

Now answer the new question.

image_description: {CAPTION}
question: {QUESTION_TEXT}
answer:
```

Simple Baseline 就是类似结构，只是它会用一个 embedding 模型选那些“更相似”的样本做 in-context example，从而拿到 SOTA 水平。([arXiv](https://arxiv.org/html/2310.13570v1?utm_source=chatgpt.com))

------

## 3. 针对 OK-VQA 的“需求规范”：你可以写进代码 / 文档里的东西

### 3.1 输入需求（给 LLM 的文本）

建议在工程上定一个统一格式，比如：

```text
[task] ok-vqa
[category] {CATEGORY or unknown}
[type] {yes_no / number / other}
[image_description]
{CAPTION_1}
{CAPTION_2}

[question]
{QUESTION_TEXT}

[candidates]
{CAND_1}; {CAND_2}; {CAND_3}

[answer]
```

好处：

- 你内部 pipeline 很清晰，每个字段都可控；
- 后续如果想加 OCR 文本、检索到的知识文档，只要加新 section 即可：

```text
[ocr_text]
{OCR_TEXT}

[retrieved_knowledge]
{DOC_1_SNIPPET}
{DOC_2_SNIPPET}
```

### 3.2 输出需求（模型必须遵守的规则）

你可以在代码里写死这些 post-process 步骤：

1. 把生成结果做：
   - `lower()`；
   - 去掉首尾空格；
   - 去掉结尾句号、引号等；
2. 如果你识别到答案类型：
   - yes/no 题：只允许 `yes` 或 `no`，否则用一个简单规则强行映射；
   - 数字题：把 `"two"` / `"three"` 等用一个小字典转成 `2,3`；
3. 可选：根据 OK-VQA annotation 做一个**同义词表**
    比如把 `bike, bicycle` 归为同一类，这样即便你的输出和其中一个人类答案不完全字符相同，也可以被映射到标准答案上再交给 VQA eval（这一步有点工程量，但能抬不少 long tail 题的得分）。

------

## 4. 实验怎么安排比较“有信息量”

你现在的模型已经能跑 OK-VQA，可以用下面这个顺序做小 ablation：

1. **Baseline 0：你当前的 prompt（只改 post-process，让输出短 phrase）**
2. **Baseline 1：换成 §2.1 的“OK-VQA 专用无 caption 模板”**
3. **+Generic Caption：在 1 的基础上加普通 caption → §2.2 模板**
4. **+Question-guided Caption：换成 PromptCap / PNPVQA 一类任务导向 caption**([ACL Anthology](https://aclanthology.org/2023.emnlp-main.919.pdf?utm_source=chatgpt.com))
5. **+Category Token：在 prompt 里加 [category]**
6. **+Answer Candidates：把你模型的 top-k 候选写进 prompt（Prophet 简化版）**([arXiv](https://arxiv.org/abs/2303.01903?utm_source=chatgpt.com))

对每一档，你看：

- 总体 accuracy 的提升；
- 哪些知识类别 / 问题类型提得最多（比如 sports、history 有没有特别明显）。

这样你就能很清楚：是“caption 不够好”导致的，还是“没有用好候选答案”，再往里微调 prompt 细节。

