#!/usr/bin/env python3
"""Heuristic audit for Chinese technical-blog AI-writing patterns.

This tool reports observable patterns with locations. It does not estimate the
probability that text was written by AI and should not be used as a detector.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Finding:
    rule: str
    category: str
    priority: str
    line: int
    match: str
    advice: str


RULES = [
    (
        "empty-era-opener",
        "空转开头",
        "high",
        re.compile(r"在当今.{0,24}(?:时代|背景|浪潮|格局)|在.{0,16}日新月异的今天"),
        "改成明确问题、真实触发点或现有方法的缺口。",
    ),
    (
        "development-opener",
        "空转开头",
        "high",
        re.compile(r"随着.{0,32}(?:不断|快速|迅速|持续)(?:发展|演进|变化|普及)"),
        "说明具体发生了什么变化，以及它为何引出本文问题。",
    ),
    (
        "article-announcement",
        "元话语",
        "medium",
        re.compile(
            r"本文将(?:会)?(?:(?:全面|深入|系统|详细|重点)[、，和与及 ]*){0,3}"
            r"(?:地)?(?:探讨|介绍|分析|阐述|解读)"
        ),
        "直接进入问题；只有路线信息确有助于阅读时才保留。",
    ),
    (
        "empty-emphasis",
        "空泛强调",
        "medium",
        re.compile(r"值得注意的是|需要指出的是|毋庸置疑(?:的是)?|不可否认(?:的是)?|众所周知"),
        "删掉提示语并直接陈述事实，或补出真正需要强调的原因。",
    ),
    (
        "inflated-language",
        "空泛大词",
        "high",
        re.compile(r"赋能|颠覆性?|革命性|重塑|里程碑|开启.{0,10}新篇章|深刻改变|全方位|多维度|形成闭环|构建生态"),
        "落到具体动作、机制、约束、成本或结果。",
    ),
    (
        "vague-authority",
        "模糊归因",
        "high",
        re.compile(r"有研究表明|研究显示|专家认为|业内人士指出|业界普遍认为|普遍认为|数据显示"),
        "核对附近是否有可查来源；否则补来源、降低语气或删除归因。",
    ),
    (
        "binary-frame",
        "工整框架",
        "medium",
        re.compile(r"不仅.{0,48}(?:而且|更)|不是.{0,48}而是|一方面.{0,80}另一方面"),
        "确认对照是否真实必要；避免用对称句替代机制说明。",
    ),
    (
        "lazy-see-verb",
        "万能动词",
        "low",
        re.compile(
            r"(?:可以|能够|能|先|再|回头|接着)(?:看|看看|看一下)(?:到|出)?"
            r"|从[^，。！？；\n]{1,18}(?:来看|看)"
            r"|看[^，。！？；\n]{1,18}的(?:影响|效果|成本|开销|风险|差异|变化|问题|关系|趋势)"
        ),
        "确认“看”是否只是万能动词；按语义改成考虑、观察、估计、衡量、比较、分析或检查。",
    ),
    (
        "generic-tradeoff",
        "伪综合",
        "medium",
        re.compile(
            r"(?:不能|不应|不要|不宜)只(?:看|考虑|关注|追求|优化|比较)"
            r"[^，。！？；\n]{1,36}[，,；;]?[^。！？\n]{0,24}"
            r"(?:放在一起看|(?:综合考虑|兼顾|权衡)[^。！？；\n]{0,24}|trade[- ]?off)"
            r"|(?:不能|不应|不要|不宜)?(?:只)?(?:片面|单纯|一味)(?:地)?(?:追求|考虑|关注|优化)"
            r"[^，。！？；\n]{1,36}[，,；;]?[^。！？\n]{0,24}"
            r"(?:(?:综合考虑|兼顾|权衡)[^。！？；\n]{0,24}|trade[- ]?off)",
            re.I,
        ),
        "若确有取舍，写出指标、约束、方向和代价；否则删掉万能平衡句。",
    ),
    (
        "clarity-promise",
        "机械路标",
        "low",
        re.compile(r"把[^，。！？；：\n]{1,18}(?:说|讲)清楚(?![：:])"),
        "这不是禁句；确认后文是否立刻给出定义、条件、例子或边界，否则直接写具体问题。",
    ),
    (
        "single-judgment-frame",
        "元话语",
        "medium",
        re.compile(r"只(?:保留|留下)一个(?:判断|结论)(?!条件|分支|节点)"),
        "像编辑指令残留；若不是实验/算法约束，直接写那个判断本身或说明证据边界。",
    ),
    (
        "narrow-conclusion-frame",
        "审慎元话语",
        "medium",
        re.compile(
            r"(?:把|将)?(?:这个|上述|前面(?:的)?|原来(?:的)?)?结论"
            r"(?:进一步)?(?:收窄|缩窄|收束)(?:到|为)?"
            r"|(?:需要|应当|要|可以|最好|只能|先)?(?:进一步)?"
            r"(?:收窄|缩窄|收束)(?:一下)?(?:这个|上述|前面(?:的)?|原来(?:的)?)?结论"
        ),
        "直接写适用边界、注意事项或不能推出的内容；少写审稿式的“收窄结论”。",
    ),
    (
        "accounting-metaphor",
        "口语比喻",
        "medium",
        re.compile(r"(?:先|首先|下面|接下来)?(?:我们)?(?:来)?算(?:一笔)?[^，。！？；\n]{1,24}的账"),
        "若不是财务/预算语境，改成开销、成本、复杂度、显存占用或通信量计算。",
    ),
    (
        "preference-frame",
        "个人偏好",
        "low",
        re.compile(r"我(?:个人)?(?:会|还是)?(?:更)?倾向于|我个人会选择|在我看来"),
        "这不是禁句；确认偏好确属作者，并在附近给出评价标准、代价或适用条件。",
    ),
    (
        "generic-conclusion",
        "模板结尾",
        "high",
        re.compile(r"综上所述|总而言之|展望未来|未来可期|让我们拭目以待|这只是(?:一个)?开始|具有(?:十分|非常|极其)?重要(?:的)?意义"),
        "回到开头的问题，写具体结论、适用边界或未决实验。",
    ),
    (
        "chatbot-residue",
        "对话残留",
        "high",
        re.compile(r"这是一个(?:非常)?好的问题|希望这对你有所帮助|如果你愿意.{0,20}我(?:还)?可以|下面让我们一起|当然可以[！!]"),
        "删除客服式对话，直接写正文。",
    ),
    (
        "generic-heading",
        "模板结构",
        "low",
        re.compile(r"^#{1,6}\s*(?:背景介绍|核心优势|面临的挑战|挑战与机遇|未来展望|总结与展望)\s*$", re.M),
        "确认标题是否反映本节真实的推理动作或对象。",
    ),
    (
        "decorative-heading",
        "装饰排版",
        "medium",
        re.compile(r"^#{1,6}\s*[🚀✨💡✅🔥🎯📌📍]", re.M),
        "技术博客标题通常不需要 emoji 装饰。",
    ),
    (
        "bold-label-bullet",
        "装饰排版",
        "low",
        re.compile(r"^\s*[-*+]\s+\*\*[^*]{1,24}\*\*[：:]", re.M),
        "若不是清单或对照关系，考虑改成自然段。",
    ),
]


MASK_PATTERNS = [
    re.compile(r"^---\s*$.*?^---\s*$", re.M | re.S),
    re.compile(r"```.*?```", re.S),
    re.compile(r"~~~.*?~~~", re.S),
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\\\[.*?\\\]", re.S),
    re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.S),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"(?<!\$)\$(?!\$)[^$\n]+\$"),
    re.compile(r"https?://[^\s)>]+"),
]


def mask_preserving_newlines(text: str) -> str:
    masked = text
    for pattern in MASK_PATTERNS:
        masked = pattern.sub(
            lambda match: "".join("\n" if char == "\n" else " " for char in match.group(0)),
            masked,
        )
    return masked


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def compact(value: str, limit: int = 90) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def audit(text: str, max_findings: int) -> dict:
    prose = mask_preserving_newlines(text)
    findings: list[Finding] = []

    for rule, category, priority, pattern, advice in RULES:
        for match in pattern.finditer(prose):
            findings.append(
                Finding(
                    rule=rule,
                    category=category,
                    priority=priority,
                    line=line_number(prose, match.start()),
                    match=compact(text[match.start() : match.end()]),
                    advice=advice,
                )
            )
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break

    connector_pattern = re.compile(r"首先|其次|再次|最后|与此同时|值得注意的是")
    connectors = list(connector_pattern.finditer(prose))
    if len(connectors) >= 6 and len(findings) < max_findings:
        sample = "、".join(match.group(0) for match in connectors[:8])
        findings.append(
            Finding(
                rule="connector-density",
                category="机械路标",
                priority="medium",
                line=line_number(prose, connectors[0].start()),
                match=sample,
                advice="全文路标偏密；逐处改成真实的因果、递进、限制或时间关系。",
            )
        )

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", prose) if part.strip()]
    question_heavy = [part for part in paragraphs if part.count("？") >= 3]
    for paragraph in question_heavy[:3]:
        if len(findings) >= max_findings:
            break
        offset = prose.find(paragraph)
        findings.append(
            Finding(
                rule="question-pile",
                category="设问堆叠",
                priority="low",
                line=line_number(prose, offset),
                match=compact(paragraph),
                advice="确认每个问题都会在后文回答；纯粹造势的问题应合并或删除。",
            )
        )

    lines = prose.splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    bullet_lines = [line for line in nonempty_lines if re.match(r"^\s*[-*+]\s+", line)]
    heading_lines = [line for line in nonempty_lines if re.match(r"^\s*#{1,6}\s+", line)]
    sentences = [piece for piece in re.split(r"[。！？!?]+", prose) if piece.strip()]

    priority_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda item: (priority_order[item.priority], item.line, item.rule))

    return {
        "notice": "启发式审稿结果，不是 AI 检测概率。",
        "metrics": {
            "characters": len(text),
            "paragraphs": len(paragraphs),
            "sentences": len(sentences),
            "headings": len(heading_lines),
            "bullet_lines": len(bullet_lines),
            "findings": len(findings),
        },
        "findings": [asdict(item) for item in findings],
    }


def render_text(report: dict) -> str:
    metrics = report["metrics"]
    output = [
        report["notice"],
        (
            "字符 {characters}｜段落 {paragraphs}｜句子 {sentences}｜标题 {headings}｜"
            "项目符号行 {bullet_lines}｜命中 {findings}"
        ).format(**metrics),
    ]
    if not report["findings"]:
        output.append("未命中内置模式；仍需人工检查论证、事实和语境。")
        return "\n".join(output)

    for item in report["findings"]:
        output.append(
            "[{priority}] L{line} {category}/{rule}: {match}\n  建议：{advice}".format(**item)
        )
    return "\n".join(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="-", help="Markdown/text path, or - for stdin")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--max-findings", type=int, default=80, help="cap reported findings")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_findings < 1:
        print("--max-findings must be positive", file=sys.stderr)
        return 2

    if args.path == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.path).read_text(encoding="utf-8")

    report = audit(text, args.max_findings)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
