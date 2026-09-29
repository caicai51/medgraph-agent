"""
RAG评估体系

功能：
1. RAG系统端到端评估
2. 检索质量评估
3. 生成质量评估
4. 幻觉检测
5. 答案质量评分
"""

import os
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

# 中文停用词与标点过滤正则（匹配纯标点/空白/单字符无意义符号）
_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
# 高频无信息量词，避免 faithfulness 被"的/是/和"等拉偏
_STOPWORDS = {
    "的", "是", "和", "了", "在", "有", "与", "及", "或", "等", "为", "以", "于",
    "也", "都", "可", "可", "不", "无", "之", "其", "这", "那", "一", "个", "中",
    "对", "由", "从", "到", "但", "而", "则", "如", "若", "如", "把", "被", "让",
    "你", "我", "他", "她", "它", "们", "上", "下", "里", "外", "前", "后",
}


def _tokenize_zh(text: str) -> List[str]:
    """中文友好的分词：优先 jieba，回退到字符级；过滤标点/停用词/单字符。

    修复原 split() 对中文无空格、整句被当成一个 token 导致 faithfulness
    评测口径严重失真的问题。
    """
    if not text:
        return []
    text = text.lower()
    if _HAS_JIEBA:
        tokens = list(jieba.lcut(text))
    else:
        tokens = list(text)
    return [
        t for t in tokens
        if t and t.strip() and not _PUNCT_RE.match(t) and t not in _STOPWORDS and len(t.strip()) > 0
    ]


class EvaluationDimension(str, Enum):
    """评估维度"""
    RETRIEVAL_RECALL = "retrieval_recall"
    RETRIEVAL_PRECISION = "retrieval_precision"
    ANSWER_RELEVANCE = "answer_relevance"
    ANSWER_FAITHFULNESS = "answer_faithfulness"
    ANSWER_ACCURACY = "answer_accuracy"
    HALLUCINATION_RATE = "hallucination_rate"
    RESPONSE_TIME = "response_time"


@dataclass
class EvaluationResult:
    """评估结果"""
    dimension: EvaluationDimension
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGEvaluationCase:
    """RAG评估案例"""
    query: str
    ground_truth_answer: str
    ground_truth_contexts: List[str]
    retrieved_contexts: List[str] = field(default_factory=list)
    generated_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalEvaluator:
    """
    检索评估器

    评估指标：
    1. Recall@K
    2. Precision@K
    3. MRR (Mean Reciprocal Rank)
    4. NDCG@K
    """

    def __init__(self):
        self.results = []

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        评估检索质量

        Args:
            query: 查询文本
            retrieved_docs: 检索返回的文档列表
            relevant_doc_ids: 相关的文档ID列表
            k_values: K值列表

        Returns:
            评估结果
        """
        retrieved_ids = [doc.get("id", doc.get("chunk_id")) for doc in retrieved_docs]
        relevant_set = set(relevant_doc_ids)

        metrics = {}

        for k in k_values:
            # Recall@K
            retrieved_k = set(retrieved_ids[:k])
            recall = len(retrieved_k & relevant_set) / len(relevant_set) if relevant_set else 0.0
            metrics[f"recall@{k}"] = recall

            # Precision@K
            precision = len(retrieved_k & relevant_set) / k if k > 0 else 0.0
            metrics[f"precision@{k}"] = precision

            # NDCG@K
            ndcg = self._ndcg_at_k(retrieved_ids[:k], relevant_doc_ids, k)
            metrics[f"ndcg@{k}"] = ndcg

        # MRR
        metrics["mrr"] = self._mrr(retrieved_ids, relevant_doc_ids)

        return {
            "query": query,
            "metrics": metrics,
            "num_retrieved": len(retrieved_ids),
            "num_relevant": len(relevant_doc_ids)
        }

    def _ndcg_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """计算NDCG@K"""
        if not relevant:
            return 0.0

        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)

        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))

        return dcg / idcg if idcg > 0 else 0.0

    def _mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """计算MRR"""
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0


class AnswerEvaluator:
    """
    答案评估器

    评估维度：
    1. 答案相关性
    2. 答案忠实度（是否基于上下文）
    3. 答案准确性
    4. 幻觉率
    """

    def __init__(self, use_llm_judgment: bool = False):
        self.use_llm_judgment = use_llm_judgment

    def evaluate_answer(
        self,
        query: str,
        generated_answer: str,
        ground_truth: str,
        contexts: List[str],
        use_llm: bool = False
    ) -> Dict[str, Any]:
        """
        评估答案质量

        Args:
            query: 用户查询
            generated_answer: 生成的答案
            ground_truth: 标准答案
            contexts: 检索到的上下文
            use_llm: 是否使用LLM评估

        Returns:
            评估结果
        """
        metrics = {}

        # 1. 答案相关性（基于词重叠）
        metrics["relevance_score"] = self._calculate_relevance(
            generated_answer, query
        )

        # 2. 答案忠实度（检查答案是否基于上下文）
        if contexts:
            metrics["faithfulness_score"] = self._calculate_faithfulness(
                generated_answer, contexts
            )
        else:
            metrics["faithfulness_score"] = 0.0

        # 3. 答案准确性（与标准答案对比）
        metrics["accuracy_score"] = self._calculate_accuracy(
            generated_answer, ground_truth
        )

        # 4. 幻觉检测
        metrics["hallucination_score"] = self._calculate_hallucination(
            generated_answer, contexts
        )

        # LLM评估（可选）
        if use_llm:
            llm_metrics = self._llm_evaluate(
                query, generated_answer, ground_truth, contexts
            )
            metrics.update(llm_metrics)

        return {
            "metrics": metrics,
            "grades": self._assign_grades(metrics)
        }

    def _calculate_relevance(self, answer: str, query: str) -> float:
        """计算答案与查询的相关性"""
        answer_words = set(_tokenize_zh(answer))
        query_words = set(_tokenize_zh(query))

        if not query_words:
            return 0.0

        overlap = len(answer_words & query_words)
        return overlap / len(query_words)

    def _calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """计算答案对上下文的忠实度

        口径修复：原实现用 answer.split() 切词，对无空格的中文会把整句
        当成单个 token 再做子串匹配，LLM 稍作改写即判为不忠实，导致
        faithfulness 系统性偏低。改为 jieba 词级匹配后，只要答案的信息
        词能在大块上下文中找到即计为忠实，口径更合理。
        """
        if not contexts:
            return 0.0

        context_text = " ".join(contexts)
        context_tokens = set(_tokenize_zh(context_text))
        answer_tokens = _tokenize_zh(answer)

        if not answer_tokens:
            return 0.0

        # 词级匹配：答案词在上下文 token 集合中出现即计为忠实
        present_in_context = sum(1 for w in answer_tokens if w in context_tokens)

        return present_in_context / len(answer_tokens)

    def _calculate_accuracy(self, answer: str, ground_truth: str) -> float:
        """计算答案准确性"""
        if not ground_truth:
            return 0.0

        # 词级重叠（jieba 分词，适配中文）
        answer_words = set(_tokenize_zh(answer))
        truth_words = set(_tokenize_zh(ground_truth))

        overlap = len(answer_words & truth_words)
        total = len(truth_words)

        return overlap / total if total > 0 else 0.0

    def _calculate_hallucination(self, answer: str, contexts: List[str]) -> float:
        """
        估算幻觉率

        策略：答案信息词中不在上下文词集合中的比例越高，幻觉率越高。
        与 faithfulness 互补（hallucination ≈ 1 - faithfulness 的信息词口径）。
        """
        if not contexts:
            return 1.0  # 无上下文时假设高幻觉率

        context_text = " ".join(contexts)
        context_tokens = set(_tokenize_zh(context_text))
        answer_tokens = _tokenize_zh(answer)

        if not answer_tokens:
            return 0.0

        # 不在上下文词集合中的信息词
        out_of_context = sum(1 for w in answer_tokens if w not in context_tokens)

        # 幻觉率 = 不相关词比例
        hallucination_rate = out_of_context / len(answer_tokens)

        # 惩罚过长的答案
        if len(answer_tokens) > 50:
            hallucination_rate *= 1.2

        return min(hallucination_rate, 1.0)

    def _llm_evaluate(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        contexts: List[str]
    ) -> Dict[str, float]:
        """使用LLM进行评估"""
        # 这里可以接入阿里云API进行LLM评估
        # 简化实现返回空字典
        return {}

    def _assign_grades(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """根据指标分配等级"""
        grades = {}

        grade_thresholds = {
            "relevance_score": (0.6, 0.8, 0.9),
            "faithfulness_score": (0.7, 0.85, 0.95),
            "accuracy_score": (0.5, 0.7, 0.85),
            "hallucination_score": (0.2, 0.1, 0.05)  # 越低越好
        }

        for metric, threshold in grade_thresholds.items():
            score = metrics.get(metric, 0.0)
            if metric == "hallucination_score":
                # 幻觉率越低越好
                if score <= threshold[2]:
                    grades[metric] = "A"
                elif score <= threshold[1]:
                    grades[metric] = "B"
                elif score <= threshold[0]:
                    grades[metric] = "C"
                else:
                    grades[metric] = "D"
            else:
                # 其他指标越高越好
                if score >= threshold[2]:
                    grades[metric] = "A"
                elif score >= threshold[1]:
                    grades[metric] = "B"
                elif score >= threshold[0]:
                    grades[metric] = "C"
                else:
                    grades[metric] = "D"

        return grades


class RAGEvaluator:
    """
    RAG系统端到端评估器

    功能：
    1. 批量评估
    2. 多维度评估
    3. 生成评估报告
    """

    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.answer_evaluator = AnswerEvaluator()
        self.test_cases: List[RAGEvaluationCase] = []

    def add_test_case(self, case: RAGEvaluationCase):
        """添加测试案例"""
        self.test_cases.append(case)

    def load_test_cases_from_file(self, file_path: str):
        """从文件加载测试案例"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            case = RAGEvaluationCase(
                query=item["query"],
                ground_truth_answer=item.get("ground_truth_answer", ""),
                ground_truth_contexts=item.get("ground_truth_contexts", []),
                retrieved_contexts=item.get("retrieved_contexts", []),
                generated_answer=item.get("generated_answer", ""),
                metadata=item.get("metadata", {})
            )
            self.test_cases.append(case)

        print(f"加载了 {len(self.test_cases)} 个测试案例")

    def evaluate_retrieval_batch(
        self,
        retrieved_results: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        批量评估检索质量

        Args:
            retrieved_results: 检索结果列表
            k_values: K值列表

        Returns:
            评估结果
        """
        all_metrics = {f"recall@{k}": [] for k in k_values}
        all_metrics.update({f"precision@{k}": [] for k in k_values})
        all_metrics.update({f"ndcg@{k}": [] for k in k_values})
        all_metrics["mrr"] = []

        for result in retrieved_results:
            metrics = result.get("metrics", {})
            for key in all_metrics.keys():
                if key in metrics:
                    all_metrics[key].append(metrics[key])

        # 计算平均值
        avg_metrics = {
            key: np.mean(values) if values else 0.0
            for key, values in all_metrics.items()
        }

        return {
            "average_metrics": avg_metrics,
            "num_cases": len(retrieved_results)
        }

    def evaluate_answer_batch(
        self,
        evaluation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        批量评估答案质量

        Args:
            evaluation_results: 评估结果列表

        Returns:
            评估结果
        """
        metrics_aggregated = {
            "relevance_score": [],
            "faithfulness_score": [],
            "accuracy_score": [],
            "hallucination_score": []
        }

        grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

        for result in evaluation_results:
            metrics = result.get("metrics", {})
            for key in metrics_aggregated.keys():
                if key in metrics:
                    metrics_aggregated[key].append(metrics[key])

            grades = result.get("grades", {})
            for g in grades.values():
                if g in grade_counts:
                    grade_counts[g] += 1

        avg_metrics = {
            key: np.mean(values) if values else 0.0
            for key, values in metrics_aggregated.items()
        }

        return {
            "average_metrics": avg_metrics,
            "grade_distribution": grade_counts,
            "num_cases": len(evaluation_results)
        }

    def generate_report(
        self,
        retrieval_results: Dict[str, Any],
        answer_results: Dict[str, Any]
    ) -> str:
        """生成评估报告"""
        report_lines = [
            "=" * 70,
            "RAG系统评估报告",
            "=" * 70,
            ""
        ]

        # 检索评估
        report_lines.extend([
            "【检索质量评估】",
            f"评估案例数: {retrieval_results['num_cases']}",
            ""
        ])

        if "average_metrics" in retrieval_results:
            report_lines.append("检索指标：")
            for metric, value in retrieval_results["average_metrics"].items():
                report_lines.append(f"  {metric}: {value:.4f}")

        report_lines.append("")

        # 答案评估
        report_lines.extend([
            "【答案质量评估】",
            f"评估案例数: {answer_results['num_cases']}",
            ""
        ])

        if "average_metrics" in answer_results:
            report_lines.append("答案指标：")
            for metric, value in answer_results["average_metrics"].items():
                report_lines.append(f"  {metric}: {value:.4f}")

        if "grade_distribution" in answer_results:
            report_lines.append("")
            report_lines.append("等级分布：")
            for grade, count in answer_results["grade_distribution"].items():
                report_lines.append(f"  {grade}: {count}")

        report_lines.append("")
        report_lines.append("=" * 70)

        return "\n".join(report_lines)


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("RAG评估体系测试")
    print("=" * 70)

    # 1. 测试检索评估
    print("\n--- 测试1: 检索评估 ---")
    ret_eval = RetrievalEvaluator()

    test_result = ret_eval.evaluate_retrieval(
        query="感冒了怎么办",
        retrieved_docs=[
            {"id": "chunk_1"},
            {"id": "chunk_2"},
            {"id": "chunk_3"}
        ],
        relevant_doc_ids=["chunk_1", "chunk_3", "chunk_5"],
        k_values=[1, 3, 5]
    )

    print(f"\n查询: {test_result['query']}")
    print("指标:")
    for metric, value in test_result['metrics'].items():
        print(f"  {metric}: {value:.4f}")

    # 2. 测试答案评估
    print("\n--- 测试2: 答案评估 ---")
    ans_eval = AnswerEvaluator()

    eval_result = ans_eval.evaluate_answer(
        query="感冒了怎么办",
        generated_answer="感冒是一种呼吸道疾病，需要多喝水休息。",
        ground_truth="感冒是一种常见的呼吸道感染疾病，主要通过飞沫传播。治疗以对症支持为主。",
        contexts=[
            "感冒是一种常见的呼吸道感染疾病",
            "主要通过飞沫传播",
            "治疗以对症支持为主"
        ]
    )

    print("\n指标:")
    for metric, value in eval_result['metrics'].items():
        print(f"  {metric}: {value:.4f}")

    print("\n等级:")
    for metric, grade in eval_result['grades'].items():
        print(f"  {metric}: {grade}")

    # 3. 测试RAG端到端评估
    print("\n--- 测试3: RAG端到端评估 ---")
    rag_evaluator = RAGEvaluator()

    # 添加测试案例
    test_case = RAGEvaluationCase(
        query="感冒了怎么办",
        ground_truth_answer="感冒是一种呼吸道疾病，需要多喝水休息。",
        ground_truth_contexts=["chunk_1", "chunk_2", "chunk_3"],
        retrieved_contexts=[
            "感冒是一种常见的呼吸道感染疾病",
            "主要通过飞沫传播",
            "治疗以对症支持为主"
        ],
        generated_answer="感冒是一种呼吸道疾病，需要多喝水休息。"
    )

    rag_evaluator.add_test_case(test_case)

    # 评估
    retrieval_results = rag_evaluator.evaluate_retrieval_batch([
        {
            "metrics": {
                "recall@5": 0.8,
                "precision@5": 0.6,
                "ndcg@5": 0.75,
                "mrr": 0.8
            }
        }
    ])

    answer_results = rag_evaluator.evaluate_answer_batch([eval_result])

    # 生成报告
    report = rag_evaluator.generate_report(retrieval_results, answer_results)
    print("\n" + report)

    print("\n" + "=" * 70)
    print("RAG评估体系测试完成!")
    print("=" * 70)