"""
Unified Retriever — 融合向量数据库 + 知识图谱的统一检索层

流程：
    Query
      ├─ NER 提取实体
      ├─ Intent Recognition 意图识别
      ├─ [并行]
      │    ├─ ① Milvus 向量检索 → 找相关文档
      │    └─ ② Neo4j 图谱检索 → 找实体关系
      ├─ 结果合并
      ├─ Jina Rerank 重排
      └─ 构建统一 prompt → 送给大模型
"""

import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

import dashscope
from dashscope import Generation
import py2neo
from dotenv import load_dotenv

load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
os.environ["LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ============================================================================
# UnifiedRetriever
# ============================================================================

class UnifiedRetriever:
    """
    统一检索器：融合 Milvus 向量检索 + Neo4j 知识图谱检索

    用法:
        from vector_db import VectorManager, UnifiedRetriever

        vm = VectorManager()
        vm.connect_milvus()
        vm.create_milvus_collection()

        retriever = UnifiedRetriever(
            vector_manager=vm,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password"
        )

        result = retriever.retrieve("感冒发烧应该吃什么药？")
        print(result["prompt"])        # 可直接送给 LLM 的 prompt
        print(result["merged_results"])  # 合并重排后的结果列表
    """

    # 意图关键词 → 属性名称映射（用于 add_shuxing_prompt 路径）
    # 注意：这些属性应该是 Neo4j 疾病节点上的直接属性
    INTENT_TO_PROPERTY = {
        "简介": "疾病简介",
        "病因": "疾病病因",
        "预防": "预防措施",
        "治疗周期": "治疗周期",
        "治愈概率": "治愈概率",
        "易感人群": "疾病易感人群",
        # 以下属性可能不在 Neo4j 节点上，但尝试查询
        "症状": "疾病症状",
        "治疗方法": "疾病的治疗方法",
        "药品": "疾病的使用药品",
        "检查项目": "疾病的检查项目",
        "宜吃食物": "疾病的饮食建议",
        "忌吃食物": "疾病的饮食建议",
        "并发疾病": "疾病的并发症",
    }

    # 意图关键词 → (关系类型, 目标节点标签) 映射（用于 add_lianxi_prompt 路径）
    INTENT_TO_RELATIONSHIP = {
        "药品": ("疾病使用药品", "药品"),
        "宜吃食物": ("疾病宜吃食物", "食物"),
        "忌吃食物": ("疾病忌吃食物", "食物"),
        "检查项目": ("疾病所需检查", "检查项目"),
        "查询疾病所属科目": ("疾病所属科目", "科目"),
        "症状": ("疾病的症状", "疾病症状"),
        "治疗方法": ("治疗的方法", "治疗方法"),
        "并发疾病": ("疾病并发疾病", "疾病"),
    }

    def __init__(
        self,
        vector_manager,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "neo4j",
        use_local: bool = False,
        local_model: str = "qwen:0.5b",
        ner_cache_dir: str = "tmp_data"
    ):
        self.vector_manager = vector_manager
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_client = None
        self.neo4j_connected = False
        self.use_local = use_local
        self.local_model = local_model
        self.ner_cache_dir = ner_cache_dir

        # NER 组件（延迟加载）
        self._ner_model = None
        self._ner_tokenizer = None
        self._ner_rule = None
        self._ner_tfidf = None
        self._ner_device = None
        self._ner_idx2tag = None

    def _connect_neo4j(self):
        """延迟连接 Neo4j"""
        if self.neo4j_client is not None:
            return self.neo4j_client
        try:
            self.neo4j_client = py2neo.Graph(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            # 测试连接
            self.neo4j_client.run("RETURN 1").data()
            self.neo4j_connected = True
            print("[UnifiedRetriever] Neo4j 连接成功")
        except Exception as e:
            print(f"[UnifiedRetriever] Neo4j 连接失败: {e}")
            self.neo4j_connected = False
        return self.neo4j_client

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        search_type: str = "hybrid",
        use_rerank: bool = True,
        rerank_top_n: int = 5,
        include_prompt: bool = True,
        include_trace: bool = True,
        collection_types: Optional[List[str]] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        统一检索入口：NER → Intent → [KG + Vector 并行] → Merge → Rerank → Prompt

        Args:
            query: 用户查询
            top_k: 向量检索返回数（平衡召回率和延迟）
            search_type: 向量检索类型 (dense/sparse/hybrid)
            use_rerank: 是否使用 Jina Rerank
            rerank_top_n: Rerank 后保留数（减少到5以降低延迟）
            include_prompt: 是否构建最终 prompt
            include_trace: 是否返回 trace 信息
            collection_types: 指定检索的Collection类型列表，默认为["medical_qa"]
            user_id: 用户ID，用于检索用户上传的记忆（病历、文档）

        Returns:
            {
                "query": str,
                "entities": Dict[str, str],
                "intents": str,
                "kg_results": List[Dict],
                "vector_results": List[Dict],
                "merged_results": List[Dict],
                "prompt": str | None,
                "trace": Dict
            }
        """
        total_start = time.time()
        trace = []

        if collection_types is None:
            collection_types = ["medical_qa"]

        # 患者个人病历查询检测（放在 NER / 意图之前，避免加载 BERT 模型、调用 LLM）
        is_patient_query, patient_name = self._detect_patient_record_query(query)

        if is_patient_query:
            # 患者个人病历查询：检索该用户上传的病历集合(scenario/semantic) + 患者节点KG(Patient)，
            # 跳过通用医学知识库(medical_qa)与疾病图谱，避免把个人体征查询带偏到"药品/用药"等通用知识。
            trace.append("patient_record_query")
            target_index = self._extract_target_indicator(query)
            entities = {"患者": patient_name, "指标": target_index}
            intents = '["患者病历查询"]'
            trace.append(f"intents:{intents}")
            trace.append(f"entities:患者={patient_name},指标={target_index}")
            # 优先查询患者节点(Patient)及其关联节点；节点不存在时返回空，不降级查疾病图谱
            record_collections = ["scenario_memory", "semantic_memory"]
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="patient-retrieval") as pool:
                kg_future = pool.submit(self._search_patient_kg, patient_name, user_id)
                vector_future = pool.submit(
                    self._search_vector, query, top_k, "dense", record_collections,
                    intents, entities, user_id, True, True
                )
                kg_results = kg_future.result()
                vector_result = vector_future.result()
            vector_results = vector_result.get("results", [])
            trace.append(f"kg_results:0,vector_results:{len(vector_results)}")
        else:
            # 通用医学问答只能检索公共医学知识库。个人病历仅能由上方的
            # 显式患者病历意图分支访问，否则诸如“感冒怎么办”会被登录用户的
            # 高血压/糖尿病病历污染，既降低召回精度也可能泄露敏感上下文。
            collection_types = [c for c in collection_types if c == "medical_qa"]

            # Step 1: NER 实体提取
            trace.append("ner_start")
            entities = self._run_ner(query)
            trace.append(f"entities:{list(entities.keys())}")

            # Step 2: 意图识别
            trace.append("intent_start")
            intents = self._run_intent(query)
            trace.append(f"intents:{intents[:100]}")

            # Step 3: 并行检索
            trace.append("parallel_retrieval_start")
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="public-retrieval") as pool:
                kg_future = pool.submit(self._search_kg, entities, intents)
                vector_future = pool.submit(
                    self._search_vector, query, top_k, search_type,
                    collection_types, intents, entities, user_id
                )
                kg_results = kg_future.result()
                vector_result = vector_future.result()
            vector_results = vector_result.get("results", [])
            trace.append(f"kg_results:{len(kg_results)},vector_results:{len(vector_results)}")

        # Step 4: 合并结果
        trace.append("merge_start")
        all_results = self._merge_results(kg_results, vector_results)
        if not is_patient_query:
            # Defence in depth: historical graph/vector imports may contain
            # patient reports without reliable metadata. Public questions must
            # never receive these records as prompt context.
            patient_markers = ("本病历仅用于", "初步诊断：", "患者信息：", "现病史", "用药处方", "患者张明华")
            all_results = [r for r in all_results if not (
                str(r.get("source_type", "")).startswith("patient")
                or any(marker in (r.get("content") or "") for marker in patient_markers)
            )]
        trace.append(f"merged:{len(all_results)}")

        # Step 5: Jina Rerank 重排
        if use_rerank and all_results:
            trace.append("rerank_start")
            merged_results = self._apply_rerank(query, all_results, rerank_top_n, entities, intents)
            trace.append(f"reranked:{len(merged_results)}")
        else:
            merged_results = all_results[:rerank_top_n]
            trace.append("rerank_skipped")

        # Evidence gate for dosage questions about a named medicine. Semantic
        # similarity alone can map an invented name such as “火星感冒药” to real
        # cold medicines. Require the complete queried medicine name to appear
        # in retrieved evidence before allowing generation.
        medicine_match = re.search(
            r"([\u4e00-\u9fffA-Za-z0-9]{2,24}(?:片|胶囊|颗粒|口服液|药))"
            r"(?:怎么吃|如何服用|怎么服用|用法|剂量)", query
        )
        if not is_patient_query and medicine_match:
            medicine_name = medicine_match.group(1)
            evidence_text = "\n".join(r.get("content", "") or "" for r in merged_results)
            if medicine_name not in evidence_text:
                merged_results = []
                trace.append(f"evidence_rejected:unknown_medicine")

        # Step 6: 构建统一 prompt
        prompt = None
        if include_prompt:
            trace.append("prompt_build_start")
            prompt = self._build_unified_prompt(query, merged_results, entities)
            trace.append("prompt_built")

        total_elapsed = (time.time() - total_start) * 1000
        trace.append(f"total:{total_elapsed:.0f}ms")

        result = {
            "query": query,
            "entities": entities,
            "intents": intents,
            "kg_results": kg_results,
            "vector_results": vector_results,
            "merged_results": merged_results,
            "prompt": prompt,
            "success": True
        }

        if include_trace:
            result["trace"] = {
                "steps": trace,
                "total_elapsed_ms": total_elapsed
            }

        return result

    # ------------------------------------------------------------------
    # NER（实体识别）
    # ------------------------------------------------------------------

    def _ensure_ner_loaded(self):
        """延迟加载 NER 模型（首次调用时加载）"""
        if self._ner_model is not None:
            return

        import pickle
        import torch
        from transformers import BertTokenizer
        import ner_model as zwk

        with open(f'{self.ner_cache_dir}/tag2idx.npy', 'rb') as f:
            tag2idx = pickle.load(f)

        self._ner_idx2tag = list(tag2idx)
        self._ner_rule = zwk.rule_find()
        self._ner_tfidf = zwk.tfidf_alignment()
        self._ner_device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )

        model_name = 'model/chinese-roberta-wwm-ext'
        self._ner_tokenizer = BertTokenizer.from_pretrained(model_name)
        self._ner_model = zwk.Bert_Model(
            model_name, hidden_size=128, tag_num=len(tag2idx), bi=True
        )
        self._ner_model.load_state_dict(
            torch.load(
                'model/best_roberta_rnn_model_ent_aug.pt',
                map_location=torch.device('cpu')
            )
        )
        self._ner_model = self._ner_model.to(self._ner_device)
        self._ner_model.eval()

    # 医疗词典：用于规则匹配 fallback
    MEDICAL_DICTIONARY = {
        "疾病": [
            "感冒", "发烧", "发热", "高血压", "糖尿病", "心脏病", "胃炎", "肠炎",
            "肝炎", "肺炎", "哮喘", "癌症", "肿瘤", "肾结石", "痔疮",
            "颈椎病", "腰椎间盘突出", "偏头痛", "失眠", "抑郁症", "焦虑症",
            "关节炎", "骨质疏松", "痛风", "甲亢", "甲减", "白内障",
            "鼻炎", "咽炎", "扁桃体炎", "中耳炎", "结膜炎", "湿疹",
            "荨麻疹", "痤疮", "脱发", "肥胖症", "贫血", "白血病",
            "中风", "心梗", "心绞痛", "心律失常", "心衰", "脑梗",
            "胃溃疡", "十二指肠溃疡", "胆结石", "胆囊炎", "胰腺炎",
            "乳腺癌", "肺癌", "胃癌", "肝癌", "肠癌", "宫颈癌", "前列腺癌",
            "流感", "禽流感", "手足口病", "水痘", "麻疹", "风疹",
            "乳腺增生", "脂肪肝", "肝硬化", "慢性肾炎", "尿路感染",
            "前列腺炎", "盆腔炎", "月经不调", "痛经", "不孕症",
            "口腔溃疡", "牙龈炎", "牙周炎", "龋齿", "近视", "远视", "散光"
        ],
        "疾病症状": [
            "头痛", "发热", "发烧", "咳嗽", "腹泻", "恶心", "呕吐", "腹痛",
            "胸闷", "胸痛", "心悸", "气短", "乏力", "失眠", "食欲下降",
            "体重下降", "水肿", "出血", "便血", "尿血", "白带异常",
            "皮肤瘙痒", "皮疹", "关节痛", "肌肉酸痛", "视力模糊",
            "头晕", "眩晕", "耳鸣", "鼻塞", "流涕", "咽痛", "声音嘶哑",
            "呼吸困难", "咯血", "呕血", "黑便", "黄疸", "腹水",
            "咳痰", "打喷嚏", "畏寒", "寒战", "盗汗", "多梦"
        ],
        "药品": [
            "阿司匹林", "布洛芬", "对乙酰氨基酚", "头孢", "阿莫西林",
            "青霉素", "甲硝唑", "奥美拉唑", "雷尼替丁", "多潘立酮",
            "蒙脱石散", "复方甘草片", "止咳糖浆", "感冒灵", "板蓝根",
            "双黄连", "连花清瘟", "达菲", "奥司他韦", "利巴韦林",
            "二甲双胍", "格列美脲", "胰岛素", "硝苯地平", "氨氯地平",
            "美托洛尔", "阿托伐他汀", "辛伐他汀", "氯吡格雷", "华法林",
            "地高辛", "呋塞米", "螺内酯", "氢氯噻嗪", "卡托普利",
            "依那普利", "缬沙坦", "厄贝沙坦", "氯沙坦", "奥美拉唑",
            "泮托拉唑", "雷贝拉唑", "多潘立酮", "莫沙必利", "蒙脱石散",
            "双歧杆菌", "乳酸菌素片", "健胃消食片", "六味地黄丸",
            "逍遥丸", "乌鸡白凤丸", "金匮肾气丸", "知柏地黄丸",
            "安宫牛黄丸", "速效救心丸", "硝酸甘油", "复方丹参片",
            "云南白药", "三七片", "红花油", "风油精", "碘伏"
        ]
    }

    def _rule_based_ner(self, query: str) -> Dict[str, str]:
        """基于规则的实体识别（fallback）"""
        entities = {}
        for entity_type, keywords in self.MEDICAL_DICTIONARY.items():
            for kw in keywords:
                if kw in query:
                    entities[entity_type] = kw
                    break
        return entities

    def _run_ner(self, query: str) -> Dict[str, str]:
        """
        运行 NER 提取医疗实体
        优化策略：优先使用规则匹配（毫秒级），仅在规则匹配无结果时加载BERT模型

        Returns:
            {'疾病': '感冒', '疾病症状': '头痛', '药品': '阿司匹林', ...}
        """
        # 1. 优先使用规则匹配（毫秒级，覆盖常见疾病/症状/药品）
        entities = self._rule_based_ner(query)
        if entities:
            return entities

        # 2. 规则匹配无结果，尝试BERT模型（需加载，较慢）
        try:
            self._ensure_ner_loaded()
            import ner_model as zwk

            entities = zwk.get_ner_result(
                self._ner_model,
                self._ner_tokenizer,
                query,
                self._ner_rule,
                self._ner_tfidf,
                self._ner_device,
                self._ner_idx2tag
            )
            if entities:
                return entities
            print("[NER] BERT模型返回空结果")
        except Exception as e:
            print(f"[NER] BERT模型失败: {e}")

        return {}

    # ------------------------------------------------------------------
    # Intent Recognition（意图识别）
    # ------------------------------------------------------------------

    # 意图关键词匹配规则（优化版，更精准匹配医疗查询）
    # 注意：规则按优先级排序，先匹配药品相关，再匹配饮食相关，避免"吃什么药"误匹配为"宜吃食物"
    INTENT_RULES = [
        ("简介", ["是什么", "介绍", "简介", "什么是", "得了", "是什么病"]),
        ("病因", ["原因", "病因", "为什么", "怎么引起", "为什么会"]),
        ("预防", ["预防", "防止", "避免", "怎么防", "怎么预防"]),
        ("治疗周期", ["多久", "多长时间", "几天", "周期", "几天能好"]),
        ("治愈概率", ["能治好", "能痊愈", "治愈", "治好", "概率", "能根治"]),
        ("易感人群", ["什么人", "哪些人", "易感", "容易得", "什么人容易"]),
        # 药品意图：优先匹配（必须在"宜吃食物"之前，避免"吃什么药"误匹配）
        ("药品", ["吃什么药", "用什么药", "药品", "药物", "怎么用药", "常用药物", "常用的药", "推荐药物", "服药", "吃药"]),
        # 饮食意图：使用更精确的关键词，避免"吃什么药"误匹配
        ("宜吃食物", ["宜吃", "食疗", "饮食建议", "吃什么好", "可以吃什么食物", "应该吃什么食物"]),
        ("忌吃食物", ["不能吃", "忌口", "忌吃", "要少吃", "不能吃什么", "不宜吃"]),
        ("检查项目", ["检查", "化验", "筛查", "做什么检查"]),
        ("查询疾病所属科目", ["什么科", "挂什么科", "科室", "科目", "看什么科"]),
        ("症状", ["症状", "表现", "有什么症状", "有哪些症状", "症状是什么", "表现是什么"]),
        ("治疗方法", ["怎么治", "治疗", "怎么办", "医治", "疗法", "怎么治疗", "治疗方法"]),
        ("并发疾病", ["并发症", "并发", "会引起", "会导致"]),
        ("生产商", ["厂家", "生产商", "哪个厂", "生产"]),
    ]

    # 个人体征/病历字段关键词（用于识别"患者个人病历查询"意图）
    # 含指标类（血压/血糖…）与病历字段类（诊断/处方/用药/检查结果…）
    PERSONAL_INDICATORS = [
        "血压", "血糖", "血脂", "体温", "心率", "脉搏", "呼吸", "血氧",
        "身高", "体重", "bmi", "体重指数",
        "白细胞", "红细胞", "血红蛋白", "血小板", "血常规",
        "尿酸", "胆固醇", "甘油三酯", "转氨酶", "肌酐", "尿素氮",
        "尿蛋白", "尿糖", "血糖值", "血压值",
        # 病历字段类
        "诊断", "处方", "用药", "检查结果", "检验结果", "病史", "主诉",
    ]

    def _detect_patient_record_query(self, query: str):
        """识别"患者个人病历查询"意图。

        当用户询问某位患者（自己或他人）的个人体征指标 / 检查结果时，
        应走病历检索逻辑（scenario_memory），而非通用医学知识问答。

        Returns:
            (is_patient_query: bool, patient_name: str)
        """
        q = query or ""
        for ind in self.PERSONAL_INDICATORS:
            if ind not in q:
                continue
            # 模式1：我的 / 本人 / 病历 / 病例 / 报告 + 指标，如 "我的血压"
            if any(kw in q for kw in ("我的", "本人的", "病历", "病例", "报告")):
                return True, ""
            # 模式2：句首人名 + "的" + 字段，如 "张明华的血压" / "张明华的诊断"
            # 用 re.match 锚定句首，并排除"糖尿病人/患者"等通用称谓，避免误判为病历查询
            m = re.match(r'([一-龥]{2,4})的' + re.escape(ind), q)
            if m and not m.group(1).endswith(("人", "者")):
                name = m.group(1)
                # 排除疾病/症状名，避免"高血压的诊断标准"这类通用问法被误判
                if name in self.MEDICAL_DICTIONARY.get("疾病", []) or name in self.MEDICAL_DICTIONARY.get("疾病症状", []):
                    continue
                return True, name
        return False, ""

    def _extract_target_indicator(self, query: str) -> str:
        """从患者病历查询中抽取目标指标（血压/血糖/体温…）。"""
        q = query or ""
        for ind in self.PERSONAL_INDICATORS:
            if ind in q:
                return ind
        return ""

    def _rule_based_intent(self, query: str) -> str:
        """基于规则的意图识别（fallback）"""
        matched = []
        # 特殊处理：如果查询包含"吃什么药"等药品关键词，不匹配"宜吃食物"
        has_drug_intent = any(kw in query for kw in ["吃什么药", "用什么药", "服药", "吃药"])
        
        for intent_key, patterns in self.INTENT_RULES:
            # 如果已经有药品意图，跳过宜吃食物匹配
            if intent_key == "宜吃食物" and has_drug_intent:
                continue
            for pattern in patterns:
                if pattern in query:
                    matched.append(intent_key)
                    break
        return '["' + '","'.join(matched) + '"] # ' + query

    def _run_intent(self, query: str) -> str:
        """
        意图识别：优先使用规则匹配（快），规则匹配无结果时使用LLM

        Returns:
            原始输出字符串，如 '["查询疾病简介","查询疾病治疗方法"] # 用户想了解...'
        """
        # 优先使用规则匹配（毫秒级响应）
        rule_result = self._rule_based_intent(query)
        
        # 解析规则匹配结果
        try:
            import re
            intents = re.findall(r'"([^"]+)"', rule_result.split("#")[0])
            if intents:
                # 规则匹配成功，直接返回（添加查询内容以便后续处理）
                return rule_result
        except:
            pass
        
        # 规则匹配无结果，使用LLM（较慢但更智能）
        prompt = self._build_intent_prompt(query)

        try:
            if self.use_local:
                import ollama
                result = ollama.generate(model=self.local_model, prompt=prompt)
                return result['response']
            else:
                response = Generation.call(
                    model=Generation.Models.qwen_plus,
                    prompt=prompt
                )
                return response.output.text
        except Exception as e:
            print(f"LLM 意图识别失败: {e}，使用规则匹配 fallback")
            # Fallback: 基于规则的意图识别
            return rule_result

    def _build_intent_prompt(self, query: str) -> str:
        """构建意图识别的 few-shot prompt"""
        return f"""阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

你需要依据的16个查询意图类别如下：
查询疾病简介, 查询疾病病因, 查询疾病预防措施, 查询疾病治疗周期, 查询治愈概率, 查询疾病易感人群, 查询疾病所需药品, 查询疾病宜吃食物, 查询疾病忌吃食物, 查询疾病所需检查项目, 查询疾病所属科目, 查询疾病的症状, 查询疾病的治疗方法, 查询疾病的并发疾病, 查询药品的生产商

例子：
问题：感冒了怎么办？
回答：["查询疾病简介","查询疾病的症状","查询疾病所需药品","查询疾病的治疗方法"] # 用户问感冒怎么办，需要全面了解感冒的简介、症状、药品和治疗方法。

问题：头痛可能是什么病？
回答：["查询疾病的症状"] # 用户描述了一个症状，需要反向查找可能的疾病。

问题：阿莫西林是哪个厂家的？
回答：["查询药品的生产商"] # 用户在询问药品的生产商信息。

问题：感冒的病因是什么？
回答：["查询疾病病因"] # 用户直接询问疾病病因。

问题：高血压不能吃什么？
回答：["查询疾病忌吃食物"] # 用户询问疾病的饮食禁忌。

问题：{query}
回答："""

    # ------------------------------------------------------------------
    # KG 检索辅助方法
    # ------------------------------------------------------------------

    # 症状到疾病的映射（fallback 使用）
    SYMPTOM_TO_DISEASE = {
        "发热": ["感冒", "流感", "肺炎", "肝炎"],
        "咳嗽": ["感冒", "流感", "肺炎", "哮喘", "支气管炎"],
        "头痛": ["感冒", "偏头痛", "高血压", "脑梗"],
        "腹泻": ["肠炎", "胃炎", "食物中毒", "糖尿病"],
        "胸闷": ["心脏病", "高血压", "焦虑症"],
        "胸痛": ["心绞痛", "心梗", "肺炎"],
        "心悸": ["心律失常", "甲亢", "心脏病"],
        "呼吸困难": ["哮喘", "肺炎", "心衰"],
        "乏力": ["贫血", "糖尿病", "甲亢", "肝病"],
        "体重下降": ["糖尿病", "癌症", "甲亢", "抑郁症"],
        "恶心": ["胃炎", "食物中毒", "肝炎", "早孕反应"],
        "呕吐": ["胃炎", "食物中毒", "胰腺炎", "偏头痛"],
        "腹痛": ["胃炎", "肠炎", "胆囊炎", "胰腺炎"],
        "失眠": ["失眠症", "焦虑症", "抑郁症", "甲亢"],
        "头晕": ["高血压", "低血压", "贫血", "颈椎病"],
        "眩晕": ["高血压", "颈椎病", "内耳眩晕症"],
        "鼻塞": ["感冒", "鼻炎", "流感"],
        "流涕": ["感冒", "鼻炎", "流感"],
        "咽痛": ["感冒", "咽炎", "扁桃体炎"],
        "皮疹": ["湿疹", "荨麻疹", "过敏", "水痘"],
        "水肿": ["心衰", "肾病", "肝病", "甲状腺疾病"],
        "关节痛": ["关节炎", "痛风", "风湿病"],
        "肌肉酸痛": ["感冒", "流感", "过度疲劳"],
    }

    # 药品到疾病的映射（fallback 使用）
    DRUG_TO_DISEASE = {
        "对乙酰氨基酚": "感冒",
        "布洛芬": "关节炎",
        "阿司匹林": "心脏病",
        "阿莫西林": "感染",
        "头孢": "感染",
        "奥美拉唑": "胃炎",
        "蒙脱石散": "肠炎",
        "二甲双胍": "糖尿病",
        "胰岛素": "糖尿病",
        "硝苯地平": "高血压",
        "美托洛尔": "高血压",
        "阿托伐他汀": "高血脂",
        "硝酸甘油": "心绞痛",
        "速效救心丸": "心脏病",
        "复方丹参片": "心脏病",
    }

    def _extract_disease_keyword(
        self, intents: str, symptom: str = "", drug: str = ""
    ) -> str:
        """从意图、症状、药品中反推可能的疾病"""
        # 1. 从药品反推疾病
        if drug and drug in self.DRUG_TO_DISEASE:
            return self.DRUG_TO_DISEASE[drug]

        # 2. 从症状反推疾病
        if symptom and symptom in self.SYMPTOM_TO_DISEASE:
            return self.SYMPTOM_TO_DISEASE[symptom][0]

        # 3. 从意图关键词推断（简单规则）
        intent_keywords = {
            "药品": "感冒",  # 默认给一个常见疾病
            "症状": "感冒",
            "治疗": "感冒",
        }
        for kw, disease in intent_keywords.items():
            if kw in intents:
                return disease

        return ""

    # ------------------------------------------------------------------
    # KG 检索（Neo4j 知识图谱）
    # ------------------------------------------------------------------

    def _search_kg(self, entities: Dict[str, str], intents: str) -> List[Dict]:
        """
        基于 NER 实体 + 意图 → 查询 Neo4j → 返回结构化结果列表
        当实体为空时，使用关键词提取作为 fallback

        Returns:
            List[Dict] 每个包含:
                id: str         — 唯一标识 (如 "kg_disease_intro_0")
                content: str    — 自然语言描述的知识
                source: str     — "kg"
                _kg_score: float — 默认 1.0（精确匹配高置信度）
                source_type: str — "property" | "relationship"
        """
        results = []
        disease = entities.get("疾病", "")

        # 如果没有识别出疾病实体，尝试从查询中提取关键词
        if not disease:
            disease = self._extract_disease_keyword(
                intents, 
                entities.get("疾病症状", ""), 
                entities.get("药品", "")
            )
            if disease:
                print(f"[KG] 关键词提取疾病: {disease}")

        if not disease:
            return results

        # 1) 属性查询（add_shuxing_prompt 路径）
        for intent_key, property_name in self.INTENT_TO_PROPERTY.items():
            if intent_key not in intents:
                continue

            data = self._query_disease_property(disease, property_name)
            if data:
                # 映射属性到文档类型
                type_mapping = {
                    "疾病简介": "疾病概览",
                    "疾病症状": "症状",
                    "疾病的检查项目": "检查",
                    "疾病的治疗方法": "治疗",
                    "疾病的使用药品": "药品",
                    "疾病的饮食建议": "饮食",
                    "疾病的病因": "病因",
                    "疾病的预防措施": "预防"
                }
                doc_type = type_mapping.get(property_name, property_name)
                results.append({
                    "id": f"kg_{disease}_{property_name}",
                    "content": f"关于{disease}的{property_name}：{data}",
                    "source": "kg",
                    "metadata": {
                        "type": doc_type,
                        "disease": disease
                    },
                    "_kg_score": 1.0,
                    "source_type": "property",
                    "entity": disease,
                    "attribute": property_name
                })

        # 2) 关系查询（add_lianxi_prompt 路径）
        for intent_key, (rel_type, target_label) in self.INTENT_TO_RELATIONSHIP.items():
            if intent_key not in intents:
                continue

            names = self._query_disease_relationships(disease, rel_type, target_label)
            if names:
                label_cn = _label_to_chinese(target_label)
                rel_cn = _rel_to_chinese(rel_type)
                # 映射关系类型到文档类型（注意：rel_type来自INTENT_TO_RELATIONSHIP，不含"的"）
                rel_type_mapping = {
                    "疾病使用药品": "药品",
                    "疾病治疗方法": "治疗",
                    "疾病并发疾病": "并发疾病",
                    "疾病宜吃食物": "饮食",
                    "疾病忌吃食物": "饮食",
                    "疾病所需检查": "检查",
                    "疾病所属科目": "科室",
                    "疾病的症状": "症状",
                    "治疗的方法": "治疗",
                }
                doc_type = rel_type_mapping.get(rel_type, rel_type)
                results.append({
                    "id": f"kg_{disease}_{rel_type}",
                    "content": f"{disease}的{rel_cn}包括：{names}",
                    "source": "kg",
                    "metadata": {
                        "type": doc_type,
                        "disease": disease
                    },
                    "_kg_score": 1.0,
                    "source_type": "relationship",
                    "entity": disease,
                    "relation": rel_type,
                    "target_label": target_label,
                    "target_names": names
                })

        # 3) 药品商查询（特殊路径，从药品查生产商）
        drug_name = entities.get("药品", "")
        if "生产商" in intents and drug_name:
            producer = self._query_drug_producer(drug_name)
            if producer:
                results.append({
                    "id": f"kg_producer_{drug_name}",
                    "content": f"药品{drug_name}的生产商是：{producer}",
                    "source": "kg",
                    "_kg_score": 1.0,
                    "source_type": "relationship",
                    "entity": drug_name,
                    "relation": "生产"
                })

        # 4) 症状反查疾病
        symptom = entities.get("疾病症状", "")
        if symptom and not disease:
            diseases = self._query_disease_by_symptom(symptom)
            if diseases:
                results.append({
                    "id": f"kg_symptom_{symptom}",
                    "content": f"症状{symptom}可能关联的疾病：{diseases}（推测性结果，需进一步确认）",
                    "source": "kg",
                    "_kg_score": 0.7,  # 推测性结果降权
                    "source_type": "reverse_lookup",
                    "entity": symptom
                })

        return results

    def _search_patient_kg(self, patient_name: str, user_id: str = None) -> List[Dict]:
        """患者病历查询专用 KG 检索：查询患者节点及其关联的疾病/用药/检查节点。

        患者节点标签为 :Patient，用 name 属性匹配患者姓名。
        若患者节点不存在，返回空数组（不降级查询通用疾病图谱）。
        """
        # A patient name is not an authorization boundary. Patient nodes must be
        # owned by the authenticated user before their properties are returned.
        if not patient_name or not user_id:
            return []
        neo4j = self._connect_neo4j()
        if neo4j is None:
            return []
        results = []
        try:
            # 1) 患者节点属性
            cursor = neo4j.run(
                "MATCH (p:Patient {name: $name, user_id: $user_id}) RETURN p",
                name=patient_name, user_id=str(user_id)
            )
            records = list(cursor)
            if not records:
                print(f"[PatientKG] 未找到患者节点: {patient_name}")
                return []

            node = records[0][0]
            props = dict(node)
            info_items = [
                f"{k}: {v}" for k, v in props.items()
                if v not in (None, "") and not isinstance(v, (list, dict)) and k != "name"
            ]
            if info_items:
                results.append({
                    "id": f"kg_patient_{patient_name}",
                    "content": f"患者{patient_name}的病历信息：{'；'.join(info_items)}",
                    "source": "kg",
                    "metadata": {"type": "患者病历", "patient": patient_name},
                    "_kg_score": 1.0,
                    "source_type": "patient",
                    "entity": patient_name,
                    "attribute": "患者信息",
                })

            # 2) 患者节点关联的疾病/用药/检查节点
            relationships = [
                ("HAS_DISEASE", "疾病", "诊断"),
                ("TAKES_DRUG", "药品", "用药"),
                ("HAS_EXAM", "检查项目", "检查"),
            ]
            for rel_type, target_label, rel_cn in relationships:
                try:
                    cursor = neo4j.run(
                        f"MATCH (p:Patient {{name: $name, user_id: $user_id}})-[:{rel_type}]->(t:{target_label}) "
                        "RETURN t.名称",
                        name=patient_name, user_id=str(user_id),
                    )
                    names = [str(r[0]) for r in cursor if r[0]]
                except Exception:
                    names = []
                if not names:
                    continue
                results.append({
                    "id": f"kg_patient_{patient_name}_{rel_type}",
                    "content": f"患者{patient_name}的{rel_cn}：{'、'.join(names)}",
                    "source": "kg",
                    "metadata": {"type": "患者病历", "patient": patient_name, "relation": rel_type},
                    "_kg_score": 0.9,
                    "source_type": "patient_relationship",
                    "entity": patient_name,
                    "relation": rel_type,
                    "target_label": target_label,
                    "target_names": names,
                })
        except Exception as e:
            print(f"[PatientKG] 查询失败: {e}")
        return results

    # 允许查询的疾病属性列表（扩展以支持更多医疗查询）
    ALLOWED_PROPERTIES = {
        "疾病简介", "疾病病因", "预防措施", "治疗周期", "治愈概率", "疾病易感人群",
        "疾病症状", "疾病的检查项目", "疾病的治疗方法", "疾病的使用药品",
        "疾病的饮食建议", "疾病的并发症", "疾病的预后", "疾病的诊断方法"
    }

    def _query_disease_property(self, disease: str, property_name: str) -> Optional[str]:
        """查询疾病节点的属性值
        
        注意：Neo4j Cypher 不支持用参数动态指定属性名，
        所以需要用字符串拼接方式构建查询。
        """
        try:
            if property_name not in self.ALLOWED_PROPERTIES:
                return None
            neo4j = self._connect_neo4j()
            if neo4j is None:
                return None
            # 使用字符串拼接（属性名不能用参数）
            cypher = f'MATCH (a:疾病{{名称:$disease}}) RETURN a.`{property_name}`'
            cursor = neo4j.run(cypher, disease=disease)
            records = list(cursor)
            if records and records[0][0]:
                return str(records[0][0])
        except Exception as e:
            print(f"KG属性查询失败 [{disease}.{property_name}]: {e}")
        return None

    ALLOWED_RELATIONSHIPS = {"疾病使用药品", "疾病宜吃食物", "疾病忌吃食物", "疾病所需检查", "疾病所属科目", "疾病的症状", "治疗的方法", "疾病并发疾病"}
    ALLOWED_LABELS = {"药品", "食物", "检查项目", "科目", "疾病症状", "治疗方法", "疾病"}

    def _query_disease_relationships(
        self, disease: str, rel_type: str, target_label: str
    ) -> Optional[str]:
        """查询疾病的关系目标节点名称列表，返回中文顿号分隔的字符串"""
        try:
            if rel_type not in self.ALLOWED_RELATIONSHIPS or target_label not in self.ALLOWED_LABELS:
                return None
            neo4j = self._connect_neo4j()
            if neo4j is None:
                return None
            cypher = (
                f"MATCH (a:疾病{{名称:$disease}})"
                f"-[r:{rel_type}]->(b:{target_label}) RETURN b.名称"
            )
            cursor = neo4j.run(cypher, disease=disease)
            names = [str(record[0]) for record in cursor if record[0]]
            return "、".join(names) if names else None
        except Exception as e:
            print(f"KG关系查询失败 [{disease}-{rel_type}->{target_label}]: {e}")
            return None

    def _query_drug_producer(self, drug_name: str) -> Optional[str]:
        """查询药品的生产商"""
        try:
            neo4j = self._connect_neo4j()
            if neo4j is None:
                return None
            cypher = (
                "MATCH (a:药品商)-[r:生产]->(b:药品{名称:$drug}) RETURN a.名称"
            )
            cursor = neo4j.run(cypher, drug=drug_name)
            names = [str(record[0]) for record in cursor if record[0]]
            return "、".join(names) if names else None
        except Exception as e:
            print(f"KG药品商查询失败 [{drug_name}]: {e}")
            return None

    def _query_disease_by_symptom(self, symptom: str) -> Optional[str]:
        """通过症状反查可能的疾病"""
        try:
            neo4j = self._connect_neo4j()
            if neo4j is None:
                return None
            cypher = (
                "MATCH (a:疾病)-[r:疾病的症状]->(b:疾病症状 {名称:$symptom}) "
                "RETURN a.名称"
            )
            cursor = neo4j.run(cypher, symptom=symptom)
            names = [str(record[0]) for record in cursor if record[0]]
            return "、".join(names) if names else None
        except Exception as e:
            print(f"KG症状反查失败 [{symptom}]: {e}")
            return None

    # ------------------------------------------------------------------
    # Vector 检索（Milvus）
    # ------------------------------------------------------------------

    # 医疗知识库（fallback 使用）
    MEDICAL_KNOWLEDGE_BASE = {
        "感冒": {
            "简介": "感冒是由病毒感染引起的上呼吸道疾病，分为普通感冒和流行性感冒。",
            "症状": "鼻塞、流涕、咽痛、咳嗽、轻度发热、头痛、乏力等。",
            "药品": "对乙酰氨基酚、布洛芬用于退热止痛；感冒灵、板蓝根等中成药可缓解症状。",
            "治疗方法": "多喝水、多休息、对症治疗。一般1-2周可自愈。",
            "预防措施": "勤洗手、避免接触患者、保持室内通风、增强免疫力。",
            "饮食": "宜清淡易消化食物，多饮水；忌辛辣刺激、油腻食物。"
        },
        "发烧": {
            "简介": "发烧是人体免疫系统对抗感染的正常反应，体温超过37.3°C。",
            "症状": "体温升高、乏力、头痛、肌肉酸痛、出汗等。",
            "药品": "对乙酰氨基酚（扑热息痛）、布洛芬、阿司匹林等退烧药。",
            "治疗方法": "物理降温（温水擦浴）、多喝水、必要时服用退烧药。",
            "预防措施": "加强锻炼、增强免疫力、避免受凉。",
            "饮食": "宜多饮水、清淡流质食物；忌油腻、辛辣食物。"
        },
        "高血压": {
            "简介": "高血压是指动脉收缩压≥140mmHg或舒张压≥90mmHg的慢性病。",
            "症状": "头痛、头晕、心悸、疲劳等，部分患者无明显症状。",
            "药品": "硝苯地平、氨氯地平、美托洛尔、缬沙坦、卡托普利等。",
            "治疗方法": "长期规律服药、定期监测血压、改善生活方式。",
            "预防措施": "低盐饮食、规律运动、控制体重、戒烟限酒、定期体检。",
            "饮食": "宜低盐低脂、多蔬菜水果；忌高盐、高脂、辛辣食物。"
        },
        "糖尿病": {
            "简介": "糖尿病是由胰岛素分泌不足或作用障碍引起的代谢性疾病。",
            "症状": "多饮、多尿、多食、体重下降、乏力、视物模糊等。",
            "药品": "二甲双胍、格列美脲、胰岛素、阿卡波糖等。",
            "治疗方法": "饮食控制、运动疗法、药物治疗、血糖监测。",
            "预防措施": "健康饮食、规律运动、控制体重、定期体检。",
            "饮食": "宜低GI食物、高纤维；忌高糖、高脂食物。"
        },
        "胃炎": {
            "简介": "胃炎是胃黏膜的炎症，分为急性胃炎和慢性胃炎。",
            "症状": "上腹痛、恶心、呕吐、反酸、嗳气等。",
            "药品": "奥美拉唑、雷尼替丁、铝碳酸镁、多潘立酮等。",
            "治疗方法": "抑酸治疗、保护胃黏膜、根除幽门螺旋杆菌。",
            "预防措施": "规律饮食、避免辛辣刺激、戒烟限酒、减少压力。",
            "饮食": "宜清淡易消化、少食多餐；忌辛辣、油腻、咖啡、酒精。"
        },
        "头痛": {
            "简介": "头痛是临床常见症状，可能由多种原因引起。",
            "症状": "头部疼痛，可能伴随恶心、呕吐、畏光等。",
            "药品": "布洛芬、对乙酰氨基酚、阿司匹林等。",
            "治疗方法": "休息、止痛药物、针对病因治疗。",
            "预防措施": "规律作息、减少压力、避免诱因。",
            "饮食": "宜清淡、避免咖啡因。"
        },
        "咳嗽": {
            "简介": "咳嗽是人体的保护性反射，可由多种疾病引起。",
            "症状": "咳嗽、咳痰、可能伴随咽痛、胸痛等。",
            "药品": "复方甘草片、止咳糖浆、氨溴索、右美沙芬等。",
            "治疗方法": "止咳祛痰、针对病因治疗。",
            "预防措施": "戒烟、保持室内空气流通、增强免疫力。",
            "饮食": "宜多饮水、润喉食物（梨、蜂蜜）；忌辛辣刺激。"
        },
        "腹泻": {
            "简介": "腹泻是指排便次数增多、粪便稀薄的症状。",
            "症状": "频繁排便、粪便稀薄、腹痛、脱水等。",
            "药品": "蒙脱石散、双歧杆菌、口服补液盐等。",
            "治疗方法": "补液、止泻、调节肠道菌群。",
            "预防措施": "注意饮食卫生、勤洗手、避免生冷食物。",
            "饮食": "宜清淡流质（米汤、粥）；忌油腻、生冷食物。"
        },
    }

    def _get_fallback_knowledge(self, query: str, entities: Dict[str, str]) -> List[Dict]:
        """从内置知识库获取 fallback 结果"""
        results = []
        disease = entities.get("疾病", "")
        symptom = entities.get("疾病症状", "")
        drug = entities.get("药品", "")

        # 查找匹配的疾病知识
        matched_disease = None
        for name in [disease, symptom]:
            if name and name in self.MEDICAL_KNOWLEDGE_BASE:
                matched_disease = name
                break

        # 药品匹配
        if not matched_disease and drug:
            for dname, knowledge in self.MEDICAL_KNOWLEDGE_BASE.items():
                for key, content in knowledge.items():
                    if drug in content:
                        matched_disease = dname
                        break
                if matched_disease:
                    break

        if not matched_disease:
            # 通用 fallback
            return [{
                "id": "fallback_general",
                "content": "作为医疗AI助手，我可以回答关于疾病的症状、治疗、用药等问题。请问您具体想了解哪方面的信息？建议您提供具体的疾病名称或症状描述，以便我给出更准确的回答。",
                "source": "fallback",
                "_kg_score": 0.3,
                "source_type": "general"
            }]

        # 返回匹配疾病的各方面知识
        knowledge = self.MEDICAL_KNOWLEDGE_BASE[matched_disease]
        # 类型映射：与Rerank的intent_type保持一致
        type_map = {
            "简介": "疾病概览",
            "症状": "症状",
            "药品": "药品",
            "治疗方法": "治疗",
            "预防措施": "预防",
            "饮食": "饮食"
        }

        for key, content in knowledge.items():
            doc_type = type_map.get(key, key)
            results.append({
                "id": f"kb_{matched_disease}_{key}",
                "content": f"【{matched_disease}·{doc_type}】{content}",
                "source": "kb",
                "metadata": {
                    "type": doc_type,
                    "disease": matched_disease
                },
                "_kg_score": 0.85,
                "source_type": "knowledge_base",
                "entity": matched_disease,
                "attribute": doc_type
            })

        return results

    def _search_vector(
        self, query: str, top_k: int, search_type: str,
        collection_types: Optional[List[str]] = None, intent: str = None, entities: Dict[str, str] = None,
        user_id: str = None, is_patient_query: bool = False, report_only: bool = False
    ) -> Dict[str, Any]:
        """
        调用 VectorManager 搜索，失败时使用内置知识库 fallback

        优化版本：根据意图识别结果增强查询，提升召回率
        """
        try:
            # 意图引导的查询增强（轻度增强，避免过度稀释语义）
            if is_patient_query:
                # 患者病历查询：检索关键词 = 患者姓名 + 目标指标，去掉"是多少/怎么样"等噪音词
                patient_name = entities.get("患者", "") if entities else ""
                target_index = entities.get("指标", "") if entities else ""
                enhanced_query = " ".join(x for x in (patient_name, target_index) if x) or query
                if enhanced_query != query:
                    print(f"[患者病历检索] {query} -> {enhanced_query}")
            else:
                enhanced_query = query
                if intent:
                    if "药品" in intent or "用药" in intent:
                        enhanced_query = f"{query} 药品 用药"
                    elif "症状" in intent:
                        enhanced_query = f"{query} 症状"
                    elif "治疗" in intent:
                        enhanced_query = f"{query} 治疗"
                    elif "饮食" in intent or "宜吃" in intent or "忌吃" in intent:
                        enhanced_query = f"{query} 饮食"

                    if enhanced_query != query:
                        print(f"[查询增强] {query} -> {enhanced_query}")

            # 提取疾病实体用于向量检索过滤（患者病历查询不做疾病过滤）
            disease_entity = ""
            if entities and not is_patient_query:
                disease_entity = entities.get("疾病", "")

            if collection_types and len(collection_types) > 1:
                result = self.vector_manager.multi_collection_search(
                    enhanced_query,
                    top_k=top_k,
                    search_type=search_type,
                    use_rerank=False,
                    collection_types=collection_types,
                    user_id=user_id,  # 传递 user_id 用于记忆集合过滤
                    report_only=report_only  # 患者病历查询：只检索病历块
                )
            else:
                col_type = collection_types[0] if collection_types else "medical_qa"
                result = self.vector_manager.search(
                    enhanced_query, top_k=top_k, search_type=search_type,
                    use_rerank=False, collection_type=col_type,
                    disease_filter=disease_entity,
                    user_id=user_id,  # 传递 user_id 用于记忆集合过滤
                    report_only=report_only
                )
            
            # 如果向量检索返回空结果，使用 fallback
            if not result.get("results"):
                if is_patient_query:
                    # 患者病历查询：未命中病历时不回退到通用医学知识，
                    # 直接返回空结果，交由上层提示"未找到该患者的病历/指标记录"
                    print("[Vector] 未检索到该患者的病历信息")
                    return {
                        "success": True,
                        "results": [],
                        "count": 0,
                        "message": "未找到该患者的病历信息",
                        "fallback": False
                    }
                print("[Vector] 向量检索返回空结果，使用内置知识库 fallback")
                fallback_results = self._get_fallback_knowledge(
                    query,
                    self._rule_based_ner(query)
                )
                return {
                    "success": True,
                    "results": fallback_results,
                    "count": len(fallback_results),
                    "fallback": True
                }
            return result
        except Exception as e:
            print(f"向量检索失败: {e}，使用内置知识库 fallback")
            # Fallback: 使用内置知识库
            fallback_results = self._get_fallback_knowledge(
                query,
                self._rule_based_ner(query)
            )
            return {
                "success": True,
                "results": fallback_results,
                "count": len(fallback_results),
                "fallback": True
            }

    # ------------------------------------------------------------------
    # Merge + Rerank
    # ------------------------------------------------------------------

    def _merge_results(
        self, kg_results: List[Dict], vector_results: List[Dict]
    ) -> List[Dict]:
        """
        合并 KG 结果和向量结果

        KG 结果没有 score 字段，赋予 _kg_score=1.0
        向量结果保留原始 score/rrf_score/rerank_score
        统一为 Jina Rerank 所需的 {id, content} 格式
        """
        merged = []
        rrf_k = int(os.getenv("CROSS_SOURCE_RRF_K", "60"))

        # KG 结果在前（结构化知识优先级高）
        for rank, item in enumerate(kg_results, 1):
            merged.append({
                "id": item.get("id", ""),
                "content": item.get("content", ""),
                "source": "kg",
                "metadata": item.get("metadata", {}),  # 保留metadata
                "source_type": item.get("source_type", ""),
                "entity": item.get("entity", ""),
                "attribute": item.get("attribute", ""),
                "_kg_score": item.get("_kg_score", 1.0),
                "kg_rank": rank,
                "rrf_score": 1.0 / (rrf_k + rank),
            })

        # 向量结果在后（包括fallback结果）
        for rank, item in enumerate(vector_results, 1):
            # fallback结果的source可能是"kb"或"fallback"
            item_source = item.get("source", "vector")
            merged.append({
                "id": item.get("id") or item.get("chunk_id", ""),
                "content": item.get("content", ""),
                "source": item_source,
                "metadata": item.get("metadata", {}),  # 保留metadata
                "score": item.get("score") or item.get("rrf_score", 0),
                "vector_rank": rank,
                "rrf_score": 1.0 / (rrf_k + rank),
                "dense_rank": item.get("dense_rank"),
                "sparse_rank": item.get("sparse_rank"),
                "parent_id": item.get("parent_id", ""),
                "document_id": item.get("document_id", ""),
            })

        # Cross-source RRF makes KG and vector candidates comparable without
        # mixing incompatible raw similarity scales. Stable source priority is
        # used only as a tie-breaker.
        return sorted(
            merged,
            key=lambda item: (
                -item.get("rrf_score", 0.0),
                0 if item.get("source") == "kg" else 1,
                item.get("id", ""),
            ),
        )

    def _apply_rerank(
        self, query: str, candidates: List[Dict], top_n: int,
        entities: Dict[str, str] = None, intent: str = None
    ) -> List[Dict]:
        """
        对合并后的 KG + Vector 结果统一调用 Jina Rerank

        Jina Rerank 根据 content 与 query 的语义相关性重新排序，
        这样 KG 结果和 Vector 结果在同一标准下公平比较
        
        Args:
            query: 用户查询
            candidates: 候选文档列表
            top_n: 返回数量
            entities: 实体识别结果（用于增强Rerank）
            intent: 意图识别结果（用于增强Rerank）
        """
        try:
            reranked = self.vector_manager.rerank_with_jina(
                query=query,
                results=candidates,
                top_n=top_n,
                entities=entities,
                intent=intent
            )
            # 确保 source 信息不丢失
            for item in reranked:
                if "source" not in item:
                    item["source"] = "unknown"
            return reranked
        except Exception as e:
            print(f"Rerank 失败，返回原始合并结果: {e}")
            # Rerank 失败时按 _kg_score 和 score 混合排序后截断
            sorted_results = sorted(
                candidates,
                key=lambda x: (
                    x.get("_kg_score", 0) or x.get("score", 0)
                ),
                reverse=True
            )
            return sorted_results[:top_n]

    # ------------------------------------------------------------------
    # 构建统一 Prompt
    # ------------------------------------------------------------------

    def _build_unified_prompt(
        self,
        query: str,
        merged_results: List[Dict],
        entities: Dict[str, str]
    ) -> str:
        """
        构建融合 KG + Vector 结果的统一 prompt
        （v3版本：高相关性+高Faithfulness版本）
        """
        # 患者病历查询：走专用提示词，禁止用通用医学知识回答个人体征问题
        is_patient_query, patient_name = self._detect_patient_record_query(query)
        if is_patient_query:
            return self._build_patient_query_prompt(query, merged_results, patient_name, entities)

        kg_items = [r for r in merged_results if r.get("source") == "kg"]
        vector_items = [r for r in merged_results if r.get("source") == "vector"]

        # 提取疾病实体
        disease_entity = entities.get("疾病", "")
        symptom_entity = entities.get("疾病症状", "")
        drug_entity = entities.get("药品", "")

        prompt_parts = []

        # 系统提示：强调相关性和忠实性
        system_hint = (
            "你是一个专业的医疗问答助手。请根据下方提供的医学知识，"
            "准确、全面地回答用户的问题。\n\n"
            "【重要规则】\n"
            "- 只能使用提供的参考资料中的信息进行回答\n"
            "- 不要添加参考资料中没有的信息\n"
            "- 严禁使用你自己的训练知识或通用医学常识补充答案，"
            "即使你知道相关内容也必须以参考资料为准\n"
            "- 优先复述参考资料中的原词和短句组合，避免改写成参考资料中"
            "没有出现的专业表述\n"
            "- 如果提供的知识不足以回答问题，请说\"根据已知信息无法回答该问题\"\n"
            "- 回答要准确、简洁、有条理\n"
        )

        # 添加疾病实体提示（让LLM聚焦）
        if disease_entity:
            system_hint += f"\n【用户查询的疾病】：{disease_entity}\n"
            system_hint += "【重要】请只针对上述疾病进行回答，不要列举其他疾病的信息。"
        
        if symptom_entity:
            system_hint += f"\n【用户描述的症状】：{symptom_entity}"
        
        if drug_entity:
            system_hint += f"\n【用户提到的药品】：{drug_entity}"

        prompt_parts.append(system_hint)

        # 知识图谱信息（结构化知识，优先级高）
        if kg_items:
            prompt_parts.append("【结构化知识】（来自医学知识图谱）：")
            kg_text_parts = []
            for item in kg_items:
                content = item.get("content", "")
                src_type = item.get("source_type", "")
                if src_type == "reverse_lookup":
                    kg_text_parts.append(
                        f"- {content}（注意：这是推测性结果）"
                    )
                else:
                    kg_text_parts.append(f"- {content}")
            prompt_parts.extend(kg_text_parts)

        # 向量检索文档
        if vector_items:
            # 分离用户上传内容和系统知识库内容
            user_uploaded_items = []
            system_items = []
            for item in vector_items:
                try:
                    meta = item.get("metadata", {})
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    # 检查是否是用户上传的内容
                    file_type = meta.get("file_type", "")
                    if file_type in ["report", "document"]:
                        user_uploaded_items.append(item)
                    else:
                        system_items.append(item)
                except:
                    system_items.append(item)
            
            # 先显示用户上传的个人数据
            if user_uploaded_items:
                prompt_parts.append("【您的个人医疗数据】（您上传的病历/文档）：")
                for i, item in enumerate(user_uploaded_items, 1):
                    content = item.get("content", "")
                    doc_type = ""
                    try:
                        meta = item.get("metadata", {})
                        if isinstance(meta, str):
                            meta = json.loads(meta)
                        doc_type = meta.get("type", "")
                        file_type = meta.get("file_type", "")
                        type_label = f"（{doc_type}，个人数据）"
                    except:
                        type_label = "（个人数据）"
                    
                    if len(content) > 400:
                        content = content[:400] + "..."
                    prompt_parts.append(f"个人资料{i}{type_label}：{content}")
            
            # 再显示系统知识库内容
            if system_items:
                prompt_parts.append("【医学知识参考】（来自系统知识库）：")
                for i, item in enumerate(system_items, 1):
                    content = item.get("content", "")
                    doc_type = ""
                    doc_disease = ""
                    try:
                        meta = item.get("metadata", {})
                        if isinstance(meta, str):
                            meta = json.loads(meta)
                        doc_type = meta.get("type", "")
                        doc_disease = meta.get("disease", "")
                    except:
                        pass
                    
                    # 标记疾病匹配状态
                    if doc_disease and disease_entity and doc_disease == disease_entity:
                        type_label = f"（{doc_type}，已确认匹配）"
                    elif doc_disease:
                        type_label = f"（{doc_type}，疾病：{doc_disease}）"
                    else:
                        type_label = f"（{doc_type}）" if doc_type else ""
                    
                    # 截断过长的内容
                    if len(content) > 400:
                        content = content[:400] + "..."
                    prompt_parts.append(f"参考资料{i}{type_label}：{content}")

        # 用户问题
        prompt_parts.append(f"【用户问题】{query}")

        # 回答指导
        answer_guide = "请综合以上所有知识，给出完整的回答。回答要求：\n"
        if disease_entity:
            answer_guide += f"1. 只回答关于「{disease_entity}」的内容，不要涉及其他疾病\n"
        else:
            answer_guide += "1. 直接回答用户的问题\n"
        answer_guide += (
            "2. 如有多种治疗方法/药物，请列出并简要说明\n"
            "3. 如有饮食建议，可一并提及\n"
            "4. 最后提醒用户：具体用药请遵医嘱\n"
            "5. 忠实性要求：答案中每个事实都必须能在上方参考资料中找到对应表述，"
            "不要输出参考资料中没有的疾病名、药名或机制描述"
        )
        prompt_parts.append(answer_guide)

        return "\n\n".join(prompt_parts)

    def _build_patient_query_prompt(
        self, query: str, merged_results: List[Dict], patient_name: str, entities: Dict[str, str]
    ) -> str:
        """患者病历查询专用提示词：只允许用患者病历片段回答，严禁通用科普。"""
        target_index = entities.get("指标", "") if entities else ""
        if not target_index:
            target_index = self._extract_target_indicator(query)

        # 只保留病历块（向量 report/document 块 + KG 患者节点结果）
        record_items = []
        for r in merged_results:
            if r.get("source") == "kg" and r.get("source_type", "").startswith("patient"):
                record_items.append(r)
                continue
            meta = r.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            ft = meta.get("file_type", "") if isinstance(meta, dict) else ""
            if ft in ("report", "document"):
                record_items.append(r)

        # 兜底：无病历数据时固定输出提示，禁止用通用知识搪塞
        if not record_items:
            return (
                "请严格按照下面这一句话逐字输出，不要添加任何解释、前缀或后缀：\n"
                f"未查询到【{patient_name}】的【{target_index}】相关病历数据。"
                "请确认患者姓名是否正确，或上传该患者的完整病历资料。"
            )

        system_hint = (
            "你是医疗助手。用户正在查询患者【" + patient_name + "】的个人病历指标【" + target_index + "】。\n"
            "【严格规则】\n"
            "1. 只能使用下方提供的该患者病历片段来回答，严禁使用任何通用医学知识、科普、注意事项。\n"
            "2. 直接给出病历中记录的具体数值/结果；若病历中未记录该指标，请明确说明未记录，禁止编造。\n"
            "3. 只输出病历中真实存在的内容；严禁输出与该患者病历无关的通用科普、注意事项。\n"
        )

        parts = [system_hint, "【患者病历片段】："]
        for i, item in enumerate(record_items, 1):
            content = item.get("content", "")
            # 去掉病历原文中的横线分隔符，避免 LLM 复现"======"这类丑格式
            content = re.sub(r'=+', '', content)
            content = re.sub(r'-{3,}', '', content)
            content = content.strip()
            if len(content) > 600:
                content = content[:600] + "..."
            parts.append(f"病历{i}：{content}")
        parts.append(f"【用户问题】{query}")

        # 输出格式要求：结构化医疗报告风格，禁止横线分割
        parts.append(
            "【输出格式要求】请用结构化的医疗报告风格回答，禁止使用\"=====\"、\"-----\"等纯字符横线分割线，"
            "改用 Markdown 标题、表格和列表组织内容，按以下模块输出（病历中缺失的模块直接省略）：\n"
            "📋 **患者基本信息**：姓名、性别、年龄等，用键值对列表展示\n"
            "🩺 **诊断结果**：用有序列表，主要诊断加粗\n"
            "📊 **关键指标**：用 Markdown 表格（指标名称 | 结果 | 参考范围/状态）\n"
            "📝 **现病史**：段落式，关键症状和时间点加粗\n"
            "💊 **用药处方**：用 Markdown 表格（药品名称 | 规格 | 用法用量 | 作用）\n"
            "🏥 **辅助检查**：化验/心电图/彩超等分点列出\n"
            "💡 **生活方式建议**：用列表展示"
        )
        return "\n\n".join(parts)


# ============================================================================
# 辅助函数
# ============================================================================

def _label_to_chinese(label: str) -> str:
    """Neo4j 节点标签 → 中文"""
    mapping = {
        "药品": "治疗药品",
        "食物": "饮食建议",
        "检查项目": "所需检查",
        "科目": "所属科室",
        "疾病症状": "常见症状",
        "治疗方法": "治疗方法",
        "疾病": "并发疾病",
    }
    return mapping.get(label, label)


def _rel_to_chinese(rel: str) -> str:
    """Neo4j 关系类型 → 中文"""
    mapping = {
        "疾病使用药品": "治疗药品",
        "疾病宜吃食物": "宜吃食物",
        "疾病忌吃食物": "忌吃食物",
        "疾病所需检查": "所需检查",
        "疾病所属科目": "所属科室",
        "疾病的症状": "常见症状",
        "治疗的方法": "治疗方法",
        "疾病并发疾病": "可能并发的疾病",
    }
    return mapping.get(rel, rel)


# ============================================================================
# 独立测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UnifiedRetriever 模块测试")
    print("=" * 60)

    # 测试 KG 结果格式转换
    print("\n[测试1] KG 搜索结果格式化")
    print("-" * 40)
    mock_kg_results = [
        {
            "id": "kg_感冒_疾病简介",
            "content": "关于感冒的疾病简介：感冒是一种常见的上呼吸道感染疾病...",
            "source": "kg",
            "_kg_score": 1.0,
            "source_type": "property",
            "entity": "感冒",
            "attribute": "疾病简介"
        },
        {
            "id": "kg_感冒_疾病使用药品",
            "content": "感冒的治疗药品包括：阿莫西林、布洛芬、对乙酰氨基酚",
            "source": "kg",
            "_kg_score": 1.0,
            "source_type": "relationship",
        }
    ]

    mock_vector_results = [
        {
            "id": "leaf_001",
            "content": "感冒患者应多休息，多饮水，可使用解热镇痛药缓解症状。",
            "score": 0.95,
            "source": "vector"
        },
        {
            "id": "leaf_002",
            "content": "对乙酰氨基酚是常用的退热药，成人剂量325-650mg/次。",
            "score": 0.88,
            "source": "vector"
        }
    ]

    retriever = UnifiedRetriever.__new__(UnifiedRetriever)  # skip __init__
    merged = retriever._merge_results(mock_kg_results, mock_vector_results)
    print(f"合并结果数: {len(merged)}")
    for r in merged:
        print(f"  [{r['source']}] {r['content'][:80]}...")

    # 测试 prompt 构建
    print("\n[测试2] 统一 Prompt 构建")
    print("-" * 40)
    prompt = retriever._build_unified_prompt(
        "感冒发烧吃什么药？", merged, {"疾病": "感冒"}
    )
    print(prompt[:800])
    print("...")

    # 测试辅助函数
    print("\n[测试3] 辅助函数")
    print("-" * 40)
    print(f"_label_to_chinese('药品') = {_label_to_chinese('药品')}")
    print(f"_rel_to_chinese('疾病的症状') = {_rel_to_chinese('疾病的症状')}")

    print("\n" + "=" * 60)
    print("UnifiedRetriever 测试完成！")
    print("=" * 60)
