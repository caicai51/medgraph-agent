"""
文件处理模块 — 解析上传文件并分块、向量化

支持：
  - 病历上传（PDF、DOCX、TXT、图片OCR）
  - 文档上传（PDF、DOCX、TXT、MD）
"""

import os
import io
import json
import uuid
import re
from typing import List, Dict, Any, Optional
from datetime import datetime


# ============================================================================
# 文件类型识别
# ============================================================================

FILE_TYPE_MAP = {
    # 病历相关
    "report": {
        "label": "病历报告",
        "extensions": [".pdf", ".docx", ".doc", ".txt", ".jpg", ".jpeg", ".png"],
        "collection": "scenario_memory",
    },
    # 文档相关
    "document": {
        "label": "医学文档",
        "extensions": [".pdf", ".docx", ".doc", ".txt", ".md"],
        "collection": "semantic_memory",
    },
}


def detect_file_type(filename: str) -> str:
    """检测文件类型"""
    ext = os.path.splitext(filename)[1].lower()
    
    # 检查是否是图片类型（病历照片）
    if ext in [".jpg", ".jpeg", ".png"]:
        return "report"
    
    # 根据扩展名匹配
    for type_key, type_info in FILE_TYPE_MAP.items():
        if ext in type_info["extensions"]:
            return type_key
    
    return "document"  # 默认作为文档处理


def get_collection_for_type(file_type: str) -> str:
    """获取对应的集合名称"""
    type_info = FILE_TYPE_MAP.get(file_type, FILE_TYPE_MAP["document"])
    return type_info["collection"]


# ============================================================================
# 文本解析
# ============================================================================

def parse_txt(content: bytes) -> str:
    """解析TXT文件"""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("gbk", errors="ignore")


def parse_json(content: bytes) -> str:
    """解析JSON文件"""
    try:
        data = json.loads(content)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return parse_txt(content)


def parse_pdf(content: bytes) -> str:
    """解析PDF文件，提取文本内容（从字节流构造）"""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        text = "\n".join(text_parts).strip()
        if not text:
            raise ValueError("PDF内容为空（可能是扫描件，无法提取文本）")
        return text
    except ImportError:
        raise ValueError("PDF解析库未安装，请安装PyPDF2")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"PDF解析失败: {str(e)}（加密或扫描件PDF不支持）")


def parse_docx(content: bytes) -> str:
    """解析DOCX文件，提取段落文本（从字节流构造）"""
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        # 也提取表格内容
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        text = "\n".join(paragraphs).strip()
        if not text:
            raise ValueError("DOCX内容为空")
        return text
    except ImportError:
        raise ValueError("DOCX解析库未安装，请安装python-docx")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"DOCX解析失败: {str(e)}")


def parse_generic(content: bytes, filename: str) -> str:
    """通用解析（基于文件扩展名）"""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".json":
        return parse_json(content)
    elif ext in [".txt", ".md", ".csv", ".log"]:
        return parse_txt(content)
    elif ext == ".pdf":
        return parse_pdf(content)
    elif ext == ".docx":
        return parse_docx(content)
    elif ext == ".doc":
        # 老式 .doc 格式 python-docx 不支持，明确报错
        raise ValueError("不支持老式 .doc 格式，请转换为 .docx 或 .txt 后上传")
    else:
        # 未知类型，尝试当文本解析
        try:
            return parse_txt(content)
        except:
            raise ValueError(f"不支持的文件类型: {ext}")


# ============================================================================
# 文本分块
# ============================================================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    file_type: str = "document"
) -> List[Dict[str, Any]]:
    """
    将长文本切分为多个语义完整的块
    
    Args:
        text: 原始文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 相邻块的重叠字符数
        file_type: 文件类型（影响分块策略）
    
    Returns:
        分块列表，每个块包含 id, content, metadata
    """
    if not text:
        return []
    
    chunks = []
    
    # 按照段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 如果段落本身就超过 chunk_size，按句子分割
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[。！？.])', para)
            for sent in sentences:
                if not sent.strip():
                    continue
                if len(current_chunk) + len(sent) > chunk_size:
                    if current_chunk:
                        chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
                        chunks.append({
                            "id": chunk_id,
                            "content": current_chunk.strip(),
                            "chunk_index": chunk_index,
                        })
                        chunk_index += 1
                    current_chunk = sent
                else:
                    current_chunk += sent
        else:
            if len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk:
                    chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
                    chunks.append({
                        "id": chunk_id,
                        "content": current_chunk.strip(),
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
    
    # 添加最后一个块
    if current_chunk:
        chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
        chunks.append({
            "id": chunk_id,
            "content": current_chunk.strip(),
            "chunk_index": chunk_index,
        })
    
    # 为每个块添加基础 metadata
    for chunk in chunks:
        chunk["metadata"] = {
            "file_type": file_type,
            "chunk_index": chunk["chunk_index"],
            "total_chunks": len(chunks),
            "created_at": datetime.now().isoformat(),
        }
    
    return chunks


# ============================================================================
# 病历专用解析
# ============================================================================

# 常见疾病名（用于从诊断/既往史文本中抽取疾病实体）
KNOWN_DISEASES = [
    "高血压", "糖尿病", "高脂血症", "高血脂", "冠心病", "脑卒中", "脑梗",
    "高尿酸血症", "痛风", "脂肪肝", "肝硬化", "慢性肾炎", "肾病",
    "胃炎", "胃溃疡", "甲状腺功能亢进", "甲亢", "甲状腺功能减退", "甲减",
    "贫血", "心律失常", "心衰", "心肌梗死", "心绞痛", "糖耐量异常",
]

# 常见检查项目名（用于从辅助检查文本中抽取检查实体）
KNOWN_EXAMS = [
    "心电图", "心脏彩超", "彩超", "血常规", "尿常规", "肝功能", "肾功能",
    "血脂", "血糖", "肌酐", "胆固醇", "X光", "CT", "MRI", "超声", "化验",
]


def _extract_first(pattern: str, text: str) -> str:
    """提取第一个正则匹配值"""
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def _extract_diseases(text: str) -> List[str]:
    """从文本中抽取已知疾病名（去重保序），并排除"无/否认/排除"等否定语境。"""
    # 去掉否定片段（如"无冠心病、脑卒中病史"），避免把未患疾病误判为已患
    negated_text = re.sub(r'无[^，,。；;\n]{0,20}(?:病史|史)', '', text)
    negated_text = re.sub(r'(?:否认|排除|未见)[^，,。；;\n]{0,10}', '', negated_text)
    found = []
    for d in KNOWN_DISEASES:
        if d in negated_text and d not in found:
            found.append(d)
    return found


def _extract_exams(text: str) -> List[str]:
    """从文本中抽取已知检查项目（去重保序）"""
    found = []
    for e in KNOWN_EXAMS:
        if e in text and e not in found:
            found.append(e)
    return found


def parse_medical_record(content: bytes, filename: str) -> Dict[str, Any]:
    """
    解析医疗病历，提取结构化信息
    
    Returns:
        包含患者信息、诊断、处方等的结构化字典
    """
    text = parse_generic(content, filename)
    
    # 尝试提取病历关键信息
    record = {
        "patient_info": {},
        "diagnosis": [],
        "prescription": [],
        "examination": [],
        "diseases": [],
        "indicators": {},
        "content": text,
    }
    
    # 提取患者姓名
    name_patterns = [
        r'(?:患者姓名|姓名|Name)\s*[:：]\s*([^\s,，\n]+)',
        r'^([\u4e00-\u9fa5]{2,4})\s*(?:男|女)',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            record["patient_info"]["name"] = match.group(1).strip()
            break
    
    # 提取性别（全文精确匹配"性别：男/女"）
    gender_match = re.search(r'性别\s*[:：]\s*([男女])', text)
    if gender_match:
        record["patient_info"]["gender"] = gender_match.group(1)

    # 提取年龄
    age = _extract_first(r'年龄\s*[:：]\s*(\d+)', text)
    if age:
        record["patient_info"]["age"] = age

    # 提取关键指标（体格检查 / 辅助检查）
    indicators = {
        "血压": _extract_first(r'血压\s*[:：为]?\s*(\d{2,3}\s*/\s*\d{2,3})', text),
        "血糖": _extract_first(r'血糖\s*[:：]\s*([\d.]+)', text),
        "体温": _extract_first(r'体温\s*[:：]\s*([\d.]+)', text),
        "脉搏": _extract_first(r'脉搏\s*[:：]\s*(\d+)', text),
        "身高": _extract_first(r'身高\s*[:：]\s*(\d+)', text),
        "体重": _extract_first(r'体重\s*[:：]\s*(\d+)', text),
        "总胆固醇": _extract_first(r'总胆固醇\s*[:：]\s*([\d.]+)', text),
        "肌酐": _extract_first(r'肌酐\s*[:：]\s*([\d.]+)', text),
    }
    record["indicators"] = {k: v for k, v in indicators.items() if v}
    
    # 提取诊断
    diagnosis_patterns = [
        r'(?:诊断|诊断结果|初步诊断)\s*[:：]\s*([^\n]+)',
    ]
    for pattern in diagnosis_patterns:
        matches = re.findall(pattern, text)
        record["diagnosis"].extend([m.strip() for m in matches if m.strip()])

    # 提取疾病实体（从诊断 + 既往史）
    record["diseases"] = _extract_diseases(text)

    # 提取检查项目（从辅助检查）
    record["examination"] = _extract_exams(text)
    
    # 提取药品
    drug_patterns = [
        r'(?:用药|处方|药品)\s*[:：]\s*([^\n]+)',
        r'([\u4e00-\u9fa5]+(?:片|胶囊|颗粒|注射液|口服液|散|丸))',
    ]
    for pattern in drug_patterns:
        matches = re.findall(pattern, text)
        record["prescription"].extend([m.strip() for m in matches if m.strip()])
    
    return record


def medical_record_to_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将结构化病历转换为可分块的文本列表
    """
    chunks = []
    
    # 患者信息块
    if record["patient_info"]:
        info_text = f"患者信息："
        for key, value in record["patient_info"].items():
            info_text += f"{key}: {value}; "
        chunks.append({
            "id": f"chunk_{uuid.uuid4().hex[:8]}",
            "content": info_text,
            "chunk_index": len(chunks),
            "metadata": {
                "file_type": "report",
                "section": "patient_info",
                "created_at": datetime.now().isoformat(),
            }
        })

    # 关键指标块（血压/血糖/体温等）
    if record["indicators"]:
        ind_text = f"关键指标：" + "；".join(
            f"{k}: {v}" for k, v in record["indicators"].items()
        )
        chunks.append({
            "id": f"chunk_{uuid.uuid4().hex[:8]}",
            "content": ind_text,
            "chunk_index": len(chunks),
            "metadata": {
                "file_type": "report",
                "section": "indicators",
                "created_at": datetime.now().isoformat(),
            }
        })

    # 诊断信息块
    if record["diagnosis"]:
        diag_text = f"诊断结果：{'；'.join(record['diagnosis'])}"
        chunks.append({
            "id": f"chunk_{uuid.uuid4().hex[:8]}",
            "content": diag_text,
            "chunk_index": len(chunks),
            "metadata": {
                "file_type": "report",
                "section": "diagnosis",
                "created_at": datetime.now().isoformat(),
            }
        })
    
    # 处方信息块
    if record["prescription"]:
        rx_text = f"用药处方：{'；'.join(record['prescription'][:10])}"
        chunks.append({
            "id": f"chunk_{uuid.uuid4().hex[:8]}",
            "content": rx_text,
            "chunk_index": len(chunks),
            "metadata": {
                "file_type": "report",
                "section": "prescription",
                "created_at": datetime.now().isoformat(),
            }
        })
    
    # 完整文本分块
    full_text_chunks = chunk_text(record["content"], chunk_size=500, file_type="report")
    for chunk in full_text_chunks:
        chunk["metadata"].update({
            "file_type": "report",
            "section": "full_text",
        })
    chunks.extend(full_text_chunks)
    
    return chunks


def import_patient_to_neo4j(record: Dict[str, Any], user_id: str = "") -> bool:
    """将结构化病历写入 Neo4j：创建 Patient 节点及关联的疾病/药品/检查节点。

    Patient 节点标签 :Patient，用 name 属性标识；关系：
      - (:Patient)-[:HAS_DISEASE]->(:疾病)
      - (:Patient)-[:TAKES_DRUG]->(:药品)
      - (:Patient)-[:HAS_EXAM]->(:检查项目)
    """
    try:
        import py2neo
        from dotenv import load_dotenv
        load_dotenv()

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "neo4j")
        graph = py2neo.Graph(uri, auth=(user, password))

        if not str(user_id).strip():
            print("[Neo4j] 缺少 user_id，拒绝写入未归属的患者节点")
            return False

        user_id = str(user_id).strip()
        patient_info = record.get("patient_info", {})
        name = patient_info.get("name", "")
        if not name:
            print("[Neo4j] 病历缺少患者姓名，跳过患者节点入库")
            return False

        indicators = record.get("indicators", {})
        props = {
            "name": name,
            "性别": patient_info.get("gender", ""),
            "年龄": patient_info.get("age", ""),
            "血压": indicators.get("血压", ""),
            "血糖": indicators.get("血糖", ""),
            "体温": indicators.get("体温", ""),
            "脉搏": indicators.get("脉搏", ""),
            "身高": indicators.get("身高", ""),
            "体重": indicators.get("体重", ""),
            "总胆固醇": indicators.get("总胆固醇", ""),
            "肌酐": indicators.get("肌酐", ""),
            "诊断": "；".join(record.get("diagnosis", [])),
            "user_id": user_id,
        }
        # 去掉空值，避免写入空属性
        props = {k: str(v) for k, v in props.items() if v not in (None, "")}

        graph.run(
            "MERGE (p:Patient {name: $name, user_id: $user_id}) SET p += $props",
            name=name, user_id=user_id, props=props,
        )

        # 关系：HAS_DISEASE -> 疾病
        for disease in record.get("diseases", []):
            graph.run(
                "MERGE (d:疾病 {名称: $disease}) "
                "MERGE (p:Patient {name: $name, user_id: $user_id}) "
                "MERGE (p)-[:HAS_DISEASE]->(d)",
                name=name, user_id=user_id, disease=disease,
            )

        # 关系：TAKES_DRUG -> 药品
        for drug in record.get("prescription", []):
            graph.run(
                "MERGE (d:药品 {名称: $drug}) "
                "MERGE (p:Patient {name: $name, user_id: $user_id}) "
                "MERGE (p)-[:TAKES_DRUG]->(d)",
                name=name, user_id=user_id, drug=drug,
            )

        # 关系：HAS_EXAM -> 检查项目
        for exam in record.get("examination", []):
            graph.run(
                "MERGE (e:检查项目 {名称: $exam}) "
                "MERGE (p:Patient {name: $name, user_id: $user_id}) "
                "MERGE (p)-[:HAS_EXAM]->(e)",
                name=name, user_id=user_id, exam=exam,
            )

        print(
            f"[Neo4j] 患者节点入库成功: {name}"
            f"（疾病{len(record.get('diseases', []))}，"
            f"药品{len(record.get('prescription', []))}，"
            f"检查{len(record.get('examination', []))}）"
        )
        return True
    except Exception as e:
        print(f"[Neo4j] 患者节点入库失败: {e}")
        return False


# ============================================================================
# 主处理函数
# ============================================================================

def process_uploaded_file(
    filename: str,
    content: bytes,
    user_id: str,
    file_type_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    处理上传的文件，返回处理结果
    
    Args:
        filename: 原始文件名
        content: 文件内容（字节）
        user_id: 用户ID
        file_type_hint: 文件类型提示（report/document），为None时自动检测
    
    Returns:
        处理结果，包含 chunks 和 metadata
    """
    # 确定文件类型
    file_type = file_type_hint or detect_file_type(filename)
    collection = get_collection_for_type(file_type)
    
    document_id = str(uuid.uuid4())
    processed_at = datetime.now().isoformat()
    
    result = {
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "collection": collection,
        "user_id": user_id,
        "processed_at": processed_at,
        "chunks": [],
        "total_chunks": 0,
        "status": "success",
    }
    
    try:
        # 解析文件内容
        if file_type == "report":
            # 病历：尝试结构化解析
            record = parse_medical_record(content, filename)
            chunks = medical_record_to_chunks(record)
            # 同步写入 Neo4j 患者节点（失败不影响向量入库）
            import_patient_to_neo4j(record, user_id)
        else:
            # 文档：直接分块
            text = parse_generic(content, filename)
            chunks = chunk_text(text, chunk_size=500, file_type=file_type)
        
        # 添加公共 metadata
        for chunk in chunks:
            chunk["metadata"].update({
                "document_id": document_id,
                "filename": filename,
                "user_id": user_id,
            })
        
        result["chunks"] = chunks
        result["total_chunks"] = len(chunks)
        
        if not chunks:
            result["status"] = "empty"
            result["message"] = "文件内容为空或无法解析"
    
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"处理失败: {str(e)}"
    
    return result


def get_file_extensions_for_type(file_type: str) -> List[str]:
    """获取文件类型对应的扩展名列表"""
    type_info = FILE_TYPE_MAP.get(file_type, {})
    return type_info.get("extensions", [])


def get_file_types_info() -> List[Dict[str, Any]]:
    """获取所有支持的文件类型信息"""
    return [
        {
            "key": key,
            "label": info["label"],
            "extensions": info["extensions"],
            "collection": info["collection"],
        }
        for key, info in FILE_TYPE_MAP.items()
    ]
