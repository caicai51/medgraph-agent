"""
一次性脚本：将病历文件导入 Neo4j 患者节点（Patient），用于验证患者病历 KG 查询。

用法：
    python import_patient_to_neo4j.py [病历文件路径] [user_id]

默认路径为项目外的测试样例：
    ../test_data/sample_medical_record.txt
默认 user_id 为 admin。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_handler import parse_medical_record, import_patient_to_neo4j


def main():
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_data", "sample_medical_record.txt",
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        sys.exit(1)

    with open(path, "rb") as f:
        content = f.read()

    print(f"解析病历: {path}")
    record = parse_medical_record(content, os.path.basename(path))
    print("\n=== 解析结果 ===")
    print("患者信息:", record.get("patient_info", {}))
    print("关键指标:", record.get("indicators", {}))
    print("诊断:", record.get("diagnosis", []))
    print("疾病实体:", record.get("diseases", []))
    print("药品:", record.get("prescription", []))
    print("检查项目:", record.get("examination", []))

    print("\n=== 写入 Neo4j ===")
    user_id = sys.argv[2] if len(sys.argv) > 2 else "admin"
    ok = import_patient_to_neo4j(record, user_id=user_id)
    if ok:
        print("[OK] 患者节点已写入 Neo4j，患者病历查询时可被 KG 召回")
    else:
        print("[WARN] 写入失败，请检查 Neo4j 连接与 .env 中的 NEO4J_PASSWORD")


if __name__ == "__main__":
    main()
