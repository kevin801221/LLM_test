#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sqlite3
import json
from pathlib import Path

def find_db_files(start_dir):
    """查找所有 .db 文件"""
    db_files = []
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.db'):
                db_files.append(os.path.join(root, file))
    return db_files

def check_sqlite_file(file_path):
    """檢查文件是否為有效的 SQLite 數據庫"""
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return len(tables) > 0
    except sqlite3.Error:
        return False

def read_sqlite_tables(file_path):
    """讀取 SQLite 數據庫中的所有表格和數據"""
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        
        # 獲取所有表格
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        results = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [column[1] for column in cursor.fetchall()]
            
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 100;")
            rows = cursor.fetchall()
            
            table_data = []
            for row in rows:
                row_dict = {}
                for i, column in enumerate(columns):
                    row_dict[column] = row[i]
                table_data.append(row_dict)
            
            results[table_name] = {
                "columns": columns,
                "data": table_data
            }
        
        conn.close()
        return results
    except sqlite3.Error as e:
        return {"error": str(e)}

def check_memory_in_json_files(start_dir):
    """檢查 JSON 文件中是否包含對話記憶"""
    memory_files = []
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 檢查是否包含對話相關的關鍵字
                        file_content = json.dumps(data).lower()
                        if any(keyword in file_content for keyword in ['conversation', 'memory', 'chat', 'message', 'dialogue']):
                            memory_files.append(file_path)
                except:
                    pass
    return memory_files

def main():
    start_dir = "D:\\YCM-Smart-Access-Control-System"
    
    print("=== 查找數據庫文件 ===")
    db_files = find_db_files(start_dir)
    
    if not db_files:
        print("未找到任何 .db 文件")
    else:
        print(f"找到 {len(db_files)} 個 .db 文件:")
        for file in db_files:
            print(f"- {file}")
            if check_sqlite_file(file):
                print("  (有效的 SQLite 數據庫)")
                tables_data = read_sqlite_tables(file)
                print(f"  包含 {len(tables_data)} 個表格:")
                for table, data in tables_data.items():
                    print(f"    - {table}: {len(data['data'])} 行")
                    if len(data['data']) > 0:
                        print(f"      列: {', '.join(data['columns'])}")
                        print(f"      第一行數據: {data['data'][0]}")
            else:
                print("  (不是有效的 SQLite 數據庫)")
    
    print("\n=== 查找可能包含對話記憶的 JSON 文件 ===")
    memory_files = check_memory_in_json_files(start_dir)
    
    if not memory_files:
        print("未找到可能包含對話記憶的 JSON 文件")
    else:
        print(f"找到 {len(memory_files)} 個可能包含對話記憶的 JSON 文件:")
        for file in memory_files:
            print(f"- {file}")
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        print(f"  包含 {len(data)} 個項目")
                        print(f"  第一個項目: {json.dumps(data[0], ensure_ascii=False)[:200]}...")
                    elif isinstance(data, dict):
                        print(f"  包含 {len(data)} 個鍵")
                        print(f"  鍵: {', '.join(list(data.keys())[:5])}...")
            except Exception as e:
                print(f"  無法讀取: {e}")

if __name__ == "__main__":
    main()
