import os
from pathlib import Path

def remove_folder_edges(graph_file: str, target_folder: str, output_file: str = "graph_updated.txt"):
    """
    从graph.txt中删除包含指定文件夹路径的所有边
    
    Args:
        graph_file: 原始图文件路径 (graph.txt)
        target_folder: 要删除相关边的目标文件夹路径
        output_file: 处理后的新图文件保存路径
    """
    # 标准化目标文件夹路径（兼容不同系统路径分隔符）
    target_folder = Path(target_folder).as_posix()
    # 存储保留的边
    remaining_edges = set()

    # 检查原始文件是否存在
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"图文件 {graph_file} 不存在")

    # 读取并过滤边
    with open(graph_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            
            try:
                # 拆分边的两个节点
                node1, node2 = line.split()
                # 标准化节点路径
                node1 = Path(node1).as_posix()
                node2 = Path(node2).as_posix()

                # 检查两个节点是否包含目标文件夹
                if target_folder in node1 or target_folder in node2:
                    print(f"删除第 {line_num} 行边: {line}")
                    continue
                
                # 保留不相关的边（去重）
                remaining_edges.add(tuple(sorted([node1, node2])))
            except ValueError:
                print(f"警告：第 {line_num} 行格式错误，跳过: {line}")
                continue

    # 写入处理后的边
    with open(output_file, 'w', encoding='utf-8') as f:
        for edge in sorted(remaining_edges):
            f.write(f"{edge[0]} {edge[1]}\n")

    print(f"\n处理完成！")
    print(f"原始边总数: {line_num} (含空行/错误行)")
    print(f"保留的边数: {len(remaining_edges)}")
    print(f"处理后的文件已保存至: {output_file}")

if __name__ == "__main__":
    # 配置参数
    GRAPH_FILE = "yunguo/graph.txt"          # 原始图文件路径
    TARGET_FOLDER = "yunguo/images/video_00044"  # 要删除的目标文件夹
    OUTPUT_FILE = "yunguo/graph_updated.txt" # 输出文件路径

    # 执行删除操作
    remove_folder_edges(
        graph_file=GRAPH_FILE,
        target_folder=TARGET_FOLDER,
        output_file=OUTPUT_FILE
    )