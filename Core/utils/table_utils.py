import html
import re
from bs4 import BeautifulSoup
from typing import List
import copy
from typing import Union, Any, List, Dict


def parse_html_table_to_grid(html_string: str) -> List[List[str]]:
    """ """
    if not html_string:
        return []

    soup = BeautifulSoup(html_string, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return []

    grid = []
    for r, row_elem in enumerate(rows):
        while len(grid) <= r:
            grid.append([])

        cells = row_elem.find_all(["td", "th"])
        for cell_elem in cells:
            # 找到当前行可以插入的第一个位置
            c = 0
            while len(grid[r]) > c and grid[r][c] is not None:
                c += 1

            colspan = int(cell_elem.get("colspan", 1))
            rowspan = int(cell_elem.get("rowspan", 1))
            text = cell_elem.get_text(strip=True)

            # 填充单元格，考虑合并情况
            for i in range(rowspan):
                for j in range(colspan):
                    row_idx, col_idx = r + i, c + j
                    # 确保目标行存在
                    while len(grid) <= row_idx:
                        grid.append([])
                    # 确保目标列存在
                    while len(grid[row_idx]) <= col_idx:
                        grid[row_idx].append(None)
                    # 只有当格子为空时才填充，防止覆盖
                    if grid[row_idx][col_idx] is None:
                        grid[row_idx][col_idx] = text
    return grid


def is_numeric(s: str) -> bool:
    """检查字符串是否可以被视为数字（包括带括号、百分号等情况）。"""
    if not isinstance(s, str):
        return False
    # 移除常见的非数值但可能出现在数值单元格的字符
    s = s.strip().replace("%", "").replace("(", "").replace(")", "").replace(",", "")
    if not s:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def identify_header_rows(grid: List[List[str]]) -> int:
    if not grid or len(grid) < 2:
        # 如果表格少于2行，无法进行比较，通常认为没有表头或只有一行表头
        return 0 if not grid else 1

    header_row_count = 0
    max_cols = max(len(row) for row in grid)

    # 逐行检查，直到找到第一个非表头行
    check_depth = min(3, len(grid) - 1)  # 最多检查前3行
    for i in range(check_depth):
        current_row = grid[i]
        next_row = grid[i + 1]

        # --- 启发式规则 1: 结构性线索 (colspan 证据) ---
        has_horizontal_span = False
        for j in range(len(current_row) - 1):
            # 如果相邻单元格内容相同且不为空，则认为有合并
            if current_row[j] and current_row[j] == current_row[j + 1]:
                has_horizontal_span = True
                break

        # --- 启发式规则 2: 内容类型差异 (与下一行比较) ---
        type_mismatch_count = 0
        text_to_numeric_count = 0

        # 确保比较时行列对齐
        for j in range(min(len(current_row), len(next_row))):
            cell_current = current_row[j]
            cell_next = next_row[j]

            is_current_numeric = is_numeric(cell_current)
            is_next_numeric = is_numeric(cell_next)

            if is_current_numeric != is_next_numeric:
                type_mismatch_count += 1
                # 这是一个非常强的信号
                if not is_current_numeric and is_next_numeric:
                    text_to_numeric_count += 1

        # 决策逻辑
        # 如果超过1/3的列发生了“文本->数字”的类型转换，这几乎肯定是表头
        is_strong_header_signal = text_to_numeric_count > (max_cols / 3)

        # 如果存在单元格合并，也是一个很强的表头信号
        is_structural_header = has_horizontal_span

        # 如果当前行与下一行在大多数列上类型不同，也算表头
        is_type_mismatch_header = type_mismatch_count > (max_cols / 2)

        if is_strong_header_signal or is_structural_header or is_type_mismatch_header:
            header_row_count += 1
        else:
            # --- Fallback Heuristic ---
            # 如果主要规则都失效了，并且我们在处理第一行，就启用后备规则
            if i == 0:
                is_current_row_all_text = all(
                    not is_numeric(c) for c in current_row if c
                )
                is_next_row_all_text = all(not is_numeric(c) for c in next_row if c)

                # 如果这是一个纯文本场景
                if is_current_row_all_text and is_next_row_all_text:
                    dissimilar_cells = 0
                    for j in range(min(len(current_row), len(next_row))):
                        # 检查单元格内容是否完全不同
                        if current_row[j] != next_row[j]:
                            dissimilar_cells += 1

                    # 如果超过一半的单元格内容都不同，我们有理由相信第一行是表头
                    if dissimilar_cells > max_cols / 2:
                        header_row_count += 1
                        # 继续检查下一行是否也可能是多级表头的一部分
                        continue

            # 如果所有规则（包括后备规则）都失败，则终止
            break

    return header_row_count


def create_hierarchical_headers(
    grid: List[List[str]], num_header_rows: int
) -> List[str]:
    column_headers = []
    if num_header_rows > 0 and grid and len(grid) >= num_header_rows:
        header_grid = grid[:num_header_rows]
        num_cols = len(header_grid[0]) if header_grid else 0

        for j in range(num_cols):
            path_parts = [header_grid[i][j].strip() for i in range(num_header_rows)]
            unique_ordered_parts = list(dict.fromkeys(p for p in path_parts if p))

            if unique_ordered_parts:
                full_header_path = " > ".join(unique_ordered_parts)
                column_headers.append(full_header_path)

    return sorted(list(set(column_headers)))


def contains_letters(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(re.search("[a-zA-Z]", text))


def intelligent_table_converter(
    table_data: dict, non_text_threshold: float = 0.5
) -> str:
    if not table_data.get("table_body", ""):
        return table_data.get("caption", "")

    soup = BeautifulSoup(table_data["table_body"], "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return table_data.get("caption", "")

    grid = []
    for r, row_elem in enumerate(rows):
        while len(grid) <= r:
            grid.append([])
        cells = row_elem.find_all(["td", "th"])
        for cell_elem in cells:
            c = 0
            while len(grid[r]) > c and grid[r][c] is not None:
                c += 1
            colspan = int(cell_elem.get("colspan", 1))
            rowspan = int(cell_elem.get("rowspan", 1))
            text = cell_elem.get_text(strip=True)
            for i in range(rowspan):
                for j in range(colspan):
                    row_idx, col_idx = r + i, c + j
                    while len(grid) <= row_idx:
                        grid.append([])
                    while len(grid[row_idx]) <= col_idx:
                        grid[row_idx].append(None)
                    if grid[row_idx][col_idx] is None:
                        grid[row_idx][col_idx] = text

    if not grid:
        return table_data.get("caption", "")

    all_cells = [cell for row in grid for cell in row if cell]
    if not all_cells:
        return table_data.get("caption", "")

    non_text_cells_count = sum(1 for cell in all_cells if not contains_letters(cell))
    non_text_ratio = non_text_cells_count / len(all_cells)

    output_parts = []
    caption = table_data.get("caption", "")
    if caption:
        output_parts.append(f"Table Caption: {caption}")

    if non_text_ratio >= non_text_threshold:
        all_labels = {cell for cell in all_cells if contains_letters(cell)}

        column_labels = {
            cell
            for r, row in enumerate(grid)
            if r < 2
            for cell in row
            if contains_letters(cell)
        }
        row_labels = {
            grid[r][c]
            for r in range(len(grid))
            for c in range(min(2, len(grid[r])))
            if contains_letters(grid[r][c])
        }

        other_labels = all_labels - column_labels - row_labels

        output_parts.append("Type: Data-heavy table (schema summary)")
        if column_labels:
            output_parts.append(
                f"Column Labels: {', '.join(sorted(list(column_labels)))}"
            )
        if row_labels:
            output_parts.append(f"Row Labels: {', '.join(sorted(list(row_labels)))}")
        if other_labels:
            output_parts.append(
                f"Other Labels: {', '.join(sorted(list(other_labels)))}"
            )

    else:
        output_parts.append("Type: Text-heavy table (full linearization)")
        header = grid[0]
        for i, row in enumerate(grid[1:]):
            row_description = f"Row {i+1}: "
            row_len = len(row)
            for idx, h in enumerate(header):
                if idx < row_len and h and row[idx]:
                    row_description += f"'{h}' is '{row[idx]}'; "
            output_parts.append(row_description.strip())

    footnote = table_data.get("footnote", "")
    if footnote:
        output_parts.append(f"Footnote: {footnote}")

    return "\n".join(output_parts)


def table2text(table_data: dict):
    output_parts = []

    caption = table_data.get("caption", "")
    if caption:
        output_parts.append(f"Caption: {caption}")

    table_body = table_data.get("table_body", "")
    if table_body:
        grid = parse_html_table_to_grid(table_body)
        num_header_rows = identify_header_rows(grid)
        column_headers = create_hierarchical_headers(grid, num_header_rows)

        header_str = "This table contains the following columns:\n"
        for col in column_headers:
            header_str += f" - {col}\n"
        output_parts.append(header_str)

        row_strings = [
            " | ".join(cell.strip() if cell else "" for cell in row) for row in grid
        ]
        output_parts.append("Table Body:\n" + "\n".join(row_strings))

    footnote = table_data.get("footnote", "")
    if footnote:
        output_parts.append(f"Footnote: {footnote}")

    return "\n".join(output_parts)


if __name__ == "__main__":
    numerical_table_json = {
        "caption": "Table 2: Main results on three datasets. We report average F1 and standard deviation. (S) means using SLM as backbone and (L) means using LLM as backbone. Previous SoTA on FewNERD and ACE is from ProtoQA, and on TACREV is from UniRE.",
        "table_body": '<table><tr><td rowspan="2"></td><td colspan="3">FewNERD (NER)</td><td colspan="3">TACREV (RE)</td><td colspan="3">ACE (ED)</td><td></td></tr><tr><td>5-shot</td><td>10-shot</td><td>20-shot</td><td>20-shot</td><td>50-shot</td><td>100-shot</td><td>5-shot</td><td>10-shot</td><td>20-shot</td><td></td></tr><tr><td rowspan="3">LLM</td><td>CODEX</td><td>53.8(0.5)</td><td>54.0(1.4)</td><td>55.9(0.5)</td><td>59.1(1.4)</td><td>60.3(2.4)</td><td>62.4(2.6)</td><td>47.1(1.2)</td><td>47.7(2.8)</td><td>47.9(0.5)</td></tr><tr><td>InstructGPT</td><td>53.6(-)</td><td>54.6(-)</td><td>57.2(-)</td><td>60.1(-)</td><td>58.3(-)</td><td>62.7(-)</td><td>52.9(-)</td><td>52.1(-)</td><td>49.3(-)</td></tr><tr><td>GPT-4</td><td>-</td><td>-</td><td>57.8(-)</td><td>-</td><td>-</td><td>59.3(-)</td><td>-</td><td>-</td><td>52.1(-)</td></tr><tr><td rowspan="3">SLM</td><td>Previous SoTA</td><td>59.4(1.5)</td><td>61.4(0.8)</td><td>61.9(1.2)</td><td>62.4(3.8)</td><td>68.5(1.6)</td><td>72.6(1.5)</td><td>55.1(4.6)</td><td>63.9(0.8)</td><td>65.8(2.0)</td></tr><tr><td>+ Ensemble (S)</td><td>59.6(1.7)</td><td>61.8(1.2)</td><td>62.6(1.0)</td><td>64.9(1.5)</td><td>71.9(2.2)</td><td>74.1(1.7)</td><td>56.9(4.7)</td><td>64.2(2.1)</td><td>66.5(1.7)</td></tr><tr><td>+ Rerank (S)</td><td>59.4(1.5)</td><td>61.0(1.7)</td><td>61.5(1.7)</td><td>64.2(3.3)</td><td>70.8(3.3)</td><td>74.3(3.2)</td><td>56.1(0.3)</td><td>64.0(1.0)</td><td>66.7(1.7)</td></tr><tr><td rowspan="8">SLM + LLM</td><td colspan="10">Vicuna-13B</td></tr><tr><td>+ Rerank (L)</td><td>60.0(1.8)</td><td>61.9(2.1)</td><td>62.2(1.4)</td><td>65.2(1.4)</td><td>70.8(1.6)</td><td>73.8(1.7)</td><td>56.9(4.0)</td><td>63.5(2.7)</td><td>66.0(2.6)</td></tr><tr><td>+ Ensemble (S) + Rerank (L)</td><td>59.9(0.7)</td><td>62.1(0.7)</td><td>62.8(1.1)</td><td>66.5(0.5)</td><td>73.6(1.4)</td><td>75.0(1.5)</td><td>57.9(5.2)</td><td>64.4(1.2)</td><td>66.2(2.4)</td></tr><tr><td>+ Rerank (L)</td><td>60.6(2.1)</td><td>62.7(0.8)</td><td>63.3(0.6)</td><td>66.8(2.6)</td><td>72.3(1.4)</td><td>75.4(1.5)</td><td>57.8(4.6)</td><td>65.3(1.7)</td><td>67.3(2.2)</td></tr><tr><td>+ Ensemble (S) + Rerank (L)</td><td>61.3(1.9)</td><td>63.2(0.9)</td><td>63.7(1.8)</td><td>68.9(1.3)</td><td>74.8(1.3)</td><td>76.8(1.2)</td><td>59.5(3.7)</td><td>65.3(1.9)</td><td>67.8(2.1)</td></tr><tr><td colspan="10">GPT-4</td></tr><tr><td>+ Rerank (L)</td><td>60.8(2.3)</td><td>62.6(2.7)</td><td>63.0(1.3)</td><td>65.9(2.7)</td><td>72.3(0.3)</td><td>74.5(1.5)</td><td>59.6(2.9)</td><td>64.9(2.5)</td><td>67.1(2.5)</td></tr><tr><td>+ Ensemble (S) + Rerank (L)</td><td>61.1(2.2)</td><td>62.8(0.9)</td><td>63.6(1.2)</td><td>68.6(1.3)</td><td>73.9(1.4)</td><td>75.9(2.4)</td><td>60.9(3.9)</td><td>65.6(1.5)</td><td>67.8(1.7)</td></tr></table>',
    }
    print(intelligent_table_converter(numerical_table_json))
    print("\n" + "=" * 50 + "\n")

    textual_table_json = {
        "caption": "Table 15: Examples of the samples corrected by LLM reranking. We sample four examples from NER, RE and ED tasks, respectively. Sentences: The sentences in which samples locate. We color the samples (entities or trigger words) to be identified. Before: The prediction before LLM reranking. Based on SLM-based methods. After: The reranked prediction using LLMs. Rationales: LLM-generated Explanations.",
        "footnote": "",
        "table_body": '<table><tr><td>Sentences (entity/event)</td><td>Before</td><td>After</td><td>Rationales</td></tr><tr><td rowspan="2">According to it, King Eleusis and Cothonea (Cyntinia), parents of Trip-tolemus, are visited by Demeter, who rears their son, feeding him divine milk by day and placing him into the fire at night, which makes Triptolemus grow faster than mortal children normally do.</td><td colspan="2">Named Entity Recognition</td><td rowspan="2">According to the context, Triptolemus is a figure in Greek mythology, the son of King Eleusis and Cothonea (Cyntinia). He was a demigod who was sent by Demeter to teach mankind the art of agriculture. So Triptole-mus is a god in some legend/religious story.</td></tr><tr><td colspan="2">Relation Extraction</td></tr><tr><td>Survivors include his wife, Sandra; four sons, Jeff, James, Douglas and Harris; a daughter, Leslie; his mother, Sally; and two brothers, Guy and Paul.</td><td>None</td><td>per:children</td><td>The sentence states &#x27;Survivors include his wife, Sandra; four sons, Jeff, James, Douglas and Harris; a daughter, Leslie; his mother, Sally; and two brothers, Guy and Paul.&#x27; This indicates &#x27;his&#x27; is the father of Jeff, James, Douglas and Harris, and the husband of Sandra. So his is the parent of Jeff.</td></tr><tr><td>Eliza Samudio, Bruno&#x27;s ex-girlfriend and his 4-month-old baby boy&#x27;s mother, disappeared from her hometown of Belo Horizonte, Minas Gerais.</td><td>per:cities_of_residence</td><td>per:states_of_residence</td><td>The sentence states that &#x27;Eliza Samudio, Bruno&#x27;s ex-girlfriend and his 4-month-old baby boy&#x27;s mother, disappeared from her hometown of Belo Horizonte, Minas Gerais.&#x27; This indicates that Eliza Samudio is from the city of Belo Horizonte, which is located in the state of Minas Gerais. So Eliza Samudio lives in the state or province Minas Gerais.</td></tr><tr><td colspan="4">Event Detection</td></tr><tr><td>We have fought every institution, every law firm... it&#x27;s cost us a fortune but now we&#x27;re going on a massive expansion.</td><td>Conflict Attack</td><td>None</td><td>In the given sentence, the word fought is used to describe the action of the speaker fighting against various institutions and law firms. This does not involve any physical violence or court proceedings, so the word fought does not trigger any known event.</td></tr></table>',
    }
    print(intelligent_table_converter(textual_table_json))
