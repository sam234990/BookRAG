from typing import List, Optional, Dict, Any, Tuple, Union, Literal

from Core.provider.llm import LLM
from Core.provider.vlm import VLM

from Core.Index.Tree import TreeNode, NodeType
from Core.Common.Memory import Memory
from Core.Common.Message import Message
from Core.prompts.gbc_prompt import (
    ITER_GENERATION_SYS_PROMPT,
    ITER_GENERATION_USER_PROMPT,
    ITER_GENERATION_GRAPH,
    VLM_GENERATION_USER_PROMPT,
    SYNTHESIS_SYS_PROMPT,
    SYNTHESIS_USER_PROMPT,
)
from Core.utils.utils import num_tokens, TextProcessor
from Core.utils.table_utils import table2text
from Core.rag.gbc_plan import SubQuestion
import logging

log = logging.getLogger(__name__)


class AnswerAgent:
    def __init__(self, llm: LLM, vlm: VLM):
        self.llm = llm
        self.vlm = vlm

    def _prepare_evidence(
        self, retrieved_nodes: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Separates retrieved nodes into text-based and image-based categories.
        Tables are processed to be included in both.
        """
        image_nodes, text_nodes = [], []
        for node in retrieved_nodes:
            node_type = node.get("type", "text")
            node["page"] = node["page"] + 1  # make page start from 1
            if node_type == NodeType.IMAGE:
                image_nodes.append(node)
            elif node_type == NodeType.TABLE:
                node["content"] = table2text(node)
                # image_nodes.append(node)
                # llm_node_data = node.copy()
                text_nodes.append(node)
            else:
                text_nodes.append(node)
        return text_nodes, image_nodes

    def _build_prompts(
        self,
        query: str,
        text_nodes: List[Dict],
        image_nodes: List[Dict],
        graph_str: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Builds chunked and formatted prompts for both LLM and VLM.
        """
        # 1. Build VLM prompts for image-based evidence
        image_prompts = []
        for node in image_nodes:
            # 假设 page_number 已处理为从1开始
            img_path = node["img_path"]
            page = node.get("page", "-1")
            page = str(page) if isinstance(page, int) else page
            node_content = node.get("content", "")
            content = f"An image in Page: {page}, Caption: {node_content}"
            vlm_prompt = (
                f"{ITER_GENERATION_SYS_PROMPT.strip()}\n\n"
                f"{VLM_GENERATION_USER_PROMPT.format(question=query, content=content).strip()}"
            )
            if img_path:
                image_prompts.append({"prompt": vlm_prompt, "image_url": img_path})

        # 2. Build chunked LLM prompts for text-based evidence
        text_prompts = []

        # Build the graph part of the prompt only if graph_str is not empty
        graph_prompt_part = ""
        if graph_str:
            graph_prompt_part = ITER_GENERATION_GRAPH.format(
                knowledge_graph_subgraph=graph_str
            )

        # Calculate token budget for retrieved content
        base_prompt_tokens = num_tokens(
            ITER_GENERATION_USER_PROMPT.format(
                user_question=query, retrieved_content=""
            )
            + graph_prompt_part
        )
        system_prompt_tokens = num_tokens(ITER_GENERATION_SYS_PROMPT)
        content_limit = (
            self.llm.config.max_tokens - system_prompt_tokens - base_prompt_tokens - 400
        )  # 400 as buffer

        processed_nodes = []
        for node in text_nodes:
            node_content = node.get("content", "")
            node_type = node.get("type", "text")
            node_page = node.get("page", -1)
            node_text = (
                f"Type: {node_type} in Page: {node_page}\nContent: {node_content}\n"
            )
            full_node_tokens = num_tokens(node_text)

            if full_node_tokens > content_limit:
                sub_content_chunks = TextProcessor.split_text_into_chunks(
                    text=node_content, max_length=content_limit
                )
                for chunk_content in sub_content_chunks:
                    processed_nodes.append(
                        {
                            "content": chunk_content,
                            "type": node_type,
                            "page": node_page,
                        }
                    )
            else:
                processed_nodes.append(node)

        # Chunking logic
        current_chunk_str = ""
        current_chunk_tokens = 0
        separator = "\n\n---\n\n"

        for node in processed_nodes:
            node_content = node.get("content", "")
            node_type = node.get("type", "text")
            node_page = node.get("page", -1)
            node_text = (
                f"Type: {node_type} in Page: {node_page}\nContent: {node_content}\n"
            )
            node_tokens = num_tokens(node_text)

            if current_chunk_str and (
                current_chunk_tokens + node_tokens > content_limit
            ):

                user_prompt = (
                    ITER_GENERATION_USER_PROMPT.format(
                        user_question=query, retrieved_content=current_chunk_str
                    )
                    + graph_prompt_part
                )
                gen_memory = Memory()
                gen_memory.add(
                    Message(role="system", content=ITER_GENERATION_SYS_PROMPT)
                )
                gen_memory.add(Message(role="user", content=user_prompt))
                text_prompts.append(gen_memory)

                # 用当前节点开始一个新块
                current_chunk_str = node_text
                current_chunk_tokens = node_tokens
            else:
                # 将节点内容添加到当前块
                if not current_chunk_str:
                    current_chunk_str = node_text
                    current_chunk_tokens = node_tokens
                else:
                    current_chunk_str += separator + node_text
                    current_chunk_tokens += node_tokens

        if current_chunk_str:
            user_prompt = (
                ITER_GENERATION_USER_PROMPT.format(
                    user_question=query, retrieved_content=current_chunk_str
                )
                + graph_prompt_part
            )
            gen_memory = Memory()
            gen_memory.add(Message(role="system", content=ITER_GENERATION_SYS_PROMPT))
            gen_memory.add(Message(role="user", content=user_prompt))
            text_prompts.append(gen_memory)

        return text_prompts, image_prompts

    def _synthesize_from_chunks(
        self,
        query: str,
        text_prompts: List[str],
        image_prompts: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes generation on chunked prompts and synthesizes a final answer.
        """
        partial_answers = []

        # Generate answers from text prompts
        for i, memory in enumerate(text_prompts):
            try:
                answer = self.llm.get_completion(memory)
                partial_answers.append(
                    {"source": f"Text Chunk {i + 1}", "content": answer}
                )
            except Exception as e:
                partial_answers.append(
                    {
                        "source": f"Text Chunk {i + 1}",
                        "content": f"[Error generating from text: {e}]",
                    }
                )

        # Generate answers from image prompts
        for i, item in enumerate(image_prompts):
            try:
                answer = self.vlm.generate(item["prompt"], images=[item["image_url"]])
                partial_answers.append({"source": f"Image {i + 1}", "content": answer})
            except Exception as e:
                partial_answers.append(
                    {
                        "source": f"Image {i + 1}",
                        "content": f"[Error generating from image: {e}]",
                    }
                )

        # Synthesize the final answer
        if not partial_answers:
            final_answer = (
                "Based on the provided information, I couldn't find an answer."
            )
            return final_answer, partial_answers

        if len(partial_answers) == 1:
            final_answer = partial_answers[0]["content"]
            return final_answer, partial_answers

        partial_answers_str = "\n".join(
            [
                f"### Analysis from {res['source']}\n{res['content']}\n---"
                for res in partial_answers
            ]
        )

        log.info("Synthesizing the final answer from partial results...")
        synthesis_user_prompt = SYNTHESIS_USER_PROMPT.format(
            user_question=query, partial_answers_str=partial_answers_str
        )
        synthesis_memory = Memory()
        synthesis_memory.add(Message(role="system", content=SYNTHESIS_SYS_PROMPT))
        synthesis_memory.add(Message(role="user", content=synthesis_user_prompt))

        try:
            final_answer = self.llm.get_completion(synthesis_memory)
        except Exception as e:
            log.error(f"Error during final synthesis step: {e}")
            error_header = (
                "I was able to analyze the provided information in parts, but "
                "encountered an error while trying to synthesize the final answer. "
                f"Here are the partial analyses I found:\n\n---\n\n"
            )
            final_answer = error_header + partial_answers_str

        return final_answer, partial_answers

    def answer_simple_question(
        self, query: str, retrieved_nodes: List[Dict], entities: List[Dict] = None
    ) -> str:
        """
        Orchestrates answering a single, simple question by preparing evidence,
        building prompts, and synthesizing results.
        """
        # 1. Prepare evidence: Separate nodes and handle optional entities
        graph_str = ""
        if entities:
            graph_str = f"There are f{len(entities)} relevant entities:\n"
            for ent in entities:
                graph_str += (
                    f"- Name: {ent['entity_name']}, Type: {ent['entity_type']}\n"
                )

        text_nodes, image_nodes = self._prepare_evidence(retrieved_nodes)

        # 2. Build chunked prompts for LLM and VLM
        text_prompts, image_prompts = self._build_prompts(
            query, text_nodes, image_nodes, graph_str
        )

        # 3. Execute prompts and synthesize the final answer
        final_answer, partial_answers = self._synthesize_from_chunks(
            query, text_prompts, image_prompts
        )

        return final_answer, partial_answers

    def answer_complex_question(
        self,
        original_query: str,
        sub_question_plan: List[SubQuestion],
        sub_question_results: List[Dict[str, Any]],
    ) -> str:
        """
        Answers a complex question by synthesizing the answers from its sub-questions
        based on the full decomposition plan.

        Args:
            original_query: The original complex user query.
            sub_question_plan: The full list of SubQuestion objects (retrieval and synthesis).
            sub_question_results: A list of dictionaries with results for the 'retrieval' tasks.
                                  e.g., [{"question": "...", "answer": ("...", [...])}]
        Returns:
            The final, synthesized answer as a string.
        """

        if not sub_question_results:
            return "I could not find the necessary information to answer the complex question."

        # 1. Format the intermediate findings from the retrieval steps
        # The 'answer' is a tuple (final_answer, partial_answers), we need the first element
        intermediate_findings = "\n\n".join(
            [
                f"--- Finding for '{res['question']}' ---\n{res['answer']}"
                for res in sub_question_results
            ]
        )

        # 2. Find the synthesis step from the plan (it might not exist)
        synthesis_step = next(
            (sq for sq in sub_question_plan if sq.type == "synthesis"), None
        )

        # 3. Dynamically construct the final synthesis prompt
        prompt_template = """
You are an expert AI assistant that synthesizes information to answer a complex question. You have been provided with the original question and a set of findings from previous information retrieval steps. Your task is to use ONLY these findings to provide a final, cohesive answer.

--- ORIGINAL COMPLEX QUESTION ---
{original_query}

--- GATHERED FINDINGS ---
{intermediate_findings}
"""

        # --- MODIFICATION START ---
        # Add the final task only if a synthesis step exists
        if synthesis_step:
            prompt_template += """
--- FINAL TASK ---
Based on the findings above, please perform the following task:
"{synthesis_question}"

Final Answer:
"""
            synthesis_prompt = prompt_template.format(
                original_query=original_query,
                intermediate_findings=intermediate_findings,
                synthesis_question=synthesis_step.question,
            )
        else:
            # If no synthesis step, just ask for the final answer directly
            prompt_template += """
--- FINAL TASK ---
Based on the findings above, provide a comprehensive final answer to the original complex question.

Final Answer:
"""
            synthesis_prompt = prompt_template.format(
                original_query=original_query,
                intermediate_findings=intermediate_findings,
            )
            
        try:
            # 4. Call the LLM to get the final answer
            final_answer = self.llm.get_completion(synthesis_prompt)
        except Exception as e:
            log.error(f"Error during complex synthesis step: {e}")
            final_answer = (
                "I encountered an error while trying to synthesize the final answer from the gathered information. "
                f"Here are the intermediate findings I was able to collect:\n\n{intermediate_findings}"
            )

        return final_answer

    def _get_partial_analysis_for_node(
        self, original_query: str, operation: str, node: Dict
    ) -> str:
        """
        Generates a partial analysis for a single node in the context of a global query,
        respecting the model's token limits.
        """
        # Determine the instruction based on the operation
        operation_instruction = {
            "LIST": "Extract and list the key information from the evidence below that is relevant to the original question.",
            "SUMMARIZE": "Summarize the key points from the evidence below as they relate to the original question.",
            "ANALYZE": "Analyze the evidence below and state its contribution to answering the original question.",
        }.get(operation, "Analyze the following piece of evidence.")

        base_prompt_template = f"""
You are working on a larger query and your current task is to analyze a single piece of retrieved evidence. Your analysis will be combined with others later to form a final answer.

--- ORIGINAL GLOBAL QUERY ---
{original_query}

--- CURRENT TASK ---
{operation_instruction}

--- EVIDENCE TO ANALYZE ---
{{node_context}}

Your concise analysis of this single piece of evidence:
"""
        node_type = node.get("type", "text")
        # Determine which model to use and its context limit
        img_path = node.get("img_path", "")
        content = node.get("content", "")
        node["page"] = node.get("page", 0) + 1  # make page start from 1
        page = node.get("page", -1)

        use_vlm = node_type == NodeType.IMAGE and img_path
        model_max_tokens = (
            self.vlm.config.max_tokens if use_vlm else self.llm.config.max_tokens
        )

        # Calculate the available token budget for the node's content
        prompt_overhead = num_tokens(base_prompt_template.format(node_context=""))
        content_budget = (
            model_max_tokens - prompt_overhead - 400
        )  # 400 as a safety buffer

        # Prepare the node content, truncating if necessary
        # Assuming node.meta_info.content and node.type are the correct attributes
        if num_tokens(content) > content_budget:
            # Using your TextProcessor logic for consistency
            content = TextProcessor.split_text_into_chunks(
                text=content, max_length=content_budget
            )[0]

        node_context = f"Type: {node_type} in Page: {page}\nContent: {content}\n"

        # Final prompt construction
        prompt = base_prompt_template.format(node_context=node_context)

        # Call the appropriate model
        if use_vlm:
            # VLM is now used ONLY for IMAGE nodes
            return self.vlm.generate(prompt, images=[img_path])
        else:
            # LLM is used for TEXT and TABLE nodes
            return self.llm.get_completion(prompt)

    def answer_global_question(
        self,
        original_query: str,
        operation: Literal["LIST", "SUMMARIZE", "ANALYZE"],
        filtered_nodes: List[Dict],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Answers a global question by analyzing each filtered node individually
        and then synthesizing the results.
        """
        if not filtered_nodes:
            return (
                "I found no items matching the specified filters to answer the question.",
                [],
            )

        # 1. Map Step: Get partial analysis for each node individually
        partial_answers = []
        for i, node in enumerate(filtered_nodes):
            try:
                node_type = node.get("type", "text")
                page = node.get("page", 0) + 1
                source_str = f"Node {i + 1} (Type: {node_type}, Page: {page})"
                analysis = self._get_partial_analysis_for_node(
                    original_query, operation, node
                )
                partial_answers.append(
                    {
                        "source": source_str,
                        "content": analysis,
                    }
                )
            except Exception as e:
                partial_answers.append(
                    {
                        "source": source_str,
                        "content": f"[Error analyzing node: {e}]",
                    }
                )

        # 2. Reduce Step: Synthesize the partial analyses into a final answer
        if not partial_answers:
            return "No analysis could be generated from the filtered items.", []

        partial_answers_str = "\n".join(
            [
                f"### Analysis from {res['source']}\n{res['content']}\n---"
                for res in partial_answers
            ]
        )

        log.info("Synthesizing the final answer from partial results...")
        synthesis_user_prompt = SYNTHESIS_USER_PROMPT.format(
            user_question=original_query, partial_answers_str=partial_answers_str
        )
        synthesis_memory = Memory()
        synthesis_memory.add(Message(role="system", content=SYNTHESIS_SYS_PROMPT))
        synthesis_memory.add(Message(role="user", content=synthesis_user_prompt))

        try:
            final_answer = self.llm.get_completion(synthesis_memory)
        except Exception as e:
            log.error(f"Error during final synthesis step: {e}")
            error_header = (
                "I was able to analyze the provided information in parts, but "
                "encountered an error while trying to synthesize the final answer. "
                f"Here are the partial analyses I found:\n\n---\n\n"
            )
            final_answer = error_header + partial_answers_str

        return final_answer, partial_answers
