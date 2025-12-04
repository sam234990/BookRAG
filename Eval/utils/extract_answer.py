from openai import OpenAI


def load_prompt():
    PROMPT_PATH = "./Eval/utils/prompt.md"
    with open(PROMPT_PATH, "r") as file:
        prompt = file.read()
    return prompt


class AnswerExtractor:
    def __init__(self, api_config_path="./Eval/utils/api.txt"):
        print("Initializing AnswerExtractor and reading API config...")
        with open(api_config_path, "r") as f:
            lines = f.readlines()
            base_url = lines[0].strip()  
            api_key = lines[1].strip()
            model_name = lines[2].strip()

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        print("Client created successfully.")
        print(f"Using model: {self.model_name}")
        self.system_prompt = load_prompt()

    def parse_evaluation_output(self, llm_output: str) -> dict:
        """
        将LLM评估器的结构化输出解析为Python字典。

        Args:
            llm_output: LLM返回的原始字符串响应。

        Returns:
            一个包含'extracted results', 'format', 'score'键的字典。
        """
        parsed_data = {}
        expected_keys = ["extracted results", "format", "score"]

        lines = llm_output.strip().split("\n")

        for line in lines:
            # 只在包含冒号的行进行处理
            if ":" in line:
                key, value = line.split(":", 1)

                clean_key = key.strip().lower()
                clean_value = value.strip()

                if clean_key in expected_keys:
                    parsed_data[clean_key] = clean_value

        for key in expected_keys:
            if key not in parsed_data:
                print(f"Warning: Key '{key}' not found in LLM output.")
                print(f"LLM Output was:\n{llm_output}\n")
                if key == "score":
                    parsed_data[key] = 0.0  # 默认分数为0.0
                else:
                    parsed_data[key] = "Failed to extract"

        return parsed_data

    def extract(self, question, output, correct_answer, model_name=None):
        """
        使用已创建的客户端来调用API。
        """
        model = model_name if model_name else self.model_name

        input_content = f"""\n\n
Question: {question}
Correct Answer: {correct_answer}
Model Response: {output}
"""

        messages = [
            {
                "role": "user",
                "content": self.system_prompt,
            },
            {
                "role": "assistant",
                "content": input_content,
            },
        ]
        parameters = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
            # "extra_body": {"enable_thinking": False},
        }
        if "qwen" in model.lower():
            parameters["extra_body"] = {"enable_thinking": False}

        try:
            response = self.client.chat.completions.create(**parameters)
            llm_res = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Failed to LLM response", "Failed", "Failed", 0.0

        parse_res = self.parse_evaluation_output(llm_res)
        score = parse_res.get("score", 0.0)
        pred_answer = parse_res.get("extracted results", "Failed to extract")
        pred_format = parse_res.get("format", "Failed to extract")
        return llm_res, pred_answer, pred_format, float(score)


if __name__ == "__main__":
    question = "What's the value of the smallest bar in the graph about the percentage of U.S. adults with Hispanic ancestry who self-identify as Hispanic or Non-Hispanic?"
    output = """### Step-by-Step Explanation:\n\n1. **Understanding the Question**: The user is asking for the value of the smallest bar in a graph that shows the percentage of U.S. adults with Hispanic ancestry who self-identify as Hispanic or Non-Hispanic. However, the context provided does not include a graph or any visual data.\n\n2. **Analyzing the Context**: The context includes a title and a short text. The title is \"Hispanic Identity Fades Across Generations as Immigrant Connections Fall Away,\" and the text states, \"11% of American adults with Hispanic ancestry do not identify as Hispanic.\" This suggests that 11% of U.S. adults with Hispanic ancestry self-identify as Non-Hispanic.\n\n3. **Interpreting the Data**: Since the context does not provide a graph, we cannot directly identify the smallest bar. However, the text gives us a percentage (11%) that represents the percentage of U.S. adults with Hispanic ancestry who self-identify as Non-Hispanic. This is likely the smallest bar in the graph, as it represents the minority group in the comparison.\n\n4. **Conclusion**: Based on the information provided, the smallest bar in the graph is likely the 11% of U.S. adults with Hispanic ancestry who self-identify as Non-Hispanic.\n\n---\n\nFinal Answer: The value of the smallest bar in the graph is 11%."""
    correct_answer = "5%"

    extractor = AnswerExtractor()
    extract_answer = extractor.extract(question, output, correct_answer)
    print("Extracted Answer in this example------\n", extract_answer)
