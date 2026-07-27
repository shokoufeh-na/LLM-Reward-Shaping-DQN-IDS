# src/llm_critic.py

import ollama
import re


class LLMCritic:
    """
    LLM Security Critic

    Uses an LLM to estimate the suspiciousness
    of a network flow.
    """

    def __init__(self, model="llama3"):
        self.model = model

    def build_prompt(self, state):
        """
        Build a prompt from one network flow.
        """

        prompt = f"""
You are an expert cybersecurity analyst.

Analyze the following network traffic.

Flow Duration: {state['Flow Duration']}
Destination Port: {state['Destination Port']}
Total Forward Packets: {state['Total Fwd Packets']}
Total Backward Packets: {state['Total Backward Packets']}
Flow Bytes/s: {state['Flow Bytes/s']}
Flow Packets/s: {state['Flow Packets/s']}
Average Packet Size: {state['Average Packet Size']}
SYN Flag Count: {state['SYN Flag Count']}
ACK Flag Count: {state['ACK Flag Count']}

Estimate how suspicious this traffic flow is.

Return ONLY ONE floating-point number
between 0.0 and 1.0.

Examples:

0.0
0.42
0.87
1.0

Do not explain your answer.
"""

        return prompt

    def evaluate(self, state):
        """
         Evaluate one network flow.

         Returns
         -------
        float
            Suspiciousness score Φ(s)
        """
        prompt = self.build_prompt(state)

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response["message"]["content"].strip()
        print("LLM Response:", text)

        match = re.search(r"\d*\.?\d+", text)
        if match:
            score = float(match.group())
            score = max(0.0, min(1.0, score))
            return score

        return 0.5

if __name__ == "__main__":

    sample_state = {

        "Flow Duration": 125000,
        "Destination Port": 80,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 4,
        "Flow Bytes/s": 180000,
        "Flow Packets/s": 32,
        "Average Packet Size": 300,
        "SYN Flag Count": 2,
        "ACK Flag Count": 8
    }

    critic = LLMCritic()

    score = critic.evaluate(sample_state)

    print("Suspiciousness Score:", score)