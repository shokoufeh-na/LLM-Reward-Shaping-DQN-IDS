# src/prompt_generator.py


class PromptGenerator:
    """
    Generates prompts for the LLM Security Critic
    using selected CICIDS2017 network-flow features.
    """

    def __init__(self):

        self.selected_features = [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Max",
            "Fwd Packet Length Mean",
            "Bwd Packet Length Max",
            "Bwd Packet Length Mean",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Mean",
            "Bwd IAT Mean",
            "SYN Flag Count",
            "ACK Flag Count",
            "RST Flag Count",
            "PSH Flag Count",
            "FIN Flag Count",
            "Average Packet Size",
            "Active Mean",
            "Idle Mean",
        ]

    def generate(self, state):
        """
        Convert one network flow into a prompt.

        Parameters
        ----------
        state : dict
            Dictionary containing one network flow
            with original-scale feature values.

        Returns
        -------
        prompt : str
        """

        feature_lines = []

        for feature in self.selected_features:

            if feature in state:

                value = state[feature]

                feature_lines.append(
                    f"{feature}: {value}"
                )

        feature_text = "\n".join(feature_lines)

        prompt = f"""
You are an expert cybersecurity analyst specializing in
network intrusion detection.

Analyze the following network flow based on its traffic
statistics.

Network Flow Features:

{feature_text}

Estimate the security risk of this network flow based only
on the provided network traffic features.

Return ONLY one floating-point number between 0.0 and 1.0.

The score represents:

0.0 = Strongly consistent with benign network traffic
0.5 = Uncertain or moderately suspicious network traffic
1.0 = Strongly consistent with malicious network traffic

Do not return explanations, labels, or additional text.
Return only the numerical score.
"""

        return prompt.strip()


if __name__ == "__main__":

    sample_state = {
        "Destination Port": 80,
        "Flow Duration": 125000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 4,
        "Total Length of Fwd Packets": 2000,
        "Total Length of Bwd Packets": 500,
        "Fwd Packet Length Max": 500,
        "Fwd Packet Length Mean": 200,
        "Bwd Packet Length Max": 250,
        "Bwd Packet Length Mean": 125,
        "Flow Bytes/s": 180000,
        "Flow Packets/s": 32,
        "Flow IAT Mean": 4000,
        "Flow IAT Std": 1200,
        "Flow IAT Max": 10000,
        "Flow IAT Min": 20,
        "Fwd IAT Mean": 5000,
        "Bwd IAT Mean": 6000,
        "SYN Flag Count": 2,
        "ACK Flag Count": 8,
        "RST Flag Count": 0,
        "PSH Flag Count": 2,
        "FIN Flag Count": 1,
        "Average Packet Size": 300,
        "Active Mean": 15000,
        "Idle Mean": 2000,
    }

    generator = PromptGenerator()

    prompt = generator.generate(
        sample_state
    )

    print(prompt)