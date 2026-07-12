class PromptGenerator:
    """
    Generates prompts for the LLM Security Critic.
    """

    def __init__(self):
        pass

    def generate(self, state):
        """
        Convert one network flow into a prompt.

        Parameters
        ----------
        state : dict
            Dictionary containing one network flow.

        Returns
        -------
        prompt : str
        """

        prompt = f"""
            You are an expert cybersecurity analyst.

            Analyze the following network flow.

            Flow Duration: {state['Flow Duration']}
            Destination Port: {state['Destination Port']}
            Total Forward Packets: {state['Total Fwd Packets']}
            Total Backward Packets: {state['Total Bwd Packets']}
            Total Length of Forward Packets: {state['Total Length of Fwd Packets']}
            Total Length of Backward Packets: {state['Total Length of Bwd Packets']}
            Flow Bytes/s: {state['Flow Bytes/s']}
            Flow Packets/s: {state['Flow Packets/s']}
            SYN Flag Count: {state['SYN Flag Count']}
            ACK Flag Count: {state['ACK Flag Count']}
            Average Packet Size: {state['Average Packet Size']}

            Estimate how suspicious this traffic flow is.

            Return ONLY one floating-point number between 0.0 and 1.0.

            Examples:

            0.0 = Definitely Benign

            0.5 = Suspicious

            1.0 = Definitely Malicious
            """

        return prompt


if __name__ == "__main__":

    sample_state = {
        "Flow Duration": 125000,
        "Destination Port": 80,
        "Total Fwd Packets": 10,
        "Total Bwd Packets" : 4,
        "Total Length of Fwd Packets": 2000,
        "Total Length of Bwd Packets": 500,
        "Flow Bytes/s": 180000,
        "Flow Packets/s": 32,
        "SYN Flag Count": 2,
        "ACK Flag Count": 8,
        "Average Packet Size": 300
    }

    generator = PromptGenerator()

    prompt = generator.generate(sample_state)

    print(prompt)